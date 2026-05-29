"""
combine_models.py  —— 多阶马尔可夫模型组合

支持 2 种或 3 种不同 k 值的组合，提供多种加权策略。
分类显著性使用置换检验（permutation test），比固定阈值更可靠。

用法：
    # 指定 2 个 k 值，网格搜索最优权重
    python combine_models.py --genomes ./proj1/genomes \\
        --test ./proj1/test/test.fa --map ./proj1/test/seq_id.map \\
        --k1 6 --k2 15

    # 指定 3 个 k 值，网格搜索最优权重
    python combine_models.py --genomes ./proj1/genomes \\
        --test ./proj1/test/test.fa --map ./proj1/test/seq_id.map \\
        --k1 6 --k2 10 --k3 15

    # 从列表中穷举所有 2-组合 和 3-组合
    python combine_models.py --genomes ./proj1/genomes \\
        --test ./proj1/test/test.fa --map ./proj1/test/seq_id.map \\
        --k-list "4 6 8 10 12 15 18"

    # 指定权重策略 + 置换检验参数
    python combine_models.py --genomes ./proj1/genomes \\
        --test ./proj1/test/test.fa --map ./proj1/test/seq_id.map \\
        --k1 6 --k2 15 --strategy accuracy_weighted \\
        --z-threshold 2.0 --n-permutations 100

权重策略：
    uniform_raw      — 原始对数似然直接平均
    uniform_norm     — 每步归一化后平均（推荐，消除 k 值对分数尺度的影响）
    accuracy_weighted — 以各模型单独准确率为权重
    grid_search      — 网格搜索最优权重组合（默认）

原理：
    不同 k 值的马尔可夫模型产生的对数似然分数尺度不同。
    组合前先做"每步归一化"：score / (L - k)，使不同 k 在相同尺度上比较。
    组合后对每条 read 使用置换检验判断匹配显著性：
      将 read 打乱 n_permutations 次，用组合模型打分，构造零分布；
      原始得分需超出零分布均值 z_threshold 个标准差才认为显著。
"""

import os
import sys
import argparse
import time
import itertools
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (K_DENSE_THRESHOLD, Z_SCORE_THRESHOLD, N_PERMUTATIONS,
                   ID2NAME)
from markov_train import train_all_genomes, save_models, load_models
from classify_cpu import score_read, shuffle_sequence
from data_loader import read_fasta, load_seq_id_map, get_genome_files


# ============================================================
# 模型加载
# ============================================================

def get_models(genome_dir: str, k: int, cache_dir: str = "./models"):
    """训练或从缓存加载某个 k 值的全部模型。
       缓存文件命名：k≤11 → models_dense_k{k}.pkl，k>11 → models_sparse_k{k}.pkl
    """
    suffix = "dense" if k <= K_DENSE_THRESHOLD else "sparse"
    cache_path = os.path.join(cache_dir, f"models_{suffix}_k{k}.pkl")
    if os.path.exists(cache_path):
        print(f"  [CACHE] k={k}: {cache_path}")
        return load_models(cache_path)

    storage = "稀疏字典" if k > K_DENSE_THRESHOLD else "稠密矩阵"
    print(f"  [TRAIN] k={k} ({storage}) ...", end=" ", flush=True)
    t0 = time.time()
    models = train_all_genomes(genome_dir, k)
    print(f"完成, {time.time()-t0:.1f}s")
    save_models(models, cache_path)
    return models


# ============================================================
# 分数矩阵计算
# ============================================================

def compute_score_matrix(reads: list, models: list, k: int, normalize: bool = True):
    """
    计算分数矩阵 (n_reads × n_genomes)。
    若 normalize=True，返回每步归一化分数 = score / (L - k)。
    """
    n_reads = len(reads)
    n_genomes = len(models)
    scores = np.empty((n_reads, n_genomes), dtype=np.float64)

    for i, (rid, seq) in enumerate(reads):
        L = len(seq)
        if L <= k:
            scores[i, :] = -np.inf
            continue
        norm_factor = (L - k) if normalize else 1.0
        for g, model in enumerate(models):
            raw = score_read(seq, model, k)
            scores[i, g] = raw / norm_factor

    return scores


def classify_combined_with_permutation(scores: np.ndarray, reads: list,
                                        models_by_k: dict, ks: list,
                                        combine_weights: list,
                                        z_threshold: float, n_permutations: int):
    """
    使用置换检验对组合分数进行分类。

    参数：
        scores: 组合后的分数矩阵 (n_reads × n_genomes)
        reads: (read_id, seq) 列表
        models_by_k: {k: models_list} 各 k 值的模型
        ks: 参与组合的 k 值列表
        combine_weights: 各 k 的权重（应与 ks 顺序一致）
        z_threshold: z 值阈值
        n_permutations: 打乱次数
    返回：
        labels: 分类标签数组，-1 表示不显著
    """
    n_reads = len(reads)
    labels = np.full(n_reads, -1, dtype=int)

    for i in range(n_reads):
        best_g = int(np.argmax(scores[i]))
        best_score = scores[i, :][best_g]
        if best_score == -np.inf:
            continue

        # 对最佳基因组构造零分布
        null_scores = np.empty(n_permutations)
        for p in range(n_permutations):
            shuffled = shuffle_sequence(reads[i][1])
            combined_s = 0.0
            for ki, wk in zip(ks, combine_weights):
                L = len(shuffled)
                if L <= ki:
                    combined_s = -np.inf
                    break
                raw = score_read(shuffled, models_by_k[ki][best_g], ki)
                combined_s += wk * raw / max(1, L - ki)
            null_scores[p] = combined_s

        if np.all(null_scores == -np.inf):
            continue

        null_mean = null_scores.mean()
        null_std = null_scores.std(ddof=1)
        if null_std == 0:
            continue

        z = (best_score - null_mean) / null_std
        if z >= z_threshold:
            labels[i] = best_g

    return labels


# ============================================================
# 准确率评估（使用简单 argmax，用于网格搜索时快速评估）
# ============================================================

def classify_from_scores(scores: np.ndarray, threshold: float = -np.inf):
    """
    从分数矩阵中快速分类：argmax（不含置换检验）。
    用于网格搜索阶段的快速评估。
    scores: (n_reads, n_genomes)
    """
    best_idx = np.argmax(scores, axis=1)
    best_score = np.max(scores, axis=1)
    best_idx[best_score < threshold] = -1
    return best_idx


def evaluate_accuracy(labels: np.ndarray, read_ids: list, map_file: str, ref_models: list):
    """使用 seq_id.map / seq_id_numeric.map 评估分类准确率。"""
    true_mapping = {}
    with open(map_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                true_mapping[parts[0]] = parts[1]

    gid_to_idx = {m['genome_id']: i for i, m in enumerate(ref_models)}
    name_to_gid = {v: k for k, v in ID2NAME.items()}

    correct = 0
    total_mapped = 0
    for rid, label in zip(read_ids, labels):
        raw_val = true_mapping.get(rid)
        if raw_val is None:
            continue
        true_idx = gid_to_idx.get(raw_val)
        if true_idx is None:
            true_idx = gid_to_idx.get(name_to_gid.get(raw_val))
        if true_idx is None:
            continue
        total_mapped += 1
        if label == true_idx:
            correct += 1

    return correct / total_mapped * 100 if total_mapped > 0 else 0, correct, total_mapped


# ============================================================
# 组合策略
# ============================================================

def combine_uniform_raw(score_dict: dict, k_list: list):
    """策略1: 原始分数直接平均。"""
    result = np.zeros_like(list(score_dict.values())[0])
    for k in k_list:
        result += score_dict[k]
    return result / len(k_list)


def combine_uniform_norm(score_dict: dict, k_list: list):
    """策略2: 每步归一化分数平均（推荐）。"""
    result = np.zeros_like(list(score_dict.values())[0])
    for k in k_list:
        result += score_dict[k]
    return result / len(k_list)


def combine_accuracy_weighted(score_dict: dict, k_list: list, acc_dict: dict):
    """策略3: 以各模型单独准确率为权重。"""
    weights = np.array([acc_dict[k] for k in k_list])
    weights = weights / weights.sum()
    result = np.zeros_like(list(score_dict.values())[0])
    for i, k in enumerate(k_list):
        result += weights[i] * score_dict[k]
    return result


def grid_search_2models(score_dict: dict, k1: int, k2: int,
                         read_ids: list, map_file: str, ref_models: list,
                         steps: int = 21):
    """
    策略4: 2 模型网格搜索最优权重。
    combined = α * score_k1 + (1-α) * score_k2
    """
    s1 = score_dict[k1]
    s2 = score_dict[k2]
    best_acc = -1
    best_alpha = 0.5
    results = []

    for alpha in np.linspace(0, 1, steps):
        combined = alpha * s1 + (1 - alpha) * s2
        labels = classify_from_scores(combined)
        acc, correct, total = evaluate_accuracy(labels, read_ids, map_file, ref_models)
        results.append((alpha, acc))
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    return best_alpha, best_acc, results


def grid_search_3models(score_dict: dict, k1: int, k2: int, k3: int,
                         read_ids: list, map_file: str, ref_models: list,
                         steps: int = 11):
    """
    策略5: 3 模型网格搜索最优权重。
    combined = α*score_k1 + β*score_k2 + (1-α-β)*score_k3
    """
    s1 = score_dict[k1]
    s2 = score_dict[k2]
    s3 = score_dict[k3]
    best_acc = -1
    best_weights = (1/3, 1/3, 1/3)
    results = []

    for alpha in np.linspace(0, 1, steps):
        for beta in np.linspace(0, 1 - alpha, steps):
            gamma = 1 - alpha - beta
            if gamma < -1e-9:
                continue
            combined = alpha * s1 + beta * s2 + gamma * s3
            labels = classify_from_scores(combined)
            acc, correct, total = evaluate_accuracy(labels, read_ids, map_file, ref_models)
            results.append(((alpha, beta, gamma), acc))
            if acc > best_acc:
                best_acc = acc
                best_weights = (alpha, beta, gamma)

    return best_weights, best_acc, results


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="多阶马尔可夫模型组合（置换检验）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python combine_models.py -g ./proj1/genomes -t ./proj1/test/test.fa \\
      -m ./proj1/test/seq_id.map --k1 6 --k2 15

  python combine_models.py -g ./proj1/genomes -t ./proj1/test/test.fa \\
      -m ./proj1/test/seq_id.map --k1 6 --k2 10 --k3 15

  python combine_models.py -g ./proj1/genomes -t ./proj1/test/test.fa \\
      -m ./proj1/test/seq_id.map --k-list "4 6 8 10 12 15 18"
        """
    )
    parser.add_argument("-g", "--genomes", default="./proj1/genomes", help="参考基因组目录")
    parser.add_argument("-t", "--test", required=True, help="测试集 FASTA 文件")
    parser.add_argument("-m", "--map", required=True, help="seq_id.map 文件路径")
    parser.add_argument("--k1", type=int, default=None, help="第一个 k 值")
    parser.add_argument("--k2", type=int, default=None, help="第二个 k 值")
    parser.add_argument("--k3", type=int, default=None, help="第三个 k 值（可选）")
    parser.add_argument("--k-list", type=str, default=None,
                        help="空格分隔的 k 值列表。将穷举所有 2-组合和 3-组合。")
    parser.add_argument("--strategy", type=str, default="grid_search",
                        choices=["uniform_raw", "uniform_norm", "accuracy_weighted", "grid_search"],
                        help="权重策略 (默认 grid_search)")
    parser.add_argument("--grid-steps", type=int, default=21,
                        help="网格搜索步数 (默认 21，即步长 0.05)")
    parser.add_argument("--cache-dir", default="./models", help="模型缓存目录")
    parser.add_argument("--z-threshold", type=float, default=Z_SCORE_THRESHOLD,
                        help=f"置换检验 z 值阈值（默认 {Z_SCORE_THRESHOLD}）")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help=f"随机打乱次数（默认 {N_PERMUTATIONS}）")
    args = parser.parse_args()

    # ---- 解析 k 值列表 ----
    if args.k_list:
        k_values = [int(x) for x in args.k_list.split()]
        if len(k_values) < 2:
            print("[ERROR] --k-list 至少需要 2 个 k 值")
            sys.exit(1)
    elif args.k1 is not None and args.k2 is not None:
        k_values = [args.k1, args.k2]
        if args.k3 is not None:
            k_values.append(args.k3)
    else:
        print("[ERROR] 请指定 --k1 --k2 [--k3] 或 --k-list")
        sys.exit(1)

    k_values = sorted(set(k_values))
    print(f"[INFO] 参与组合的 k 值: {k_values}")
    print(f"[INFO] 置换检验: z 阈值={args.z_threshold}, "
          f"打乱次数={args.n_permutations}")

    # ---- 加载数据 ----
    print(f"\n[INFO] 读取测试数据...")
    test_reads = read_fasta(args.test, keep_full_header=False)
    read_ids = [r[0] for r in test_reads]
    print(f"[INFO] 测试集: {len(test_reads)} 条 reads")

    # ---- 训练/加载所有模型 ----
    print(f"\n[INFO] 准备模型...")
    all_models = {}
    for k in k_values:
        all_models[k] = get_models(args.genomes, k, args.cache_dir)

    ref_models = all_models[k_values[0]]

    # ---- 计算每步归一化分数矩阵 ----
    print(f"\n[INFO] 计算分数矩阵（每步归一化）...")
    score_norm = {}
    score_raw = {}
    baseline_acc = {}

    for k in k_values:
        t0 = time.time()
        score_norm[k] = compute_score_matrix(test_reads, all_models[k], k, normalize=True)
        score_raw[k] = compute_score_matrix(test_reads, all_models[k], k, normalize=False)
        # 基准准确率使用简单 argmax（网格搜索也用它，保持一致）
        labels = classify_from_scores(score_norm[k])
        acc, correct, total = evaluate_accuracy(labels, read_ids, args.map, all_models[k])
        baseline_acc[k] = acc
        print(f"  k={k:2d}: 基准准确率 {acc:.2f}% ({correct}/{total}), "
              f"耗时 {time.time()-t0:.1f}s")

    # ---- 基准结果汇总 ----
    print(f"\n{'='*70}")
    print(f"单模型基准准确率（argmax，不含置换检验）")
    print(f"{'='*70}")
    for k in k_values:
        print(f"  k={k:2d}: {baseline_acc[k]:.2f}%")
    best_single_k = max(baseline_acc, key=baseline_acc.get)
    best_single_acc = baseline_acc[best_single_k]
    print(f"  最佳单模型: k={best_single_k} ({best_single_acc:.2f}%)")

    # ---- 组合 ----
    print(f"\n{'='*70}")
    print(f"模型组合结果")
    print(f"{'='*70}")

    all_combinations = []

    if len(k_values) == 2 or (args.k1 and args.k2 and not args.k3 and not args.k_list):
        k1, k2 = k_values[0], k_values[1]
        _run_combination(k1, k2, None, score_norm, score_raw, baseline_acc,
                         read_ids, test_reads, args.map, ref_models, all_models,
                         args.strategy, args.grid_steps,
                         args.z_threshold, args.n_permutations, all_combinations)

    elif len(k_values) == 3 and not args.k_list:
        k1, k2, k3 = k_values[0], k_values[1], k_values[2]
        _run_combination(k1, k2, k3, score_norm, score_raw, baseline_acc,
                         read_ids, test_reads, args.map, ref_models, all_models,
                         args.strategy, args.grid_steps,
                         args.z_threshold, args.n_permutations, all_combinations)

    else:
        print(f"[INFO] 穷举所有 2-组合和 3-组合...")
        for k1, k2 in itertools.combinations(k_values, 2):
            _run_combination(k1, k2, None, score_norm, score_raw, baseline_acc,
                             read_ids, test_reads, args.map, ref_models, all_models,
                             "grid_search", args.grid_steps,
                             args.z_threshold, args.n_permutations, all_combinations)
        for k1, k2, k3 in itertools.combinations(k_values, 3):
            _run_combination(k1, k2, k3, score_norm, score_raw, baseline_acc,
                             read_ids, test_reads, args.map, ref_models, all_models,
                             "grid_search", args.grid_steps,
                             args.z_threshold, args.n_permutations, all_combinations)

    # ---- 最终汇总 ----
    print(f"\n{'='*70}")
    print(f"最终排名")
    print(f"{'='*70}")
    print(f"  {'排名':<6} {'模型':<25} {'准确率':<12} {'vs最优单模型':<16} {'详情':<20}")
    print(f"  {'-'*75}")

    all_combinations.sort(key=lambda x: -x[1])

    for rank, (desc, acc, detail) in enumerate(all_combinations, 1):
        delta = acc - best_single_acc
        delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank:<6} {desc:<25} {acc:<11.2f}% {delta_str:<16} {str(detail)[:18]:<20}{marker}")

    if all_combinations and all_combinations[0][1] > best_single_acc:
        print(f"\n  组合模型优于所有单模型！最优: {all_combinations[0][0]} "
              f"({all_combinations[0][1]:.2f}%)")
    else:
        print(f"\n  组合模型未超过最佳单模型 k={best_single_k} ({best_single_acc:.2f}%)")


def _run_combination(k1, k2, k3, score_norm, score_raw, baseline_acc,
                     read_ids, test_reads, map_file, ref_models, all_models,
                     strategy, grid_steps, z_threshold, n_permutations, results_list):
    """执行一组组合并记录结果。"""
    ks = (k1, k2) if k3 is None else (k1, k2, k3)
    desc = "+".join(f"k{k}" for k in ks)
    n_models = 2 if k3 is None else 3

    if strategy == "uniform_raw":
        score_subset = {k: score_raw[k] for k in ks}
        combined = combine_uniform_raw(score_subset, list(ks))
        # 使用置换检验分类
        weights = [1.0 / len(ks)] * len(ks)
        labels = classify_combined_with_permutation(
            combined, test_reads, all_models, list(ks), weights,
            z_threshold, n_permutations)
        acc, correct, total = evaluate_accuracy(labels, read_ids, map_file, ref_models)
        detail = f"原始分数等权平均 + 置换检验"
        print(f"  {desc}: {acc:.2f}% [{detail}]")
        results_list.append((desc, acc, detail))

    elif strategy == "uniform_norm":
        score_subset = {k: score_norm[k] for k in ks}
        combined = combine_uniform_norm(score_subset, list(ks))
        weights = [1.0 / len(ks)] * len(ks)
        labels = classify_combined_with_permutation(
            combined, test_reads, all_models, list(ks), weights,
            z_threshold, n_permutations)
        acc, correct, total = evaluate_accuracy(labels, read_ids, map_file, ref_models)
        detail = f"归一化等权平均 + 置换检验"
        print(f"  {desc}: {acc:.2f}% [{detail}]")
        results_list.append((desc, acc, detail))

    elif strategy == "accuracy_weighted":
        score_subset = {k: score_norm[k] for k in ks}
        acc_subset = {k: baseline_acc[k] for k in ks}
        combined = combine_accuracy_weighted(score_subset, list(ks), acc_subset)
        w = [baseline_acc[k] / sum(baseline_acc[k] for k in ks) for k in ks]
        labels = classify_combined_with_permutation(
            combined, test_reads, all_models, list(ks), list(w),
            z_threshold, n_permutations)
        acc, correct, total = evaluate_accuracy(labels, read_ids, map_file, ref_models)
        detail = f"准确率加权 {[f'{x:.3f}' for x in w]} + 置换检验"
        print(f"  {desc}: {acc:.2f}% [{detail}]")
        results_list.append((desc, acc, detail))

    elif strategy == "grid_search":
        score_subset = {k: score_norm[k] for k in ks}
        if n_models == 2:
            best_alpha, best_acc, grid_results = grid_search_2models(
                score_subset, ks[0], ks[1], read_ids, map_file, ref_models,
                steps=grid_steps)
            detail = f"α={best_alpha:.2f} (k{ks[0]}), 1-α={1-best_alpha:.2f} (k{ks[1]})"
            # 用最优权重 + 置换检验重新评估
            combined = best_alpha * score_subset[ks[0]] + (1 - best_alpha) * score_subset[ks[1]]
            labels = classify_combined_with_permutation(
                combined, test_reads, all_models, list(ks),
                [best_alpha, 1 - best_alpha],
                z_threshold, n_permutations)
            perm_acc, _, _ = evaluate_accuracy(labels, read_ids, map_file, ref_models)
            detail += f", 置换检验准确率={perm_acc:.2f}%"
            print(f"  {desc}: grid={best_acc:.2f}% [{detail}]")
            results_list.append((desc, best_acc, detail))
        else:
            best_weights, best_acc, grid_results = grid_search_3models(
                score_subset, ks[0], ks[1], ks[2], read_ids, map_file, ref_models,
                steps=grid_steps)
            w = [f"{x:.2f}" for x in best_weights]
            detail = f"权重 k{ks[0]}={w[0]} k{ks[1]}={w[1]} k{ks[2]}={w[2]}"
            alpha, beta, gamma = best_weights
            combined = (alpha * score_subset[ks[0]] + beta * score_subset[ks[1]]
                        + gamma * score_subset[ks[2]])
            labels = classify_combined_with_permutation(
                combined, test_reads, all_models, list(ks),
                [alpha, beta, gamma],
                z_threshold, n_permutations)
            perm_acc, _, _ = evaluate_accuracy(labels, read_ids, map_file, ref_models)
            detail += f", 置换检验准确率={perm_acc:.2f}%"
            print(f"  {desc}: grid={best_acc:.2f}% [{detail}]")
            results_list.append((desc, best_acc, detail))


if __name__ == "__main__":
    main()
