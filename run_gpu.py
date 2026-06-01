"""
run_gpu.py  —— GPU 加速分类管线（稠密马尔可夫模型 + 置换检验）

用法：
    # 使用 GPU 对 reads.fa 进行分类（k=6）
    python run_gpu.py --genomes ./proj1/genomes --reads ./proj1/reads.fa --k 6

    # 同时评估测试集准确率
    python run_gpu.py --genomes ./proj1/genomes --reads ./proj1/reads.fa \\
                      --test ./proj1/test/test.fa --map ./proj1/test/seq_id.map --k 6

    # 仅评估测试集准确率
    python run_gpu.py --genomes ./proj1/genomes --test ./proj1/test/test.fa --map ./proj1/test/seq_id.map --k 6

    # 指定 batch size 和置换检验参数
    python run_gpu.py --genomes ./proj1/genomes --reads ./proj1/reads.fa --k 6 \\
                      --batch-size 1024 --z-threshold 2.58 --n-permutations 50

    # 扫描 k=3..10 寻找最优 k（GPU 稠密模型）
    python run_gpu.py --genomes ./proj1/genomes --test ./proj1/test.fa --map ./proj1/seq_id.map --sweep

说明：
    1. GPU 分类仅支持稠密模型（k ≤ 11）。若 k > 11 或模型超出显存限制，程序会打印警告并跳过分类。
    2. 模型训练仍在 CPU 上进行（训练时间不计入分类时间）。

"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (K_DENSE_THRESHOLD, Z_SCORE_THRESHOLD, N_PERMUTATIONS,
                   ID2NAME)
from markov_train import train_all_genomes, save_models, load_models
from classify_gpu import classify_all_gpu
from data_loader import read_fasta


def train_or_load_models(genome_dir, k, cache_dir="./models"):
    """训练模型或从缓存加载。GPU 分类仅支持稠密模型，因此固定使用 'dense' 后缀缓存文件。"""
    suffix = "dense"
    cache_path = os.path.join(cache_dir, f"models_{suffix}_k{k}.pkl")
    if os.path.exists(cache_path):
        print(f"[CACHE] 从 {cache_path} 加载模型")
        return load_models(cache_path)
    print(f"[TRAIN] 训练 k={k} 稠密模型...")
    models = train_all_genomes(genome_dir, k)
    os.makedirs(cache_dir, exist_ok=True)
    save_models(models, cache_path)
    print(f"       模型已保存至 {cache_path}")
    return models


def print_statistics(labels, models):
    """打印分类统计结果）。"""
    genome_ids = [m['genome_id'] for m in models]
    assigned_mask = labels >= 0
    total = len(labels)
    n_assigned = int(assigned_mask.sum())
    n_unassigned = total - n_assigned

    print("\n" + "=" * 60)
    print("分类统计结果")
    print("=" * 60)
    print(f"\n  (1) 序列总数: {total}")
    print(f"  (2) 可分配到基因组的 reads 数量: {n_assigned}")
    print(f"  (3) 无法分配的 reads 数量: {n_unassigned}")
    if total > 0:
        print(f"      分配率: {n_assigned / total * 100:.1f}%")

    assigned_labels = labels[assigned_mask]
    group_counts = Counter(int(l) for l in assigned_labels)

    print(f"\n  (4) 至少分配 j 个序列的组数:")
    print(f"      {'j':<8} {'组数':<8}")
    print(f"      {'-'*16}")
    for j in [1, 5, 10, 50]:
        n_groups = sum(1 for c in group_counts.values() if c >= j)
        print(f"      {j:<8} {n_groups:<8}")

    print(f"\n  各组分配详情:")
    print(f"      {'基因组 ID':<20} {'名称':<60} {'分配数':<8}")
    print(f"      {'-'*88}")
    for idx, gid in enumerate(genome_ids):
        count = group_counts.get(idx, 0)
        name = ID2NAME.get(gid, gid)
        name_short = name[:57] + "..." if len(name) > 60 else name
        print(f"      {gid:<20} {name_short:<60} {count:<8}")

    print(f"\n  (5) 分配 ≥11 条序列的组:")
    major_groups = [(idx, count) for idx, count in group_counts.items() if count >= 10]
    major_groups.sort(key=lambda x: -x[1])
    if major_groups:
        print(f"      {'基因组 ID':<20} {'分配数':<8} {'占比':<8}")
        print(f"      {'-'*36}")
        for idx, count in major_groups:
            gid = genome_ids[idx]
            pct = count / n_assigned * 100 if n_assigned > 0 else 0
            print(f"      {gid:<20} {count:<8} {pct:<8.1f}%")
    else:
        print("      无任何组分配 ≥11 条序列。")
    return group_counts


def evaluate_accuracy(labels, read_ids, map_file, models):
    """使用 seq_id.map 计算分类准确率。自动兼容物种名和 genome_id 两种格式。"""
    true_mapping = {}
    with open(map_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                true_mapping[parts[0]] = parts[1]

    gid_to_idx = {m['genome_id']: i for i, m in enumerate(models)}
    name_to_gid = {v: k for k, v in ID2NAME.items()}

    correct = 0
    total_mapped = 0
    for rid, label in zip(read_ids, labels):
        raw_val = true_mapping.get(rid)
        if raw_val is None:
            continue
        true_idx = gid_to_idx.get(raw_val)
        if true_idx is None:
            gid = name_to_gid.get(raw_val)
            if gid is not None:
                true_idx = gid_to_idx.get(gid)
        if true_idx is None:
            continue
        total_mapped += 1
        if label == true_idx:
            correct += 1

    acc = correct / total_mapped * 100 if total_mapped > 0 else 0.0
    return acc, correct, total_mapped


def sweep_k_gpu(genome_dir, test_fa, map_file,
                k_min=3, k_max=K_DENSE_THRESHOLD,
                cache_dir="./models",
                z_threshold=Z_SCORE_THRESHOLD,
                n_permutations=N_PERMUTATIONS,
                batch_size=2048):
    """
    在 GPU 上扫描 k 值范围，返回 (results, best_k, best_acc)。
    results 为列表，每项 (k, acc, correct, total, elapsed)。
    """
    print(f"\n[INFO] GPU 扫描 k={k_min}..{k_max} ...")
    test_reads = read_fasta(test_fa, keep_full_header=False)
    test_ids = [r[0] for r in test_reads]
    test_seqs = [r[1] for r in test_reads]

    results = []
    best_k = None
    best_acc = -1.0

    for k in range(k_min, k_max + 1):
        t0 = time.time()
        models = train_or_load_models(genome_dir, k, cache_dir)
        labels = classify_all_gpu(test_seqs, models, k,
                                  z_threshold, n_permutations, batch_size)
        elapsed = time.time() - t0

        # 如果全为 -1 说明 GPU 分类失败，跳过统计
        if (labels == -1).all() and len(test_seqs) > 0:
            print(f"  k={k:2d}: GPU 分类返回全 -1，跳过评估。")
            continue

        acc, correct, total = evaluate_accuracy(labels, test_ids, map_file, models)
        results.append((k, acc, correct, total, elapsed))
        print(f"  k={k:2d}: 准确率={acc:.2f}% ({correct}/{total}), 耗时 {elapsed:.1f}s")
        if acc > best_acc:
            best_acc = acc
            best_k = k

        # 主动释放显存，避免累积
        del models
        torch.cuda.empty_cache()

    return results, best_k, best_acc


def print_sweep_results(results, best_k, best_acc, title="GPU k 值扫描结果汇总"):
    """格式化打印扫描结果。"""
    if not results:
        print("\n[WARN] 无有效结果。")
        return
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"  {'k':<6} {'准确率':<12} {'正确/总数':<16} {'耗时':<10}")
    print(f"  {'-'*44}")
    for k, acc, correct, total, elapsed in results:
        marker = " <-- BEST" if k == best_k else ""
        print(f"  {k:<6} {acc:<11.2f}% {correct}/{total:<14} {elapsed:<9.1f}s{marker}")
    print(f"\n  最优 k = {best_k}, 准确率 = {best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="GPU 加速马尔可夫模型分类")
    parser.add_argument("--genomes", default="./proj1/genomes", help="参考基因组目录")
    parser.add_argument("--reads", default=None, help="待分类 reads 文件")
    parser.add_argument("--test", default=None, help="测试集文件（用于评估准确率）")
    parser.add_argument("--map", default=None, help="seq_id.map 文件路径（配合 --test 使用）")
    parser.add_argument("--k", type=int, default=6, help="模型阶数 (默认 6, 仅支持 ≤11)")
    parser.add_argument("--sweep", action="store_true",
                        help=f"扫描 k=3..{K_DENSE_THRESHOLD}，在 GPU 上寻找最优 k")
    parser.add_argument("--cache-dir", default="./models", help="模型缓存目录")
    parser.add_argument("--z-threshold", type=float, default=Z_SCORE_THRESHOLD,
                        help=f"置换检验 z 值阈值 (默认 {Z_SCORE_THRESHOLD})")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help=f"随机打乱次数 (默认 {N_PERMUTATIONS})")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="GPU 批处理大小 (默认 2048)")
    args = parser.parse_args()

    # ----- 扫描模式 -----
    if args.sweep:
        if not args.test or not args.map:
            print("[ERROR] --sweep 需要 --test 和 --map 参数")
            sys.exit(1)
        results, best_k, best_acc = sweep_k_gpu(
            args.genomes, args.test, args.map,
            k_min=3, k_max=K_DENSE_THRESHOLD,
            cache_dir=args.cache_dir,
            z_threshold=args.z_threshold,
            n_permutations=args.n_permutations,
            batch_size=args.batch_size)
        print_sweep_results(results, best_k, best_acc)
        return

    # ----- 普通分类/评估模式 -----
    if not args.reads and not args.test:
        print("[ERROR] 请至少指定 --reads 或 --test")
        sys.exit(1)

    if args.k > K_DENSE_THRESHOLD:
        print(f"[ERROR] GPU 分类仅支持 k ≤ {K_DENSE_THRESHOLD}（稠密模型），当前 k={args.k}")
        sys.exit(1)

    print(f"[INFO] k={args.k} 阶马尔可夫模型 (GPU 加速)")
    print(f"[INFO] 置换检验: z 阈值={args.z_threshold}, 打乱次数={args.n_permutations}")
    print(f"[INFO] GPU 批处理大小: {args.batch_size}")

    models = train_or_load_models(args.genomes, args.k, args.cache_dir)

    if args.reads:
        print(f"\n[GPU] 读取并分类 reads.fa ...")
        reads = read_fasta(args.reads, keep_full_header=False)
        read_ids = [r[0] for r in reads]
        read_seqs = [r[1] for r in reads]
        t0 = time.time()
        labels = classify_all_gpu(read_seqs, models, args.k,
                                  args.z_threshold, args.n_permutations,
                                  args.batch_size)
        elapsed = time.time() - t0
        if (labels == -1).all() and len(read_seqs) > 0:
            print("[WARN] GPU 分类返回全 -1，可能因显存不足或模型不符，请检查日志。")
        else:
            print(f"[GPU] 分类完成，耗时 {elapsed:.2f}s")
        print_statistics(labels, models)

    if args.test:
        print(f"\n[GPU] 读取并分类测试集 ...")
        test_reads = read_fasta(args.test, keep_full_header=False)
        test_ids = [r[0] for r in test_reads]
        test_seqs = [r[1] for r in test_reads]
        t0 = time.time()
        labels = classify_all_gpu(test_seqs, models, args.k,
                                  args.z_threshold, args.n_permutations,
                                  args.batch_size)
        elapsed = time.time() - t0
        if (labels == -1).all() and len(test_seqs) > 0:
            print("[WARN] GPU 分类返回全 -1，跳过评估。")
        else:
            print(f"[GPU] 分类完成，耗时 {elapsed:.2f}s")
            if args.map:
                acc, correct, total = evaluate_accuracy(labels, test_ids, args.map, models)
                print(f"  测试准确率: {acc:.2f}% ({correct}/{total})")
            else:
                print("[WARN] 未提供 --map，跳过准确率评估。")


    del models
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()