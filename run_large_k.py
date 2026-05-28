"""
run_large_k.py  —— 大 k (k > 12) 马尔可夫模型分类管线（稀疏字典形式）

用法：
    # 仅分类 reads.fa，输出统计（问题 1-5）
    python run_large_k.py --genomes ./proj1/genomes --reads ./proj1/reads.fa --k 15

    # 仅在测试集上评估准确率（问题 6）
    python run_large_k.py --genomes ./proj1/genomes \\
                          --test ./proj1/test.fa --map ./proj1/seq_id.map --k 15

    # 分类 + 评估一起做
    python run_large_k.py --genomes ./proj1/genomes --reads ./proj1/reads.fa \\
                          --test ./proj1/test.fa --map ./proj1/seq_id.map --k 15

    # 扫描 k=13..20 找最优 k（纯稀疏字典）
    python run_large_k.py --genomes ./proj1/genomes \\
                          --test ./proj1/test.fa --map ./proj1/seq_id.map --sweep

    # 全范围扫描 k=3..20（稠密→稀疏自动切换，有内存保护）
    python run_large_k.py --genomes ./proj1/genomes --test ./proj1/test.fa --map ./proj1/seq_id.map --full-sweep

    # 强制小 k 也用稀疏字典（节省内存）
    python run_large_k.py --genomes ./proj1/genomes --reads ./proj1/reads.fa \\
                          --k 10 --dense-threshold 8

说明：
    当 k > dense_threshold 时使用稀疏字典存储转移概率，大幅降低内存占用。
    字典中仅保存训练数据中实际出现的 k-mer 上下文。
    未出现的上下文在打分时使用 default_log_prob = log(0.25) 回退。
"""

import os
import sys
import argparse
import time
import psutil
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import K_DENSE_THRESHOLD, THRESHOLD_FACTOR, ID2NAME
from markov_train import train_all_genomes, save_models, load_models
from classify_cpu import classify_all, score_read
from data_loader import (
    read_fasta, load_seq_id_map, get_genome_files,
)


def estimate_dense_memory(k: int) -> float:
    """估算稠密矩阵内存（GB）。"""
    states = 4 ** k
    bytes_per_matrix = states * 4 * 8
    return bytes_per_matrix / (1024 ** 3)


def should_use_sparse(k: int, dense_threshold: int, safe_gb: float = 10.0) -> bool:
    """判断是否应该使用稀疏字典。"""
    if k > dense_threshold:
        return True
    est_gb = estimate_dense_memory(k)
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    if est_gb > available_gb * 0.8 or est_gb > safe_gb:
        return True
    return False


class Timer:
    """计时器，记录各阶段耗时。"""
    def __init__(self):
        self.records = {}
        self._start = {}

    def start(self, name: str):
        self._start[name] = time.time()

    def stop(self, name: str) -> float:
        elapsed = time.time() - self._start[name]
        self.records[name] = elapsed
        return elapsed

    def summary(self):
        print("\n" + "=" * 60)
        print("运行时间汇总")
        print("=" * 60)
        total = sum(self.records.values())
        for name, elapsed in self.records.items():
            pct = elapsed / total * 100 if total > 0 else 0
            print(f"  {name:<30} {elapsed:>8.2f}s  ({pct:>5.1f}%)")
        print(f"  {'-'*42}")
        print(f"  {'总计':<30} {total:>8.2f}s")


def train_or_load(genome_dir: str, k: int, cache_dir: str = "./models", timer: Timer = None):
    """训练模型或从缓存加载。自动选择稠密/稀疏。"""
    is_sparse = k > K_DENSE_THRESHOLD
    suffix = "sparse" if is_sparse else "dense"
    cache_path = os.path.join(cache_dir, f"models_{suffix}_k{k}.pkl")
    if os.path.exists(cache_path):
        if timer:
            timer.start("加载模型(缓存)")
        print(f"[CACHE] 命中！从 {cache_path} 加载{suffix}模型，跳过训练")
        models = load_models(cache_path)
        elapsed = timer.stop("加载模型(缓存)") if timer else 0
        if timer:
            print(f"[CACHE] 加载耗时 {elapsed:.1f}s，共 {len(models)} 个基因组")
        return models

    storage = "稀疏字典" if is_sparse else "稠密矩阵"
    print(f"[TRAIN] 未找到缓存，开始训练 k={k} 阶马尔可夫模型（{storage}）...")
    if timer:
        timer.start(f"训练(k={k})")
    models = train_all_genomes(genome_dir, k)
    elapsed = timer.stop(f"训练(k={k})") if timer else 0

    print(f"[TRAIN] 训练完成，耗时 {elapsed:.1f}s，共 {len(models)} 个基因组")
    for m in models:
        if m['type'] == 'sparse':
            n_entries = len(m['log_prob_dict'])
            print(f"       {m['genome_id']}: 稀疏字典 {n_entries} 个条目")
        else:
            print(f"       {m['genome_id']}: 稠密矩阵 4^{k}={4**k} 状态")
    save_models(models, cache_path)
    print(f"[TRAIN] 模型已保存至: {cache_path}")
    return models


def classify_reads_fasta(models, fasta_path: str, k: int, timer: Timer = None):
    """对 FASTA 文件中的所有 reads 进行分类。"""
    if timer:
        timer.start("load_reads")
    print(f"[INFO] 读取 {fasta_path} ...")
    reads = read_fasta(fasta_path, keep_full_header=False)
    read_ids = [r[0] for r in reads]
    read_seqs = [r[1] for r in reads]
    if timer:
        timer.stop("load_reads")

    if timer:
        timer.start("classify")
    print(f"[INFO] 对 {len(reads)} 条 reads 进行分类 (k={k})...")
    labels = classify_all(read_seqs, models, k)
    elapsed = timer.stop("classify") if timer else 0
    avg_time = elapsed / len(reads) * 1000 if len(reads) > 0 else 0
    print(f"[INFO] 分类完成，耗时 {elapsed:.1f}s (平均 {avg_time:.3f} ms/read)")
    return read_ids, read_seqs, labels


def print_statistics(labels, models):
    """打印分类统计信息（回答问题 1-5）。"""
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

    print(f"\n  (5) 分配 ≥10 条序列的组:")
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
        print("      无任何组分配 ≥10 条序列。")

    return group_counts


def evaluate_accuracy(labels, read_ids, map_file: str, models):
    """使用 seq_id.map / seq_id_numeric.map 评估分类准确率。
       自动识别两种格式：
         - 数值格式: "read_id genome_id"（如 0 NC_015656）
         - 原始格式: "read_id\\t物种全名"（如 0\\tFrankia symbiont...）
    """
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
        # 先尝试直接匹配 genome_id，再尝试物种名反向查表
        true_idx = gid_to_idx.get(raw_val)
        if true_idx is None:
            true_idx = gid_to_idx.get(name_to_gid.get(raw_val))
        if true_idx is None:
            continue
        total_mapped += 1
        if label == true_idx:
            correct += 1

    if total_mapped == 0:
        print(f"  [WARN] evaluate_accuracy: 未能匹配任何 read。请检查 map 文件格式。")
        print(f"          map 文件应包含 read_id + genome_id（如 seq_id_numeric.map）")
        print(f"          或 read_id + 物种全名（如原始 seq_id.map）")

    acc = correct / total_mapped * 100 if total_mapped > 0 else 0
    return acc, correct, total_mapped


def sweep_k(genome_dir: str, test_fasta: str, map_file: str,
            k_min: int = 13, k_max: int = 20, cache_dir: str = "./models") -> list:
    """扫描大 k 值范围（纯稀疏字典），返回 [(k, acc, correct, total, elapsed), ...]。"""
    print(f"\n[INFO] 扫描 k={k_min}..{k_max} （稀疏字典模式）...")
    results = []

    test_reads = read_fasta(test_fasta, keep_full_header=False)
    test_ids = [r[0] for r in test_reads]
    test_seqs = [r[1] for r in test_reads]

    for k in range(k_min, k_max + 1):
        t0 = time.time()
        models = train_or_load(genome_dir, k, cache_dir)
        labels = classify_all(test_seqs, models, k)
        acc, correct, total = evaluate_accuracy(labels, test_ids, map_file, models)
        elapsed = time.time() - t0
        results.append((k, acc, correct, total, elapsed))
        print(f"  k={k:2d}: 准确率={acc:.2f}% ({correct}/{total}), 耗时 {elapsed:.1f}s "
              f"[稀疏字典]")

    return results


def full_sweep(genome_dir: str, test_fasta: str, map_file: str,
               dense_threshold: int = K_DENSE_THRESHOLD,
               safe_gb: float = 10.0, cache_dir: str = "./models") -> list:
    """
    全范围扫描 k=3..20，自动切换稠密/稀疏，带内存保护。
    稠密部分：k ≤ min(dense_threshold, 内存安全上限)
    稀疏部分：其余 k 值
    """
    print("\n" + "=" * 60)
    print("[INFO] 全范围扫描 k=3..20")
    print(f"[INFO] 稠密/稀疏阈值: k={dense_threshold}, 安全内存: {safe_gb} GB")
    print("=" * 60)

    all_results = []
    test_reads = read_fasta(test_fasta, keep_full_header=False)
    test_ids = [r[0] for r in test_reads]
    test_seqs = [r[1] for r in test_reads]

    for k in range(3, 21):
        est_gb = estimate_dense_memory(k)
        available_gb = psutil.virtual_memory().available / (1024 ** 3)

        use_sparse = should_use_sparse(k, dense_threshold, safe_gb)
        mode = "稀疏字典" if use_sparse else "稠密矩阵"

        if use_sparse and k <= dense_threshold:
            print(f"\n  k={k:2d}: 稠密需 {est_gb:.2f} GB（可用 {available_gb:.2f} GB），"
                  f"自动切换稀疏")

        t0 = time.time()
        try:
            models = train_or_load(genome_dir, k, cache_dir)
            labels = classify_all(test_seqs, models, k)
            acc, correct, total = evaluate_accuracy(labels, test_ids, map_file, models)
            elapsed = time.time() - t0
            all_results.append((k, acc, correct, total, elapsed, mode))
            print(f"  k={k:2d}: 准确率={acc:.2f}% ({correct}/{total}), "
                  f"耗时 {elapsed:.1f}s [{mode}]")
        except MemoryError:
            print(f"  k={k:2d}: [ERROR] 内存不足，跳过")
            continue

    return all_results


def print_sweep_results(results, title="k 值扫描结果汇总"):
    """格式化打印扫描结果。"""
    if not results:
        print("\n[WARN] 无有效结果。")
        return None, None

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"  {'k':<6} {'准确率':<12} {'正确/总数':<16} {'耗时':<10} {'模式':<10}")
    print(f"  {'-'*54}")

    best_k, best_acc = None, -1
    max_acc = max(r[1] for r in results)
    for row in results:
        k, acc, correct, total, elapsed = row[0], row[1], row[2], row[3], row[4]
        mode = row[5] if len(row) > 5 else "?"
        marker = " <--" if acc == max_acc else ""
        print(f"  {k:<6} {acc:<11.2f}% {correct}/{total:<14} {elapsed:<9.1f}s "
              f"{mode:<10}{marker}")
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"\n  最优 k = {best_k}, 准确率 = {best_acc:.2f}%")
    return best_k, best_acc


def main():
    parser = argparse.ArgumentParser(description="大 k 马尔可夫模型分类（稀疏字典）")
    parser.add_argument("--genomes", default="./proj1/genomes", help="参考基因组目录")
    parser.add_argument("--reads", default=None, help="待分类 reads 文件（可选，仅做分类统计时使用）")
    parser.add_argument("--test", default=None, help="测试集文件（用于评估准确率）")
    parser.add_argument("--map", default=None, help="seq_id.map 文件路径（配合 --test 使用）")
    parser.add_argument("--k", type=int, default=15, help="模型阶数 (默认 15)")
    parser.add_argument("--sweep", action="store_true", help="扫描 k=13..20 寻找最优 k")
    parser.add_argument("--full-sweep", action="store_true",
                        help="全范围扫描 k=3..20（稠密+稀疏，带内存保护）")
    parser.add_argument("--cache-dir", default="./models", help="模型缓存目录")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_FACTOR,
                        help="分类阈值（默认 -5.0）")
    parser.add_argument("--dense-threshold", type=int, default=K_DENSE_THRESHOLD,
                        help=f"稠密/稀疏切换阈值 (默认 {K_DENSE_THRESHOLD})")
    parser.add_argument("--safe-memory-gb", type=float, default=10.0,
                        help="稠密矩阵安全内存上限 GB (默认 10.0)")
    args = parser.parse_args()

    timer = Timer()

    # ---- 全范围扫描模式 (k=3..20) ----
    if args.full_sweep:
        if not args.test or not args.map:
            print("[ERROR] --full-sweep 需要 --test 和 --map 参数")
            sys.exit(1)

        timer.start("full_sweep")
        results = full_sweep(args.genomes, args.test, args.map,
                             dense_threshold=args.dense_threshold,
                             safe_gb=args.safe_memory_gb,
                             cache_dir=args.cache_dir)
        timer.stop("full_sweep")
        print_sweep_results(results, "全范围 k 值扫描结果 (k=3..20)")
        timer.summary()
        return

    # ---- 大 k 扫描模式 ----
    if args.sweep:
        if not args.test or not args.map:
            print("[ERROR] --sweep 需要 --test 和 --map 参数")
            sys.exit(1)

        timer.start("sweep")
        results = sweep_k(args.genomes, args.test, args.map,
                          k_min=max(13, args.dense_threshold + 1), k_max=20,
                          cache_dir=args.cache_dir)
        timer.stop("sweep")
        print_sweep_results(results, "大 k 值扫描结果汇总")
        timer.summary()
        return

    # ---- 标准分类模式 ----
    if not args.reads and not args.test:
        print("[ERROR] 请至少指定 --reads 或 --test")
        sys.exit(1)

    use_sparse = should_use_sparse(args.k, args.dense_threshold, args.safe_memory_gb)
    storage = "稀疏字典" if use_sparse else "稠密矩阵"
    print(f"[INFO] 使用 k={args.k} 阶马尔可夫模型（{storage}）")

    timer.start("total")
    models = train_or_load(args.genomes, args.k, args.cache_dir, timer)

    # 分类 reads.fa（仅当指定了 --reads）
    if args.reads:
        read_ids, read_seqs, labels = classify_reads_fasta(models, args.reads, args.k, timer)
        timer.start("statistics")
        group_counts = print_statistics(labels, models)
        timer.stop("statistics")

    # 测试集评估
    if args.test and args.map:
        timer.start("evaluate")
        print(f"\n[INFO] 使用测试集评估准确率...")
        test_reads = read_fasta(args.test, keep_full_header=False)
        test_ids = [r[0] for r in test_reads]
        test_seqs = [r[1] for r in test_reads]
        test_labels = classify_all(test_seqs, models, args.k)
        acc, correct, total = evaluate_accuracy(test_labels, test_ids, args.map, models)
        print(f"  测试准确率: {acc:.2f}% ({correct}/{total})")
        timer.stop("evaluate")
    elif args.test and not args.map:
        print("[WARN] --test 需要配合 --map 使用，跳过评估")

    timer.stop("total")
    timer.summary()


if __name__ == "__main__":
    main()
