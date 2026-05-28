"""
run_small_k.py  —— 小 k (k ≤ 12) 马尔可夫模型分类管线（稠密矩阵形式）

用法：
    # 仅分类 reads.fa，输出统计（问题 1-5）
    python run_small_k.py --genomes ./proj1/genomes --reads ./proj1/reads.fa --k 6

    # 仅在测试集上评估准确率（问题 6）
    python run_small_k.py --genomes ./proj1/genomes \\
                          --test ./proj1/test/test.fa --map ./proj1/test/seq_id.map --k 6

    # 分类 + 评估一起做
    python run_small_k.py --genomes ./proj1/genomes --reads ./proj1/reads.fa \\
                          --test ./proj1/test/test.fa --map ./proj1/test/seq_id.map --k 6

    # 扫描 k=3..12 找最优 k
    python run_small_k.py --genomes ./proj1/genomes \\
                          --test ./proj1/test/test.fa --map ./proj1/test/seq_id.map --sweep

    # 调整稠密/稀疏阈值和内存限制
    python run_small_k.py --genomes ./proj1/genomes --reads ./proj1/reads.fa \\
                          --k 6 --dense-threshold 10 --safe-memory-gb 0.5
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
    """估算稠密矩阵内存（GB）。4^k 状态 × 4 碱基 × 8 字节(float64)。"""
    states = 4 ** k
    bytes_per_matrix = states * 4 * 8
    gb = bytes_per_matrix / (1024 ** 3)
    return gb


def check_memory_safety(k: int, safe_gb: float = 10.0, auto_fallback: bool = True) -> bool:
    """
    检查 k 值是否能用稠密矩阵安全训练。
    返回 True 表示安全，False 表示不安全。
    若不安全且 auto_fallback=True，则报错退出。
    """
    est_gb = estimate_dense_memory(k)
    available_gb = psutil.virtual_memory().available / (1024 ** 3)

    print(f"[MEM] k={k}: 稠密矩阵估算 {est_gb:.2f} GB (4^{k} × 4 × 8 bytes)")
    print(f"[MEM] 系统可用内存: {available_gb:.2f} GB, 安全阈值: {safe_gb:.2f} GB")

    if est_gb > available_gb * 0.8:
        if auto_fallback:
            print(f"[WARN] 稠密矩阵内存需求 ({est_gb:.2f} GB) 超过系统可用内存 80%，")
            print(f"       自动切换到稀疏字典模式。请使用 run_large_k.py 或提高 --dense-threshold 后重试。")
            sys.exit(1)
        return False

    if est_gb > safe_gb:
        print(f"[WARN] 稠密矩阵估算 {est_gb:.2f} GB 超过安全阈值 {safe_gb:.2f} GB")
        if auto_fallback:
            print(f"       请使用 run_large_k.py（稀疏字典）或降低 k 值。")
            print(f"       如需强制运行，请使用 --force-dense 参数。")
            sys.exit(1)
        return False
    return True


class Timer:
    """简单的计时器，记录各阶段耗时。"""
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
    """训练模型或从缓存加载。"""
    cache_path = os.path.join(cache_dir, f"models_dense_k{k}.pkl")
    if os.path.exists(cache_path):
        if timer:
            timer.start("加载模型(缓存)")
        print(f"[CACHE] 命中！从 {cache_path} 加载稠密模型，跳过训练")
        models = load_models(cache_path)
        elapsed = timer.stop("加载模型(缓存)") if timer else 0
        if timer:
            print(f"[CACHE] 加载耗时 {elapsed:.1f}s，共 {len(models)} 个基因组")
        return models

    print(f"[TRAIN] 未找到缓存，开始训练 k={k} 阶马尔可夫模型（稠密矩阵）...")
    if timer:
        timer.start(f"训练(k={k})")
    models = train_all_genomes(genome_dir, k)
    elapsed = timer.stop(f"训练(k={k})") if timer else 0
    print(f"[TRAIN] 训练完成，耗时 {elapsed:.1f}s，共 {len(models)} 个基因组")

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
    # 读取 map: read_id → genome_id 或物种名
    true_mapping = {}
    with open(map_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                true_mapping[parts[0]] = parts[1]

    # genome_id → 模型索引
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
            true_idx = gid_to_idx.get(name_to_gid.get(raw_val))
        if true_idx is None:
            continue
        total_mapped += 1
        if label == true_idx:
            correct += 1

    acc = correct / total_mapped * 100 if total_mapped > 0 else 0
    return acc, correct, total_mapped


def sweep_k(genome_dir: str, test_fasta: str, map_file: str,
            k_min: int = 3, k_max: int = 12, cache_dir: str = "./models",
            safe_gb: float = 10.0) -> list:
    """扫描 k 值范围，返回 [(k, acc, correct, total, elapsed), ...]。
       自动检测内存：k 过大时跳过稠密模式。"""
    print(f"\n[INFO] 扫描 k={k_min}..{k_max} ...")
    results = []

    test_reads = read_fasta(test_fasta, keep_full_header=False)
    test_ids = [r[0] for r in test_reads]
    test_seqs = [r[1] for r in test_reads]

    for k in range(k_min, k_max + 1):
        est_gb = estimate_dense_memory(k)
        available_gb = psutil.virtual_memory().available / (1024 ** 3)

        if est_gb > available_gb * 0.8 or est_gb > safe_gb:
            print(f"  k={k:2d}: [SKIP] 稠密矩阵需 {est_gb:.2f} GB，超出安全限制，跳过")
            continue

        t0 = time.time()
        models = train_or_load(genome_dir, k, cache_dir)
        labels = classify_all(test_seqs, models, k)
        acc, correct, total = evaluate_accuracy(labels, test_ids, map_file, models)
        elapsed = time.time() - t0
        results.append((k, acc, correct, total, elapsed))
        print(f"  k={k:2d}: 准确率={acc:.2f}% ({correct}/{total}), 耗时 {elapsed:.1f}s, "
              f"内存 ~{est_gb:.2f}GB")

    return results


def print_sweep_results(results, title="k 值扫描结果汇总"):
    """格式化打印扫描结果。"""
    if not results:
        print("\n[WARN] 无有效结果。")
        return None, None
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"  {'k':<6} {'准确率':<12} {'正确/总数':<16} {'耗时':<10}")
    print(f"  {'-'*44}")
    best_k, best_acc = None, -1
    for k, acc, correct, total, elapsed in results:
        marker = " <--" if acc == max(r[1] for r in results) else ""
        print(f"  {k:<6} {acc:<11.2f}% {correct}/{total:<14} {elapsed:<9.1f}s{marker}")
        if acc > best_acc:
            best_acc = acc
            best_k = k
    print(f"\n  最优 k = {best_k}, 准确率 = {best_acc:.2f}%")
    return best_k, best_acc


def main():
    parser = argparse.ArgumentParser(description="小 k 马尔可夫模型分类（稠密矩阵）")
    parser.add_argument("--genomes", default="./proj1/genomes", help="参考基因组目录")
    parser.add_argument("--reads", default=None, help="待分类 reads 文件（可选，仅做分类统计时使用）")
    parser.add_argument("--test", default=None, help="测试集文件（用于评估准确率）")
    parser.add_argument("--map", default=None, help="seq_id.map 文件路径（配合 --test 使用）")
    parser.add_argument("--k", type=int, default=6, help="模型阶数 (默认 6)")
    parser.add_argument("--sweep", action="store_true", help="扫描 k=3..12 寻找最优 k")
    parser.add_argument("--cache-dir", default="./models", help="模型缓存目录")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_FACTOR,
                        help="分类阈值（默认 -5.0）")
    parser.add_argument("--dense-threshold", type=int, default=K_DENSE_THRESHOLD,
                        help=f"稠密/稀疏切换阈值 (默认 {K_DENSE_THRESHOLD})")
    parser.add_argument("--safe-memory-gb", type=float, default=10.0,
                        help="稠密矩阵安全内存上限 GB (默认 10.0)")
    parser.add_argument("--force-dense", action="store_true",
                        help="强制使用稠密矩阵（忽略内存检查）")
    args = parser.parse_args()

    timer = Timer()

    # ---- 扫描模式 ----
    if args.sweep:
        if not args.test or not args.map:
            print("[ERROR] --sweep 需要 --test 和 --map 参数")
            sys.exit(1)

        timer.start("total_sweep")
        results = sweep_k(args.genomes, args.test, args.map,
                          k_min=3, k_max=args.dense_threshold,
                          cache_dir=args.cache_dir, safe_gb=args.safe_memory_gb)
        timer.stop("total_sweep")
        print_sweep_results(results)
        timer.summary()
        return

    # ---- 标准分类模式 ----
    if not args.reads and not args.test:
        print("[ERROR] 请至少指定 --reads 或 --test")
        sys.exit(1)

    print(f"[INFO] 使用 k={args.k} 阶马尔可夫模型（稠密矩阵）")
    print(f"[INFO] 稠密/稀疏阈值: k={args.dense_threshold}")

    # 内存检查
    if not args.force_dense:
        check_memory_safety(args.k, safe_gb=args.safe_memory_gb, auto_fallback=True)

    # 训练
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
