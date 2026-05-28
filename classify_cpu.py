"""
classify_cpu.py
CPU 版分类与打分（忽略初始概率，仅使用转移概率）

调用方法：
    score = score_read(read, model, k)                     # 计算单条 read 得分
    label = classify_read(read, models, k, threshold)      # 单条分类
    labels = classify_all(reads, models, k, threshold)     # 批量分类

    ·score_read 只累加转移对数概率，忽略初始概率。原因：高 k 稀疏模型的初始概率分布不全，与低 k 不可比，且对分类贡献微小。确保高低 k 模型得分尺度一致（已与项目组确认）。
    ·classify_read / classify_all 的 threshold_factor 参数为绝对对数似然阈值，
    ·当最佳得分 < threshold_factor 时返回 -1（无法分配）。该阈值可在 utils.py 中统一设置。

    ·稀疏模型的回退：遇到字典中没有的上下文，使用 default_log_prob（均匀分布的对数）。
    ·含 N 的 k-mer 在编码阶段标记为 -1，打分时自动跳过，不影响最终似然值。
"""
import numpy as np
from typing import List, Dict
from encoding import encode_read
from utils import THRESHOLD_FACTOR


def score_read(read: str, model: Dict, k: int) -> float:
    """
    计算一条 read 在给定模型下的对数似然（只使用转移概率）。
    参数：
        read: 待评估序列（可含 N，含 N 的 k-mer 自动跳过）
        model: 单个基因组的模型字典（dense 或 sparse）
        k: 模型阶数
    返回：
        对数似然值（总加和，未归一化）。若长度 < k+1 则返回 -inf。
    """
    if len(read) < k + 1:
        return -np.inf

    ctx_indices, base_indices = encode_read(read, k)
    if len(ctx_indices) == 0:
        return -np.inf

    n_valid = 0
    if model['type'] == 'dense':
        log_prob = model['log_prob']
        total = 0.0
        for ctx, base in zip(ctx_indices, base_indices):
            if ctx < 0:          # 含 N 的无效位置，跳过
                continue
            total += log_prob[ctx, base]
            n_valid += 1
    else:  # sparse
        log_prob_dict = model['log_prob_dict']
        default = model['default_log_prob']
        total = 0.0
        for ctx, base in zip(ctx_indices, base_indices):
            if ctx < 0:          # 含 N 的无效位置，跳过
                continue
            n_valid += 1
            if ctx in log_prob_dict:
                total += log_prob_dict[ctx][base]
            else:
                total += default

    if n_valid == 0:
        return -np.inf

    return total


def classify_read(read: str, models: List[Dict], k: int,
                  threshold_factor: float = THRESHOLD_FACTOR) -> int:
    """
    对一条 read 分类。
    参数：
        read: 待分类序列
        models: 所有基因组的模型列表（长度 = 基因组数）
        k: 模型阶数
        threshold_factor: 绝对阈值。若最佳得分 < threshold_factor 则判为 -1。
                          默认值从 utils.THRESHOLD_FACTOR 读取。
    返回：
        int: 最佳基因组索引（0 ~ N-1），若不可靠则返回 -1。
    """
    best_idx = -1
    best_score = -np.inf
    for idx, model in enumerate(models):
        s = score_read(read, model, k)
        if s > best_score:
            best_score = s
            best_idx = idx

    if threshold_factor is not None:
        norm_score = best_score / max(1, len(read) - k)
        if norm_score < threshold_factor:
            return -1
    return best_idx


def classify_all(reads: List[str], models: List[Dict], k: int,
                 threshold_factor: float = THRESHOLD_FACTOR) -> np.ndarray:
    """
    批量分类 reads。
    参数：同 classify_read，但 reads 是序列列表。
    返回：形状 (len(reads),) 的整数标签数组，-1 表示无法分配。
    """
    labels = np.empty(len(reads), dtype=int)
    for i, read in enumerate(reads):
        labels[i] = classify_read(read, models, k, threshold_factor)
    return labels
