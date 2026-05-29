"""
classify_cpu.py
CPU 版分类与打分：使用置换检验（permutation test）判断匹配显著性。

原理：
  对每条待测序列，先在所有基因组模型上评分，找到最佳匹配基因组。
  然后将该序列随机打乱 N 次，每次在最佳基因组模型上打分，构造零分布。
  若原始得分经长度归一化后，超出零分布均值 Z_SCORE_THRESHOLD 个标准差，
  则认为匹配显著，返回该基因组索引；否则返回 -1（无法分配）。

  这种方法比固定阈值更可靠：零分布来自同一条序列的随机打乱，
  天然控制了碱基组成的影响。

调用方法：
    score = score_read(read, model, k)                     # 计算单条 read 得分
    label = classify_read(read, models, k)                 # 单条分类（置换检验）
    labels = classify_all(reads, models, k)                # 批量分类

    · score_read 只累加转移对数概率，忽略初始概率。
    · 稀疏模型的回退：遇到字典中没有的上下文，使用 default_log_prob（均匀分布的对数）。
    · 含 N 的 k-mer 在编码阶段标记为 -1，打分时自动跳过。
"""
import numpy as np
from typing import List, Dict, Tuple
from encoding import encode_read
from utils import Z_SCORE_THRESHOLD, N_PERMUTATIONS


def shuffle_sequence(seq: str) -> str:
    """
    随机打乱 DNA 序列，保留碱基组成不变。
    用于构造置换检验的零分布。
    """
    chars = list(seq)
    np.random.shuffle(chars)
    return ''.join(chars)


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

    if model['type'] == 'dense':
        log_prob = model['log_prob']
        total = 0.0
        for ctx, base in zip(ctx_indices, base_indices):
            if ctx < 0:          # 含 N 的无效位置，跳过
                continue
            total += log_prob[ctx, base]
    else:  # sparse
        log_prob_dict = model['log_prob_dict']
        default = model['default_log_prob']
        total = 0.0
        for ctx, base in zip(ctx_indices, base_indices):
            if ctx < 0:          # 含 N 的无效位置，跳过
                continue
            if ctx in log_prob_dict:
                total += log_prob_dict[ctx][base]
            else:
                total += default

    if total == 0.0:
        return -np.inf

    return total


def _null_distribution(read: str, model: Dict, k: int,
                       n_permutations: int) -> Tuple[float, float]:
    """
    通过随机打乱序列构造零分布，返回 (均值, 标准差)。
    原始序列的得分若显著高于该分布的均值，则匹配有意义。
    """
    scores = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = shuffle_sequence(read)
        scores[i] = score_read(shuffled, model, k)
    return float(scores.mean()), float(scores.std(ddof=1))


def classify_read(read: str, models: List[Dict], k: int,
                  z_threshold: float = Z_SCORE_THRESHOLD,
                  n_permutations: int = N_PERMUTATIONS) -> int:
    """
    对一条 read 分类，使用置换检验判断显著性。
    参数：
        read: 待分类序列
        models: 所有基因组的模型列表（长度 = 基因组数）
        k: 模型阶数
        z_threshold: z 值阈值（单侧）。原始得分归一化后需超过零分布均值
                     z_threshold 个标准差才认为显著。默认为 utils.Z_SCORE_THRESHOLD。
        n_permutations: 随机打乱次数，默认为 utils.N_PERMUTATIONS。
    返回：
        int: 最佳基因组索引（0 ~ N-1），若不显著则返回 -1。
    """
    if len(read) < k + 1:
        return -1

    # 在所有基因组上评分，找到最佳匹配
    best_idx = -1
    best_score = -np.inf
    for idx, model in enumerate(models):
        s = score_read(read, model, k)
        if s > best_score:
            best_score = s
            best_idx = idx

    if best_score == -np.inf or best_idx < 0:
        return -1

    norm_factor = max(1, len(read) - k)

    # 对最佳基因组做置换检验
    null_mean, null_std = _null_distribution(read, models[best_idx], k, n_permutations)

    # 归一化原始得分和零分布
    norm_original = best_score / norm_factor
    norm_mean = null_mean / norm_factor
    norm_std = null_std / norm_factor

    if norm_std == 0:
        # 所有打乱序列得分相同，无法判断显著性，保守地拒绝
        return -1

    z_score = (norm_original - norm_mean) / norm_std

    if z_score >= z_threshold:
        return best_idx
    return -1


def classify_all(reads: List[str], models: List[Dict], k: int,
                 z_threshold: float = Z_SCORE_THRESHOLD,
                 n_permutations: int = N_PERMUTATIONS) -> np.ndarray:
    """
    批量分类 reads，每条 read 使用置换检验判断显著性。
    参数：同 classify_read，但 reads 是序列列表。
    返回：形状 (len(reads),) 的整数标签数组，-1 表示无法分配。
    """
    labels = np.empty(len(reads), dtype=int)
    for i, read in enumerate(reads):
        labels[i] = classify_read(read, models, k, z_threshold, n_permutations)
    return labels
