"""
markov_train.py
模型训练与存取（高低 k 分离，统一返回模型字典）

调用方法：
    model = train_one_genome(seq, k, pseudocount)          # 训练单个基因组
    models = train_all_genomes(genome_dir, k, pseudocount)  # 批量训练
    save_models(models, filepath)                           # 保存到磁盘
    models = load_models(filepath)                          # 从磁盘加载

    训练结果统一返回模型字典（Dict）
    模型字典中包含 'type' 字段（'dense' 或 'sparse'），下游函数可根据此字段选择查表方式。

    1) 伪计数平滑：低 k 与高 k 均使用相同的伪计数策略。
       低 k：初始化矩阵时直接填充 PSEUDOCOUNT，再叠加实际计数。
       高 k：转移计数用 defaultdict(lambda: np.full(4, pseudocount))，
              初始计数用 defaultdict(lambda: pseudocount)，并在每次出现时加 1，
              保证与低 k 数学完全一致。
    2) 初始概率：虽然训练时计算了 init_log_prob / init_log_prob_dict，
       但分类时不再使用（见 classify_cpu.score_read 的注释）。
       原因是高 k 的初始概率无法覆盖所有可能的 k-mer（稀疏），造成高低 k 尺度不一致。
    3) 高 k 模型额外提供 'default_log_prob'（均匀分布的对数值），
       用于打分时遇到训练末见的上下文时回退。
"""

import os
import pickle
import numpy as np
from typing import Dict, List
from collections import defaultdict
from data_loader import merge_genome
from utils import PSEUDOCOUNT, BASE2ID, K_DENSE_THRESHOLD


# 低 k 训练（稠密矩阵）
def _train_low(seq: str, k: int, pseudocount: float) -> Dict:
    num_states = 4 ** k
    trans_count = np.full((num_states, 4), pseudocount, dtype=np.float64)
    init_count = np.full(num_states, pseudocount, dtype=np.float64)

    clean_seq = ''.join([b if b in BASE2ID else 'N' for b in seq])
    contigs = clean_seq.split('N')

    for contig in contigs:
        if len(contig) <= k:
            continue
        seq_int = np.array([BASE2ID[b] for b in contig], dtype=np.int64)
        L = len(seq_int)

        for i in range(L - k + 1):
            ctx = 0
            for j in range(k):
                ctx = ctx * 4 + seq_int[i + j]
            init_count[ctx] += 1

        for i in range(L - k):
            ctx = 0
            for j in range(k):
                ctx = ctx * 4 + seq_int[i + j]
            next_base = seq_int[i + k]
            trans_count[ctx, next_base] += 1

    row_sums = trans_count.sum(axis=1, keepdims=True)
    log_prob = np.log(trans_count / row_sums)
    init_log_prob = np.log(init_count / init_count.sum())   # 保留但不用于打分
    return {'k': k, 'type': 'dense', 'log_prob': log_prob, 'init_log_prob': init_log_prob}


# 高 k 训练（稀疏字典）
def _train_high(seq: str, k: int, pseudocount: float) -> Dict:
    trans_count = defaultdict(lambda: np.full(4, pseudocount, dtype=np.float64))
    init_count = defaultdict(lambda: pseudocount)            # 修改 2

    clean_seq = ''.join([b if b in BASE2ID else 'N' for b in seq])
    contigs = clean_seq.split('N')

    for contig in contigs:
        if len(contig) <= k:
            continue
        seq_int = np.array([BASE2ID[b] for b in contig], dtype=np.int64)
        L = len(seq_int)

        for i in range(L - k + 1):
            ctx = 0
            for j in range(k):
                ctx = ctx * 4 + seq_int[i + j]
            init_count[ctx] += 1                              # 修改 3

        for i in range(L - k):
            ctx = 0
            for j in range(k):
                ctx = ctx * 4 + seq_int[i + j]
            next_base = seq_int[i + k]
            trans_count[ctx][next_base] += 1

    log_prob_dict = {}
    for ctx, counts in trans_count.items():
        log_prob_dict[ctx] = np.log(counts / counts.sum())
    total_init = sum(init_count.values())
    init_log_prob_dict = {ctx: np.log(cnt / total_init) for ctx, cnt in init_count.items()}

    return {
        'k': k,
        'type': 'sparse',
        'log_prob_dict': log_prob_dict,
        'init_log_prob_dict': init_log_prob_dict,
        'default_log_prob': np.log(0.25),
        'backoff_k': k - 1
    }


def train_one_genome(seq: str, k: int, pseudocount: float = PSEUDOCOUNT) -> Dict:
    if k <= K_DENSE_THRESHOLD:
        return _train_low(seq, k, pseudocount)
    else:
        return _train_high(seq, k, pseudocount)


def train_all_genomes(genome_dir: str, k: int, pseudocount: float = PSEUDOCOUNT) -> List[Dict]:
    """
    批量训练目录下所有参考基因组。
    参数：genome_dir(基因组文件目录), k, pseudocount
    返回：List[Dict]，每个元素为一个基因组的模型字典，额外添加了 'genome_id' 字段。
    """
    models = []
    for fname in sorted(os.listdir(genome_dir)):
        if fname.endswith(('.fa', '.fasta', '.fna')):
            filepath = os.path.join(genome_dir, fname)
            seq = merge_genome(filepath, use_reverse_complement=True)
            model = train_one_genome(seq, k, pseudocount)
            model['genome_id'] = os.path.splitext(fname)[0]
            models.append(model)
    return models


def save_models(models: List[Dict], filepath: str) -> None:
    """将模型列表保存到磁盘（pickle 格式）。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(models, f)


def load_models(filepath: str) -> List[Dict]:
    """从磁盘加载模型列表。"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)