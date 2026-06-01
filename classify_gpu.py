"""
classify_gpu.py
GPU 版分类与打分：使用置换检验（permutation test）判断匹配显著性。

原理：
  对每条待测序列，先在所有基因组模型上评分，找到最佳匹配基因组。
  然后将该序列随机打乱 N 次，每次在最佳基因组模型上打分，构造零分布。
  若原始得分经长度归一化后，超出零分布均值 Z_SCORE_THRESHOLD 个标准差，
  则认为匹配显著，返回该基因组索引；否则返回 -1（无法分配）。

加速策略：
  1. 模型评分（所有基因组）在 GPU 上批量完成。
  2. 置换检验通过批量生成打乱序列、一次性编码及评分，在 GPU 上并行计算 z 值。
  3. 仅支持稠密模型（k ≤ 10）；若 GPU 不可用、存在稀疏模型或显存不足，
     直接跳过 GPU 分类（返回全 -1）并打印警告。

调用方法：
    labels = classify_all_gpu(reads, models, k)
"""

import numpy as np
import torch
from typing import List, Dict
from utils import BASE2ID, Z_SCORE_THRESHOLD, N_PERMUTATIONS

GPU_MEMORY_LIMIT = 0.8


def _is_dense(model: Dict) -> bool:
    """判断模型是否为稠密矩阵类型"""
    return model.get('type', 'dense') == 'dense'


def _estimate_model_memory(models: List[Dict], dtype_size: int = 4) -> int:
    """估算将所有稠密模型的 log_prob 载入 GPU 需要的显存（字节）"""
    total = 0
    for m in models:
        if _is_dense(m):
            k = m['k']
            total += (4 ** k) * 4 * dtype_size
    return total


def _gpu_memory_sufficient(models: List[Dict], safety_factor: float = GPU_MEMORY_LIMIT) -> bool:
    """检查 GPU 剩余显存是否足够放下所有稠密模型"""
    if not torch.cuda.is_available():
        return False
    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        free = total_mem - reserved
        needed = _estimate_model_memory(models)
        return needed <= free * safety_factor
    except Exception:
        return False


def _strings_to_tensor(reads: List[str], device: str):
    """将字符串列表转换为整数张量（A->0, C->1, G->2, T->3, 其他->-1），返回 (seq_tensor, lengths)"""
    B = len(reads)
    max_len = max(len(r) for r in reads)
    seq_tensor = torch.full((B, max_len), -1, dtype=torch.int64, device=device)
    lengths = torch.zeros(B, dtype=torch.int64, device=device)
    for i, r in enumerate(reads):
        L = len(r)
        lengths[i] = L
        ints = [BASE2ID.get(c, -1) for c in r.upper()]
        seq_tensor[i, :L] = torch.tensor(ints, dtype=torch.int64, device=device)
    return seq_tensor, lengths


def _seq_to_ctx_base(seq_tensor: torch.Tensor, k: int):
    """
    将整数序列 (N, max_len) 转换为上下文索引和下一碱基。
    返回 ctx_idx (N, L-k), base (N, L-k), lengths (N,)。
    """
    N, L = seq_tensor.shape
    if L <= k:
        ctx = torch.full((N, 0), -1, dtype=torch.int64, device=seq_tensor.device)
        base = torch.full((N, 0), -1, dtype=torch.int64, device=seq_tensor.device)
        lengths = torch.zeros(N, dtype=torch.int64, device=seq_tensor.device)
        return ctx, base, lengths

    windows = seq_tensor.unfold(1, k + 1, 1)
    ctx = windows[:, :, :k]
    base = windows[:, :, k]

    powers = torch.tensor([4 ** (k - 1 - i) for i in range(k)], dtype=torch.int64, device=seq_tensor.device)
    ctx_idx = (ctx * powers).sum(dim=2)

    valid = (windows != -1).all(dim=2)
    ctx_idx = torch.where(valid, ctx_idx, -1)
    base = torch.where(valid, base, -1)
    lengths = valid.sum(dim=1)
    return ctx_idx, base, lengths


def _score_batch(ctx: torch.Tensor, base: torch.Tensor, lengths: torch.Tensor,
                 log_prob: torch.Tensor) -> torch.Tensor:
    """批量计算 reads 在单个模型下的对数似然（仅转移概率）"""
    B, L_max = ctx.shape
    mask = torch.arange(L_max, device=ctx.device).unsqueeze(0) < lengths.unsqueeze(1)
    safe_ctx = ctx.clamp(min=0)
    safe_base = base.clamp(min=0)
    per_pos = log_prob[safe_ctx, safe_base] * mask.float()
    return per_pos.sum(dim=1)


def _generate_permuted_batch(seq_tensor: torch.Tensor, lengths: torch.Tensor,
                             P: int, device: str):
    """为 batch 中每条序列生成 P 个随机打乱版本"""
    B, max_len = seq_tensor.shape
    shuffled_list = []
    for i in range(B):
        L = lengths[i].item()
        if L == 0:
            shuf = torch.full((P, max_len), -1, dtype=torch.int64, device=device)
        else:
            seq_i = seq_tensor[i, :L]
            rand = torch.rand(P, L, device=device)
            perm_idx = torch.argsort(rand)
            shuf_i = seq_i[perm_idx]
            shuf = torch.full((P, max_len), -1, dtype=torch.int64, device=device)
            shuf[:, :L] = shuf_i
        shuffled_list.append(shuf)
    shuffled = torch.cat(shuffled_list, dim=0)
    perm_lengths = lengths.repeat_interleave(P)
    return shuffled, perm_lengths


def classify_all_gpu(reads: List[str], models: List[Dict], k: int,
                     z_threshold: float = Z_SCORE_THRESHOLD,
                     n_permutations: int = N_PERMUTATIONS,
                     batch_size: int = 2048) -> np.ndarray:
    """
    GPU 批量分类 reads（含置换检验显著性判断）。

    若 GPU 不可用、模型非全稠密或显存不足，打印警告并跳过分类（返回全 -1）。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("GPU 不可用，跳过 GPU 分类，返回全 -1。")
        return np.full(len(reads), -1, dtype=int)

    if not all(_is_dense(m) for m in models):
        print("存在稀疏模型，跳过 GPU 分类，返回全 -1。")
        return np.full(len(reads), -1, dtype=int)

    if not _gpu_memory_sufficient(models):
        est_mb = _estimate_model_memory(models) / 1e6
        print(f"模型所需显存约 {est_mb:.1f} MB，超出 GPU 安全限制，跳过分类，返回全 -1。")
        return np.full(len(reads), -1, dtype=int)

    log_probs = torch.stack([torch.tensor(m['log_prob'], dtype=torch.float32, device=device)
                             for m in models], dim=0)
    G = len(models)
    V = 4 ** k
    # 将 log_probs 展平为 [G * V * 4] 的一维张量，用于线性索引
    log_probs_flat = log_probs.view(G, V * 4).contiguous().view(-1)  # [G * V * 4]

    total = len(reads)
    labels = np.empty(total, dtype=int)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_reads = reads[start:end]
        B = len(batch_reads)

        seq_tensor, lengths = _strings_to_tensor(batch_reads, device)

        ctx, base, eff_lengths = _seq_to_ctx_base(seq_tensor, k)
        # 计算原始得分
        scores = torch.empty(B, G, dtype=torch.float32, device=device)
        for g in range(G):
            scores[:, g] = _score_batch(ctx, base, eff_lengths, log_probs[g])
        best_scores, best_idxs = scores.max(dim=1)

        P = n_permutations
        # 生成打乱序列
        shuffled, perm_lengths = _generate_permuted_batch(seq_tensor, lengths, P, device)
        ctx_perm, base_perm, eff_perm = _seq_to_ctx_base(shuffled, k)
        N_perm = ctx_perm.size(0)          # B * P
        L_perm = ctx_perm.size(1)

        # 为每个打乱序列位置构造线性索引，直接从 log_probs_flat 中取值
        # seq_idx: [B*P]，每个打乱序列对应的原始序列索引
        seq_idx = torch.arange(B, device=device).repeat_interleave(P)   # [B*P]
        # 扩展为 [B*P, L_perm]
        seq_idx_expand = seq_idx[:, None].expand(-1, L_perm)            # [B*P, L_perm]

        # 获取每个位置对应的最佳模型索引
        best_idx_expand = best_idxs[seq_idx_expand]                     # [B*P, L_perm]

        ctx_clamp = ctx_perm.clamp(min=0)          # [B*P, L_perm]
        base_clamp = base_perm.clamp(min=0)        # [B*P, L_perm]

        # 线性索引: idx = best_idx * (V*4) + ctx*4 + base
        linear_idx = best_idx_expand * (V * 4) + ctx_clamp * 4 + base_clamp   # [B*P, L_perm]

        # 取出概率值
        per_pos = log_probs_flat[linear_idx]        # [B*P, L_perm]

        # 计算零分布得分
        mask = torch.arange(L_perm, device=device).unsqueeze(0) < eff_perm.unsqueeze(1)
        null_scores = (per_pos * mask.float()).sum(dim=1)          # [B*P]
        null_scores = null_scores.view(B, P)                       # [B, P]

        null_mean = null_scores.mean(dim=1)
        null_std = null_scores.std(dim=1, unbiased=True)
        null_std = torch.where(null_std == 0, torch.ones_like(null_std), null_std)
        norm = eff_lengths.clamp(min=1).float()
        z = (best_scores / norm - null_mean / norm) / (null_std / norm)

        batch_labels = torch.where(z >= z_threshold, best_idxs, -1).cpu().numpy()
        labels[start:end] = batch_labels

        # 清理中间变量，释放显存
        del seq_tensor, lengths, ctx, base, eff_lengths, scores, best_scores, best_idxs
        del shuffled, perm_lengths, ctx_perm, base_perm, eff_perm, null_scores
        del seq_idx, seq_idx_expand, best_idx_expand, linear_idx, per_pos, mask
        torch.cuda.empty_cache()

    return labels