import numpy as np
from utils import BASE2ID,BASES,ID2BASE
from typing import Tuple

#编码从高位到低位编码，比如序列AC被编码为四进制01

def kmer_to_int(kmer: str) -> int:
    """将 A/C/G/T 组成的 k-mer 字符串编码为整数 0 ~ 4^k-1。"""
    number=0
    for char in kmer:
        number*=4
        number+=BASE2ID[char]
    return number

def int_to_kmer(idx: int, k: int) -> str:
    """将整数解码回 k-mer 字符串（调试用）。"""
    bases = []
    for _ in range(k):
        bases.append(ID2BASE[idx % 4])
        idx //= 4
    return ''.join(reversed(bases))

def encode_read(read: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """将一条序列编码为两个数组（滚动哈希，O(L) 复杂度）：
       - ctx_indices: 长度为 L-k 的整数数组，含非 ACGT 字符的位置标记为 -1。
       - base_indices: 长度为 L-k 的整数数组，含非 ACGT 字符的位置标记为 -1。
       若 L <= k 则返回空数组。
       支持 IUPAC 模糊碱基（N/W/R/Y/S/K/M/B/D/H/V），这些位置自动跳过。
    """
    L = len(read)
    if L <= k:
        return np.array([], dtype=int), np.array([], dtype=int)

    n = L - k

    # ---- 快速路径：序列只含纯 ACGT（大小写均可）----
    if set(read.upper()) <= {'A', 'C', 'G', 'T'}:
        ctx_indices = np.empty(n, dtype=int)
        base_indices = np.empty(n, dtype=int)

        ctx = 0
        for j in range(k):
            ctx = ctx * 4 + BASE2ID[read[j]]

        mask = 4 ** (k - 1)
        for i in range(n):
            ctx_indices[i] = ctx
            nxt = BASE2ID[read[i + k]]
            base_indices[i] = nxt
            ctx = (ctx % mask) * 4 + nxt
        return ctx_indices, base_indices

    # ---- 慢速路径：含非 ACGT 字符（N/W/R/Y/...），跳过无效 k-mer ----
    ctx_indices = np.full(n, -1, dtype=int)
    base_indices = np.full(n, -1, dtype=int)

    read_int = np.array([BASE2ID.get(c.upper(), -1) for c in read], dtype=int)
    mask = 4 ** (k - 1)
    ctx = -1

    for i in range(n):
        # 下一个碱基无效 → 跳过
        if read_int[i + k] < 0:
            ctx = -1
            continue

        # 窗口中任意碱基无效 → 跳过
        ok = True
        for j in range(k):
            if read_int[i + j] < 0:
                ok = False
                break
        if not ok:
            ctx = -1
            continue

        # 重新计算 ctx（在 N 之后需要）
        if ctx < 0:
            ctx = 0
            for j in range(k):
                ctx = ctx * 4 + read_int[i + j]

        ctx_indices[i] = ctx
        base_indices[i] = read_int[i + k]
        ctx = (ctx % mask) * 4 + read_int[i + k]

    return ctx_indices, base_indices

    
    
    
