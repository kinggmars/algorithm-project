"""
test_markov.py
用于验证 k 阶马尔可夫模型各模块的正确性。
适配 markov_train.py 返回模型字典。
"""

import numpy as np
from utils import BASES, BASE2ID, PSEUDOCOUNT
from encoding import kmer_to_int, int_to_kmer, encode_read
from markov_train import train_one_genome
from classify_cpu import score_read


def test_kmer_encoding():
    """测试 kmer 编解码的可逆性"""
    k = 5
    original = "ACGTA"
    encoded = kmer_to_int(original)
    decoded = int_to_kmer(encoded, k)
    assert original == decoded, f"编解码失败: {original} -> {encoded} -> {decoded}"
    print("[PASS] test_kmer_encoding")

    # 测试边界：全A 和 全T
    assert kmer_to_int("AAA") == 0, f"AAA 编码应为 0，实际 {kmer_to_int('AAA')}"
    assert kmer_to_int("TTT") == 63, f"TTT 编码应为 63，实际 {kmer_to_int('TTT')}"
    print("[PASS] kmer_to_int 基础边界测试")


def get_log_prob_init(model_dict):
    """从模型字典中提取 log_prob 和 init_log_prob（兼容 dense/sparse）"""
    if model_dict['type'] == 'dense':
        return model_dict['log_prob'], model_dict['init_log_prob']
    else:  # sparse
        # 稀疏模型返回字典形式的 log_prob，同时附带默认值
        return model_dict['log_prob_dict'], model_dict.get('init_log_prob_dict', {})


def test_model_sanity():
    """测试模型概率表的数值稳定性（使用低 k 确保 dense 输出）"""
    seq = "AAAAACCCCGGGGTTTT" * 50
    k = 3
    model_dict = train_one_genome(seq, k)          # 返回字典
    log_prob, init_log_prob = get_log_prob_init(model_dict)

    # 检查是否含有 NaN 或 Inf（仅 dense 模式）
    if isinstance(log_prob, np.ndarray):
        assert not np.isnan(log_prob).any(), "log_prob 包含 NaN"
        assert not np.isinf(log_prob).any(), "log_prob 包含 Inf"
        assert not np.isnan(init_log_prob).any(), "init_log_prob 包含 NaN"
        assert not np.isinf(init_log_prob).any(), "init_log_prob 包含 Inf"
        print("[PASS] test_model_sanity: 无 NaN/Inf")

        # 检查转移概率行和 ≈ 1
        probs = np.exp(log_prob)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"转移概率行和不为1，最大偏差 {np.max(np.abs(row_sums - 1.0))}"
        print("[PASS] test_model_sanity: 转移概率行和 ≈ 1")

        # 检查初始概率和
        init_probs = np.exp(init_log_prob)
        assert np.isclose(init_probs.sum(), 1.0, atol=1e-6), "初始概率和不为1"
        print("[PASS] test_model_sanity: 初始概率和 ≈ 1")
    else:
        print("[SKIP] 高 k 模型使用字典存储，不做矩阵检验。可手工检查。")


def test_known_sequence():
    """验证 score_read 正确使用了模型中的转移概率（不考虑初始概率）"""
    seq = "AAAA" + "AAAC" + "AAAG" + "AAAT"
    k = 3
    model_dict = train_one_genome(seq, k)
    log_prob, init_log_prob = get_log_prob_init(model_dict)

    test_read = "AAAC"
    model = {
        'k': k,
        'log_prob': log_prob,
        'init_log_prob': init_log_prob,
        'type': model_dict['type']
    }
    if model['type'] == 'sparse':
        model['default_log_prob'] = model_dict.get('default_log_prob', np.log(0.25))

    from encoding import kmer_to_int, BASE2ID
    ctx = kmer_to_int("AAA")
    base_idx = BASE2ID['C']


    if isinstance(log_prob, np.ndarray):
        expected = log_prob[ctx, base_idx]
    else:
        expected = log_prob.get(ctx, model['default_log_prob'])

    score = score_read(test_read, model, k)
    assert np.isclose(score, expected, atol=1e-5), f"期望分数 {expected}, 实际 {score}"
    print("[PASS] test_known_sequence: 模型查表一致性验证通过")


if __name__ == "__main__":
    test_kmer_encoding()
    test_model_sanity()
    test_known_sequence()
    print("\n所有测试通过！")