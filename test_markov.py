"""
test_markov.py
用于验证 k 阶马尔可夫模型各模块的正确性。
适配 markov_train.py 返回模型字典，以及 classify_cpu.py 的置换检验。
"""

import numpy as np
from utils import BASES, BASE2ID, PSEUDOCOUNT, Z_SCORE_THRESHOLD, N_PERMUTATIONS
from encoding import kmer_to_int, int_to_kmer, encode_read
from markov_train import train_one_genome, save_models, load_models
from classify_cpu import score_read, shuffle_sequence, classify_read


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
        return model_dict['log_prob_dict'], model_dict.get('init_log_prob_dict', {})


def test_model_sanity():
    """测试模型概率表的数值稳定性（使用低 k 确保 dense 输出）"""
    seq = "AAAAACCCCGGGGTTTT" * 50
    k = 3
    model_dict = train_one_genome(seq, k)
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
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            f"转移概率行和不为1，最大偏差 {np.max(np.abs(row_sums - 1.0))}"
        print("[PASS] test_model_sanity: 转移概率行和 ≈ 1")

        # 检查初始概率和
        init_probs = np.exp(init_log_prob)
        assert np.isclose(init_probs.sum(), 1.0, atol=1e-6), "初始概率和不为1"
        print("[PASS] test_model_sanity: 初始概率和 ≈ 1")
    else:
        print("[SKIP] 高 k 模型使用字典存储，不做矩阵检验。")


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

    ctx = kmer_to_int("AAA")
    base_idx = BASE2ID['C']

    if isinstance(log_prob, np.ndarray):
        expected = log_prob[ctx, base_idx]
    else:
        expected = log_prob.get(ctx, model['default_log_prob'])

    score = score_read(test_read, model, k)
    assert np.isclose(score, expected, atol=1e-5), f"期望分数 {expected}, 实际 {score}"
    print("[PASS] test_known_sequence: 模型查表一致性验证通过")


def test_shuffle_preserves_composition():
    """测试随机打乱保留碱基组成"""
    seq = "AAAACCCCGGGGTTTT"
    shuffled = shuffle_sequence(seq)
    assert sorted(seq) == sorted(shuffled), \
        f"打乱后碱基组成应不变: 原={sorted(seq)}, 打乱后={sorted(shuffled)}"
    # 大概率不会相同（除非极巧合）
    assert seq != shuffled or len(set(seq)) == 1, \
        "打乱后序列应与原序列不同（除非全部相同碱基）"
    print("[PASS] test_shuffle_preserves_composition")


def test_classify_read():
    """测试 classify_read 使用置换检验的基本功能"""
    # 构造两个容易区分的模型（使用不同 k-mer 模式，避免单一碱基导致零方差）
    seq_a = "ACGTACGTACGT" * 50   # 周期性的 ACGT 模式
    seq_b = "TGCATGCATGCA" * 50   # 周期性的 TGCA 模式
    k = 3
    model_a = train_one_genome(seq_a, k)
    model_b = train_one_genome(seq_b, k)

    # 测试序列匹配模型 A 的 k-mer 模式
    test_read = "ACGTACGTACGT" * 5
    label = classify_read(test_read, [model_a, model_b], k,
                          z_threshold=-1.0,  # 非常宽松的阈值
                          n_permutations=30)
    assert label == 0, f"ACGT... 应匹配模型 A (idx=0)，实际 {label}"
    print("[PASS] test_classify_read: 基本分类正确")

    # 随机序列在严格阈值下不应匹配
    np.random.seed(42)
    random_read = ''.join(np.random.choice(list('ACGT'), 100))
    label = classify_read(random_read, [model_a, model_b], k,
                          z_threshold=5.0, n_permutations=30)
    print(f"[INFO] test_classify_read: 随机序列分类结果 = {label} "
          f"(z_threshold=5.0，预期为 -1)")


def test_save_load_models():
    """测试模型保存/加载的往返一致性"""
    import os
    import tempfile
    seq = "AAAACCCCGGGGTTTT" * 50
    k = 3
    model = train_one_genome(seq, k)
    model['genome_id'] = 'test_genome'

    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, 'test_models.pkl')
    try:
        save_models([model], filepath)
        loaded = load_models(filepath)
        assert len(loaded) == 1, f"应加载 1 个模型，实际 {len(loaded)}"
        assert loaded[0]['genome_id'] == 'test_genome'
        assert loaded[0]['type'] == model['type']
        assert loaded[0]['k'] == model['k']
        # 验证概率矩阵一致
        if model['type'] == 'dense':
            assert np.allclose(loaded[0]['log_prob'], model['log_prob'])
        print("[PASS] test_save_load_models: 保存/加载往返一致")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rmdir(tmpdir)


if __name__ == "__main__":
    test_kmer_encoding()
    test_model_sanity()
    test_known_sequence()
    test_shuffle_preserves_composition()
    test_classify_read()
    test_save_load_models()
    print("\n所有测试通过！")
