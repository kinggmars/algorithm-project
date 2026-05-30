"""
test_shuffle_distribution.py
测试随机打乱序列后的得分分布。
- 使用 test.fa 中的第一条序列
- 使用 k=6 的模型
- 随机打乱 N 次，统计每次打乱后在各基因组模型上的得分
- 输出分布统计信息
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from scipy import stats
from data_loader import read_fasta
from markov_train import load_models
from classify_cpu import score_read, shuffle_sequence

# ==================== 配置 ====================
TEST_FA = "./proj1/test.fa"
MODEL_PATH = "./models/models_dense_k6.pkl"
K = 6
N_SHUFFLES = 10000  # 打乱次数
SEED = 42

np.random.seed(SEED)

# ==================== 加载数据 ====================
print("=" * 60)
print("加载数据...")

# 读取 test.fa 第一条序列
reads = read_fasta(TEST_FA)
first_id, first_seq = reads[0]
print(f"第一条序列 ID: {first_id}")
print(f"序列长度: {len(first_seq)} bp")
print(f"序列前60bp: {first_seq[:60]}")

# 加载模型
models = load_models(MODEL_PATH)
print(f"加载了 {len(models)} 个基因组模型 (k={K})")
for i, m in enumerate(models):
    print(f"  模型 {i}: {m.get('genome_id', 'unknown')}")

# ==================== 原始序列得分 ====================
print("\n" + "=" * 60)
print("原始序列得分（归一化: 总分 / 有效k-mer数）:")
norm_factor = max(1, len(first_seq) - K)
for i, model in enumerate(models):
    s = score_read(first_seq, model, K)
    norm_s = s / norm_factor
    gid = model.get('genome_id', f'model_{i}')
    print(f"  {gid}: raw={s:.4f}, 归一化={norm_s:.6f}")

# ==================== 打乱序列得分分布 ====================
print("\n" + "=" * 60)
print(f"随机打乱 {N_SHUFFLES} 次，统计得分分布...")

# 存储每个模型的打乱得分
shuffle_scores = {i: [] for i in range(len(models))}

for n in range(N_SHUFFLES):
    shuffled = shuffle_sequence(first_seq)
    for i, model in enumerate(models):
        s = score_read(shuffled, model, K)
        shuffle_scores[i].append(s / norm_factor)  # 归一化

    if (n + 1) % 200 == 0:
        print(f"  已完成 {n + 1}/{N_SHUFFLES}...")

# ==================== 统计分析 ====================
print("\n" + "=" * 60)
print("打乱序列得分分布统计 (归一化得分):")
print("-" * 60)

all_shuffle_scores = []
for i, model in enumerate(models):
    scores = np.array(shuffle_scores[i])
    all_shuffle_scores.extend(scores)
    gid = model.get('genome_id', f'model_{i}')

    # 原始序列在该模型上的归一化得分
    orig_score = score_read(first_seq, model, K) / norm_factor

    # z-score of original vs shuffled
    mean_s = scores.mean()
    std_s = scores.std(ddof=1)
    z = (orig_score - mean_s) / std_s if std_s > 0 else 0

    print(f"\n模型 {i} ({gid}):")
    print(f"  原始序列得分:        {orig_score:.6f}")
    print(f"  打乱序列得分均值:    {mean_s:.6f}")
    print(f"  打乱序列得分标准差:  {std_s:.6f}")
    print(f"  打乱序列得分最小值:  {scores.min():.6f}")
    print(f"  打乱序列得分最大值:  {scores.max():.6f}")
    print(f"  打乱序列得分中位数:  {np.median(scores):.6f}")
    print(f"  原始序列 z-score:    {z:.4f}")
    print(f"  原始序列百分位:      {(scores < orig_score).mean() * 100:.2f}%")

# 全部打乱得分汇总
all_arr = np.array(all_shuffle_scores)
print("\n" + "=" * 60)
print("全部打乱得分汇总 (跨所有模型):")
print(f"  均值:     {all_arr.mean():.6f}")
print(f"  标准差:   {all_arr.std(ddof=1):.6f}")
print(f"  最小值:   {all_arr.min():.6f}")
print(f"  最大值:   {all_arr.max():.6f}")
print(f"  中位数:   {np.median(all_arr):.6f}")

# ==================== 画图 ====================
print("\n" + "=" * 60)
print("生成分布图（正态性通过 QQ-plot 可视化评估）...")

# --- 图1: 每个模型的直方图 + QQ-plot (2行10列) ---
fig, axes = plt.subplots(4, 5, figsize=(22, 16))
axes = axes.flatten()
N_MODELS = len(models)

for i, model in enumerate(models):
    scores = np.array(shuffle_scores[i])
    gid = model.get('genome_id', f'model_{i}')
    short_name = gid.replace(' chromosome, complete genome', '')[:40]
    orig_score = score_read(first_seq, model, K) / norm_factor

    # 直方图 (上半部分: row 0-1, col i)
    hist_ax = axes[i]
    hist_ax.hist(scores, bins=40, alpha=0.7, color='steelblue', edgecolor='white')
    hist_ax.axvline(orig_score, color='red', linestyle='--', linewidth=2,
                    label=f'Original: {orig_score:.4f}')
    hist_ax.axvline(scores.mean(), color='orange', linestyle='-', linewidth=1,
                    label=f'Mean: {scores.mean():.4f}')
    hist_ax.set_title(f'{short_name}', fontsize=8)
    hist_ax.legend(fontsize=6, loc='upper right')
    hist_ax.tick_params(labelsize=7)
    hist_ax.set_xlabel('Normalized score', fontsize=7)
    hist_ax.set_ylabel('Frequency', fontsize=7)

    # QQ-plot (下半部分: row 2-3, col i)
    qq_ax = axes[N_MODELS + i]
    stats.probplot(scores, dist="norm", plot=qq_ax)
    qq_ax.set_title(f'{short_name} — QQ-plot', fontsize=8)
    qq_ax.tick_params(labelsize=7)
    # 获取 QQ 图的线条并设置颜色
    qq_lines = qq_ax.get_lines()
    if len(qq_lines) >= 2:
        qq_lines[0].set_markerfacecolor('steelblue')
        qq_lines[0].set_markeredgecolor('steelblue')
        qq_lines[0].set_markersize(3)
        qq_lines[1].set_color('red')
        qq_lines[1].set_linewidth(1.5)

# 隐藏多余的 subplot
for j in range(2 * N_MODELS, 4 * 5):
    axes[j].set_visible(False)

plt.suptitle(
    f'Shuffled Sequence Score Distribution (k={K}, N={N_SHUFFLES})\n'
    f'Sequence: test.fa #{first_id}, length={len(first_seq)}bp',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('shuffle_distribution_k6.png', dpi=150)
print("图片已保存: shuffle_distribution_k6.png")

# --- 图2: 所有模型得分叠加直方图 ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
for i, model in enumerate(models):
    scores = np.array(shuffle_scores[i])
    gid = model.get('genome_id', f'model_{i}')
    short_name = gid.replace(' chromosome, complete genome', '')[:30]
    ax2.hist(scores, bins=40, alpha=0.4, color=colors[i], label=short_name)

ax2.set_xlabel('Normalized Log-Likelihood Score')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Overlay of Shuffled Sequence Score Distributions\n(k={K}, N={N_SHUFFLES} shuffles)')
ax2.legend(fontsize=7)
plt.tight_layout()
plt.savefig('shuffle_distribution_overlay_k6.png', dpi=150)
print("图片已保存: shuffle_distribution_overlay_k6.png")

# --- 图3: QQ-plot 叠加 (所有模型) ---
fig3 = plt.figure(figsize=(14, 10))
grid = fig3.add_gridspec(4, 3, hspace=0.4, wspace=0.35)
qq_axes = [fig3.add_subplot(grid[i // 3, i % 3]) for i in range(N_MODELS)]
for i, model in enumerate(models):
    scores = np.array(shuffle_scores[i])
    gid = model.get('genome_id', f'model_{i}')
    short_name = gid.replace(' chromosome, complete genome', '')[:40]
    stats.probplot(scores, dist="norm", plot=qq_axes[i])
    qq_axes[i].set_title(f'{short_name}', fontsize=8)
    qq_axes[i].tick_params(labelsize=7)
    lines = qq_axes[i].get_lines()
    if len(lines) >= 2:
        lines[0].set_markerfacecolor('steelblue')
        lines[0].set_markeredgecolor('steelblue')
        lines[0].set_markersize(3)
        lines[1].set_color('red')
        lines[1].set_linewidth(1.5)

# 隐藏多余的 subplot
for j in range(N_MODELS, 4 * 3):
    fig3.delaxes(fig3.add_subplot(grid[j // 3, j % 3]))

fig3.suptitle(
    f'QQ-Plots of Shuffled Scores (k={K}, N={N_SHUFFLES})\n'
    f'Sequence: test.fa #{first_id}, length={len(first_seq)}bp',
    fontsize=13, fontweight='bold')
fig3.savefig('shuffle_qqplot_k6.png', dpi=150)
print("图片已保存: shuffle_qqplot_k6.png")

print("\n完成！")
