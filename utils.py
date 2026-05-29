# utils.py

# 碱基顺序常量
BASES = 'ACGT'

# 碱基到编码的映射
BASE2ID = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# 编码到碱基的映射
ID2BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

# gene_id到name的映射

ID2NAME={'NC_009767': 'Roseiflexus castenholzii DSM 13941 chromosome, complete genome', 'NC_015859': 'Corynebacterium variabile DSM 44702 chromosome, complete genome', 'NC_015722': 'Candidatus Midichloria mitochondrii IricVA chromosome, complete genome', 'NC_008709': 'Psychromonas ingrahamii 37 chromosome, complete genome', 'NC_007984': 'Baumannia cicadellinicola str. Hc (Homalodisca coagulata), complete genome', 'NC_011138': "Alteromonas macleodii str. 'Deep ecotype' chromosome, complete genome", 'NC_011126': 'Hydrogenobaculum sp. Y04AAS1 chromosome, complete genome', 'NC_015656': 'Frankia symbiont of Datisca glomerata chromosome, complete genome', 'NC_013943': 'Denitrovibrio acetiphilus DSM 12809 chromosome, complete genome', 'NC_009511': 'Sphingomonas wittichii RW1 chromosome, complete genome'}

# 全局伪计数 (用于拉普拉斯平滑，防止出现 log(0))
PSEUDOCOUNT = 1


K_DENSE_THRESHOLD = 11  # k <= 11 使用稠密矩阵，否则使用稀疏字典

# 显著性检验：z 值阈值。原始序列得分需超过随机打乱序列得分均值
# 至少 Z_SCORE_THRESHOLD 个标准差，才认为匹配显著（单侧检验）。
# 1.96 对应约 97.5% 置信水平（单侧），可在运行时通过命令行覆盖。
Z_SCORE_THRESHOLD = 1.645

# 显著性检验：随机打乱次数，用于构建零分布。
# 值越大检验越稳定，但耗时线性增长。推荐 50~100。
N_PERMUTATIONS = 50
