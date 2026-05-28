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


K_DENSE_THRESHOLD = 12  # k <= 12 使用稠密矩阵，否则使用稀疏字典

THRESHOLD_FACTOR = -5.0  # 分类得分的阈值因子，低于该值则分类为 -1
