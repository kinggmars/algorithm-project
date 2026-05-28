# data_loader.py
# 有九个函数
# count_fasta_sequences统计fasta序列中序列的总条数
# read_fasta_one_by_one通过生成器的方式，一次返回一个序列，返回的是元组形式，序列编号+序列
# read_fasta_all输出所有序列
# get_reverse_complement得到反转序列
# load_seq_id_map读取seq_id.map文件，返回字典 {read_id: true_genome_name}
# merge_genome合并基因组序列，可选是否合并正反链，注意合并正反链的时候中间加了20个N，在统计kmer的时候可能要用到
# get_genome_files获取指定目录下所有的参考基因组文件路径，返回列表 [(genome_id, full_path),...]
#id_to_name函数获取gene_id到物种名的函数
#convert_map_to_numeric转换较复杂的map 

import os
import re

def count_fasta_sequences(filepath: str) -> int:
    """
    快速统计 FASTA 文件中的序列总条数。
    """
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count

def read_fasta_one_by_one(file_path: str, keep_full_header: bool = False):    
    """
    生成器：逐条读取 FASTA 文件。
    
    参数:
    - file_path: 文件路径
    - keep_full_header: 
        如果为 False (默认，适用于 reads.fa和test.fa)，将 ">0" 解析为 "0"。
        如果为 True (适用于 fna)，将保留 ">" 之后的整行内容。
    """
    with open(file_path, 'r') as f:
        seq_id = None  
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None: 
                    yield seq_id, "".join(seq_lines)
                
                if keep_full_header:
                    seq_id = line[1:].strip()
                else:
                    parts = line[1:].split()
                    seq_id = parts[0] if parts else "" 
                    
                seq_lines = []
            else:
                seq_lines.append(line.upper()) 
                
        if seq_id is not None:
            yield seq_id, "".join(seq_lines)

def read_fasta(file_path: str, keep_full_header: bool = False) -> list:
    """
    读取 FASTA 文件，返回 序列ID + 序列 的字符串数组。
    
    参数:
    - file_path: 文件路径
    - keep_full_header: 
        如果为 False (默认，适用于 reads.fa和test.fa)，将 ">0" 解析为 "0"。
        如果为 True (适用于 fna)，将保留 ">" 之后的整行内容。
    
    返回:
    - list: 每个元素是 (序列ID, 序列字符串)
    """
    result = []  
    with open(file_path, 'r') as f:
        seq_id = None  
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None: 
                    result.append((seq_id, "".join(seq_lines)))
                
                if keep_full_header:
                    seq_id = line[1:].strip()
                else:
                    parts = line[1:].split()
                    seq_id = parts[0] if parts else "" 
                    
                seq_lines = []
            else:
                seq_lines.append(line.upper()) 
                
        if seq_id is not None:
            result.append((seq_id, "".join(seq_lines)))
    
    return result  # 返回数组

def get_reverse_complement(seq: str) -> str:
    """获取 DNA 序列的反向互补链，完整支持 IUPAC 模糊碱基。"""
    COMPLEMENT = str.maketrans(
        'ACGTacgtRYSWKMBDHVNryswkmbdhvn',
        'TGCAtgcaYRSWMKVHDBNyrsWMkvhdBn'
    )
    return seq.translate(COMPLEMENT)[::-1]

def load_seq_id_map(map_file: str) -> dict[str, str]:
    """
    读取 seq_id.map，返回字典 {read_id: true_genome_name}
    
    """
    mapping = {}
    with open(map_file, 'r') as f:
        for line in f:
            line = line.rstrip('\n') 
            if not line.strip():     
                continue
            
            parts = line.split('\t', maxsplit=1) 
            
            if len(parts) == 2:
                read_id = parts[0].strip()
                genome_desc = parts[1].strip()
                mapping[read_id] = genome_desc
                
    return mapping

def merge_genome(filepath: str, use_reverse_complement: bool = True) -> str:
    """
    将一个基因组文件中的所有 contig (序列片段) 拼接成单一字符串。
    可选是否追加反向互补链。
    """
    seq_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                seq_lines.append(line.upper())
    
    forward_seq = "".join(seq_lines)
    
    if use_reverse_complement:
        rev_comp_seq = get_reverse_complement(forward_seq)
        # 插入 20 个 N 确保在 k=3~20 的测试中，滑动窗口跨过边界。
        separator = "N" * 20 
        return forward_seq + separator + rev_comp_seq
    else:
        return forward_seq

def get_genome_files(genomes_dir: str) -> list[tuple[str, str]]:
    """
    获取指定目录下所有的参考基因组文件路径。
    使用文件名(不含后缀)作为 key，而不是 fna 内的 header。
    """
    genome_files = []
    for file in os.listdir(genomes_dir):
        if file.endswith(".fna"):
            full_path = os.path.join(genomes_dir, file)
            genome_id = os.path.splitext(file)[0] 
            genome_files.append((genome_id, full_path))
    return genome_files

def id_to_name(genomes_dir: str) -> dict[str, str]:
    id2name = {}
    for genome_id, full_path in get_genome_files(genomes_dir):
        with open(full_path, "r") as f:
            first_line = f.readline().strip()
            if first_line.startswith(">"):
                name = first_line.split("|")[-1].strip()
            else:
                name = first_line
        id2name[genome_id] = name
    return id2name

def convert_map_to_numeric(original_map_file: str, genome_dir: str, output_map_file: str) -> None:
    """
    将原始 seq_id.map (每行: read_id + 物种全称) 转换为 genome_id 格式 (每行: read_id genome_id)
    这里的 genome_id 是参考基因组文件的名称（不含 .fna 后缀），而非数字索引。

    通过物种名模糊匹配自动建立映射。
    """
 
    id2name = id_to_name(genome_dir)                     # 如 {'NC_009767': 'Frankia sp. ...'}
    if not id2name:
        raise RuntimeError("未找到任何基因组文件，请检查 genomes 目录。")


    def normalize(text: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

    fna_norm = {gid: normalize(name) for gid, name in id2name.items()}

    # 读取原始 map，获取所有唯一的物种描述
    raw_mapping = load_seq_id_map(original_map_file)     # {read_id: species_description}
    unique_species = set(raw_mapping.values())

    #建立 物种描述 -> genome_id (字符串) 的映射
    species_to_gid = {}
    for sp_desc in unique_species:
        sp_norm = normalize(sp_desc)
        matched_gid = None
        # 双向包含检查
        for gid, fn in fna_norm.items():
            if fn and (fn in sp_norm or sp_norm in fn):
                matched_gid = gid
                break
        if matched_gid is not None:
            species_to_gid[sp_desc] = matched_gid
        else:
            # 匹配失败时抛出错误，避免产生无效映射（因为物种数应该与基因组数一致）
            raise ValueError(
                f"无法将物种描述 '{sp_desc}' 匹配到任何 .fna 文件。\n"
                f"请检查基因组文件头信息与 seq_id.map 中的物种名是否对应。"
            )

    with open(output_map_file, 'w') as fout:
        for read_id, sp_desc in raw_mapping.items():
            gid = species_to_gid[sp_desc]
            fout.write(f"{read_id} {gid}\n")

    print(f"Converted map file saved to {output_map_file}")
    print(f"Species-to-genome mapping: {species_to_gid}")

if __name__ == '__main__':
    print(id_to_name("./proj1/genomes"))
    convert_map_to_numeric("./proj1/seq_id.map", "./proj1/genomes", "./proj1/seq_id_numeric.map")