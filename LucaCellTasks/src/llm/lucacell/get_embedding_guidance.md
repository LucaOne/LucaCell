
# Parameters:
## 带默认值的参数 
* --lucaone_version: str, lucaone版本，默认使用lucaone-gene版本, 预训练也是使用该版本，可以忽略使用默认值     
* --lucaone_step: str, lucaone-gene的checkpoint，默认使用36800000, 预训练也是使用lucaone的该checkpoint，可以忽略使用默认值   
* --llm_dir: str, lucacell的checkpoint的存储路径，默认值../../ 即在该Project下，可以忽略使用默认值   
* --llm_type: str, 模型类型，默认值lucacell，可以忽略使用默认值     
* --llm_version: str, lucacell版本，这里使用lucacell-2400/v1.0，可以忽略使用默认值    
* --llm_time_str: str, lucacell预训练时间，目前默认值20250726103509，可以忽略使用默认值     
* --llm_step: str, lucacell的checkpoint，目前默认值400000，可以忽略使用默认值      
* --truncation_nucleotide_seq_length: int, 样本中基因的核酸序列最多处理多长，目前默认值10240，可以忽略使用默认值       
* --truncation_gene_seq_length: int, 样本中基因list的最多使用多少个基因，目前默认值4096，推理可以自由扩展，受限于计算资源，训练使用的长度是2400      
* --trunc_type: str, 基因核酸序列与基因list的超长了截断方式，right截断右边，left截断左边，默认right        

## 重点参数
* --input_file: str, 输入文件
input_file contains 4或者5 columns: sample_id,sample_type,sample_gene_id_list,sample_gene_seq_list(可选),sample_gene_express_list            
请指定每一列对应的列号:   
  * --sample_id_idx: int, the column index of the unique sample id，embedding后存储的文件名使用sample_id去命名    
  * --sample_typ_idx: int, the column index of the sample type (目前only RNA)          
  * --sample_gene_id_list_idx: int, the column index of the list of gene ids        
  * --sample_gene_seq_list_idx: int, the column index of the list of gene nucleotide seqs, please set the gene seqs fasta file(--sample_gene_seq_meta_fasta, multi fasta files separated by Comma) when gene seqs not in the input file     
  * --sample_gene_seq_meta_fasta: str, 如果--input_file中没有基因的核酸序列，则需要指定序列的fasta文件，程序会基于--sample_gene_id_list中的gene id去fasta中加载，如果有多个fasta文件，则使用逗号分隔，比如一个物种一个fasta文件       
  * --sample_gene_express_list_idx: int, the column index of the list of gene express bins   

* --need_head_weights: store_true, 是否返回attention scores, shape: n_layers x n_heads x L x L, 如果是则保存在文件名为: ${save_path}/attentions/attention_{sample_id}.pt 文件
* --save_path: str, embedding保存路径        
* --embedding_type: str, embedding类型:matrix or vector     
* --save_type: str, embedding保存类型:numpy or tensor       
* --matrix_add_special_token: store_true, 如果--embedding_type是matrix，矩阵首尾行是否加上[CLS]与[EOS]两个特殊token的embedding向量         
* --embedding_complete: store_true, 是否将gene list全部进行embedding，如果GPU显存不足，则会调用cpu，cpu内存不足则会爆错           
* --embedding_fixed_len_a_time_for_seq, --embedding_fixed_len_a_time_for_cell: int, 如果报错请指定这两个参数，分别是序列embedding一次性长度与gene list个数的embedding一次性长度，超过则分片，A100，如果用显存则序列最长一次性可以3400，gene list一次性可以4096       
* --embedding_complete_seg_overlap: store_true, 分片是否进行overlap分片，建议使用         
* --gpu_id: int, 使用显卡的索引，-1代表使用CPU      
