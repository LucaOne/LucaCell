# 人类
cd ./src/lucaone/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/hg38.ensGene_genes_split/hg38.ensGene_genes.fasta \
    --save_path ../../../embedding/lucaone/meta/hg38.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0

# 小鼠
cd ./src/lucaone/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/mm10.ensGene_genes_split/mm10.ensGene_genes.fasta \
    --save_path ../../../embedding/lucaone/meta/mm10.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 1