# 人类
# cd ./src/lucaone/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nohup \
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/hg38.ensGene_genes_split/hg38.ensGene_genes.part_001.fa \
    --save_path ../../../embedding/lucaone/meta/hg38.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0 \
&

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nohup \
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/hg38.ensGene_genes_split/hg38.ensGene_genes.part_002.fa \
    --save_path ../../../embedding/lucaone/meta/hg38.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 1 \
&

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nohup \
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/hg38.ensGene_genes_split/hg38.ensGene_genes.part_003.fa \
    --save_path ../../../embedding/lucaone/meta/hg38.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 2 \
&

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nohup \
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/hg38.ensGene_genes_split/hg38.ensGene_genes.part_004.fa \
    --save_path ../../../embedding/lucaone/meta/hg38.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_fixed_len_a_time 10240 \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 3 \
&

# 小鼠
# cd ./src/lucaone/
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nohup \
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/mm10.ensGene_genes_split/mm10.ensGene_genes.part_001.fa \
    --save_path ../../../embedding/lucaone/meta/mm10.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 4 \
&

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nohup \
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/mm10.ensGene_genes_split/mm10.ensGene_genes.part_002.fa \
    --save_path ../../../embedding/lucaone/meta/mm10.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 5 \
&

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nohup \
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/mm10.ensGene_genes_split/mm10.ensGene_genes.part_003.fa \
    --save_path ../../../embedding/lucaone/meta/mm10.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 6 \
&

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nohup \
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/mm10.ensGene_genes_split/mm10.ensGene_genes.part_004.fa \
    --save_path ../../../embedding/lucaone/meta/mm10.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_fixed_len_a_time 10240 \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 7 \
&

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nohup \
python get_embedding.py \
    --llm_dir ../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 60000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../meta/hg38_mm10.ensGene_genes.long_seqs.fa \
    --save_path ../../../embedding/lucaone/meta/mm10.ensGene_genes/mean_vector/ \
    --save_type tensor \
    --embedding_type vector \
    --vector_type mean \
    --matrix_add_special_token \
    --embedding_fixed_len_a_time 10240 \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 7 \
&

