## Alignment-free single-microbe embedding
# human gut 
export CUDA_VISIBLE_DEVICES="0"
step=324392 
python predict_v2.py \
    --model_path ../ \
    --dataset_name CRA016741_finetune_lucacellv2 \
    --dataset_type cell \
    --task_type multi_class \
    --task_level_type cell_level \
    --model_type lucasingle \
    --input_type matrix \
    --input_mode single \
    --time_str 20260410151230 \
    --step $step \
    --input_file ../dataset/CRA016741_finetune_lucacellv2/cell/multi_class/remaining.csv \
    --seq_fasta_filepath ../../data/microbe/PRJCA017256/CRA016741/sampled_renamed_seqs.all.fa \
    --seq_max_length 10242 \
    --seq_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/CRA016741_finetune/seq_embeddings1/#../../embedding/lucacellv2/lucacell-2048pos/v1.0/CRA016741_finetune/seq_embeddings/ \
    --cell_max_length 9602 \
    --save_path ../../predicts/CRA016741_finetune_lucacellv2_$step/remaining_prediction_results.csv \
    --output_classification_vector_dirpath ../../predicts/CRA016741_finetune_lucacellv2_$step/remaining_cell_vectors/ \
    --output_attention_pooling_scores_dirpath ../../predicts/CRA016741_finetune_lucacellv2_$step/remaining_attention/ \
    --print_per_num 1000 \
    --ground_truth_idx 6 \
    --gpu_id 0 


# bacterial calture
export CUDA_VISIBLE_DEVICES="0"
step=324392
python predict_v2.py \
    --model_path ../ \
    --dataset_name CRA016741_finetune_lucacellv2 \
    --dataset_type cell \
    --task_type multi_class \
    --task_level_type cell_level \
    --model_type lucasingle \
    --input_type matrix \
    --input_mode single \
    --time_str 20260410151230 \
    --step $step \
    --input_file ../dataset/CRA011274_finetune_lucacellv2/cell/multi_class/remaining.csv \
    --seq_fasta_filepath ../../data/microbe/PRJCA017452/CRA011274/sampled_renamed_seqs.all.fa \
    --seq_max_length 10242 \
    --seq_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/CRA011274_finetune_lucacellv2/seq_embeddings1/#../../embedding/lucacellv2/lucacell-2048pos/v1.0/CRA011274_finetune_lucacellv2/seq_embeddings/ \
    --cell_max_length 9602 \
    --save_path ../../predicts/CRA011274_finetune_lucacellv2_$step/remaining_prediction_results.csv \
    --output_classification_vector_dirpath ../../predicts/CRA011274_finetune_lucacellv2_$step/remaining_cell_vectors/ \
    --output_attention_pooling_scores_dirpath ../../predicts/CRA011274_finetune_lucacellv2_$step/remaining_attention/ \
    --print_per_num 100 \
    --ground_truth_idx 6 \
    --gpu_id 0 



## SNP-aware gene expression modeling 
# onek1k nosnp
export CUDA_VISIBLE_DEVICES="0"
step=472360 # 388080 # 566440 # 513520 # 194040
python predict_v2.py \
    --model_path ../ \
    --dataset_name onek1k_lucacellv2_expr_nosnp_cls_noXY_egenes \
    --dataset_type gene_cell \
    --task_type regression \
    --task_level_type cell_level \
    --model_type lucapair_heter \
    --input_type gene_vs_matrix \
    --input_mode pair \
    --time_str 20260421160534 \
    --step $step \
    --input_file ../dataset/onek1k_lucacellv2_expr_nosnp_cls_noXY_egenes/gene_cell/regression/test/test.csv \
    --seq_max_length 10242 \
    --seq_vector_embedding_exists \
    --seq_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/onek1k_noXY_egenes/cell_embeddings_nosnp_cls_6400000/gene_seq_embeddings/ \
    --cell_max_length 9602 \
    --cell_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/onek1k_noXY_egenes/cell_embeddings_nosnp_cls_6400000/\
    --save_path ../../predicts/onek1k_lucacellv2_expr_nosnp_cls_noXY_egenes_"$step"/test_prediction_results.csv \
    --output_classification_vector_dirpath ../../predicts/onek1k_lucacellv2_expr_nosnp_cls_noXY_egenes_"$step"/test_cell_vectors/ \
    --output_attention_pooling_scores_dirpath ../../predicts/onek1k_lucacellv2_expr_nosnp_cls_noXY_egenes_"$step"/test_attention/ \
    --print_per_num 100 \
    --ground_truth_idx 12 \
    --gpu_id 0 

# onek1k snp
export CUDA_VISIBLE_DEVICES="0"
python predict_v2.py \
    --model_path ../ \
    --dataset_name onek1k_lucacellv2_expr_snp_cls_noXY_egenes \
    --dataset_type gene_cell \
    --task_type regression \
    --task_level_type cell_level \
    --model_type lucapair_heter \
    --input_type gene_vs_matrix \
    --input_mode pair \
    --time_str 20260421160521 \
    --step $step \
    --input_file ../dataset/onek1k_lucacellv2_expr_snp_cls_noXY_egenes/gene_cell/regression/test/test.csv \
    --seq_max_length 10242 \
    --seq_vector_embedding_exists \
    --seq_emb_dir ../../data/onek1k/individual_mRNA_seq_cls_embeddings1/#../../data/onek1k/individual_mRNA_seq_cls_embeddings2/#../../data/onek1k/individual_mRNA_seq_cls_embeddings_snp1/#../../data/onek1k/individual_mRNA_seq_cls_embeddings_snp2/#../../data/onek1k/individual_mRNA_seq_cls_embeddings_snp3/ \
    --cell_max_length 9602 \
    --cell_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/onek1k_noXY_egenes/cell_embeddings_snp_cls_6400000/\
    --save_path ../../predicts/onek1k_lucacellv2_expr_snp_cls_noXY_egenes_"$step"/test_prediction_results.csv \
    --output_classification_vector_dirpath ../../predicts/onek1k_lucacellv2_expr_snp_cls_noXY_egenes_"$step"/test_cell_vectors/ \
    --output_attention_pooling_scores_dirpath ../../predicts/onek1k_lucacellv2_expr_snp_cls_noXY_egenes_"$step"/test_attention/ \
    --print_per_num 100 \
    --ground_truth_idx 12 \
    --gpu_id 0 


## viral load
export CUDA_VISIBLE_DEVICES="0"
step=58125
python predict_v2.py \
    --model_path ../ \
    --dataset_name bronchoalveolar_test2_all \
    --dataset_type gene_cell \
    --task_type regression \
    --task_level_type cell_level \
    --model_type lucapair_homo \
    --input_type matrix_vs_matrix \
    --input_mode pair \
    --time_str 20260603093815 \
    --step $step \
    --input_file ../dataset/bronchoalveolar_test2_cal07/gene_cell/regression/infer_expanded1.csv \
    --seq_max_length 10242 \
    --seq_vector_embedding_exists \
    --seq_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/bronchoalveolar/cell_embeddings_6400000/gene_seq_embeddings/ \
    --cell_max_length 9602 \
    --cell_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/bronchoalveolar/cell_embeddings_6400000/\
    --save_path ../../predicts/bronchoalveolar_test2_all_"$step"/infer_prediction_results.csv \
    --output_classification_vector_dirpath ../../predicts/bronchoalveolar_test2_all_"$step"/infer_cell_vectors/ \
    --output_attention_pooling_scores_dirpath ../../predicts/bronchoalveolar_test2_all_"$step"/infer_attention/ \
    --print_per_num 100 \
    --ground_truth_idx 12 \
    --gpu_id 0 


## cross-species cell type annotation
# human and lemur test 
export CUDA_VISIBLE_DEVICES="0"
step=1897896
python predict_v2.py \
    --model_path ../ \
    --dataset_name cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1 \
    --dataset_type cell \
    --task_type multi_class \
    --task_level_type cell_level \
    --model_type lucasingle \
    --input_type matrix \
    --input_mode single \
    --time_str 20260518111026 \
    --step $step \
    --input_file ../dataset/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1/cell/multi_class/test/test.csv \
    --seq_fasta_filepath ../../data/cellxgene/kidney/seqs.fa \
    --seq_max_length 10242 \
    --seq_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/kidney/human_ATAC_gene_seq_embeddings/#../../embedding/lucacellv2/lucacell-2048pos/v1.0/kidney/mouse_ATAC_gene_seq_embeddings/#/mnt/luca/workspace/embedding/gene_seq_embeddings_lucaone_60000000/ \
    --cell_max_length 9602 \
    --save_path ../../predicts/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1_$step/test_prediction_results.csv \
    --output_classification_vector_dirpath ../../predicts/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1_$step/test_cell_vectors/ \
    --output_attention_pooling_scores_dirpath ../../predicts/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1_$step/test_attention/ \
    --print_per_num 100 \
    --ground_truth_idx 6 \
    --gpu_id 0 

# mouse test
export CUDA_VISIBLE_DEVICES="0"
step=1897896
python predict_v2.py \
    --model_path ../ \
    --dataset_name cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1 \
    --dataset_type cell \
    --task_type multi_class \
    --task_level_type cell_level \
    --model_type lucasingle \
    --input_type matrix \
    --input_mode single \
    --time_str 20260518111026 \
    --step $step \
    --input_file ../dataset/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1/cell/multi_class/test2/test2.csv \
    --seq_fasta_filepath ../../data/cellxgene/kidney/seqs.fa \
    --seq_max_length 10242 \
    --seq_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/kidney/human_ATAC_gene_seq_embeddings/#../../embedding/lucacellv2/lucacell-2048pos/v1.0/kidney/mouse_ATAC_gene_seq_embeddings/#/mnt/luca/workspace/embedding/gene_seq_embeddings_lucaone_60000000/ \
    --cell_max_length 9602 \
    --save_path ../../predicts/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1_$step/test2_prediction_results.csv \
    --output_classification_vector_dirpath ../../predicts/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1_$step/test2_cell_vectors/ \
    --output_attention_pooling_scores_dirpath ../../predicts/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1_$step/test2_attention/ \
    --print_per_num 100 \
    --ground_truth_idx 6 \
    --gpu_id 0 

# mouse test shuffled sequences
export CUDA_VISIBLE_DEVICES="0"
step=1897896
python predict_v2.py \
    --model_path ../ \
    --dataset_name cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1 \
    --dataset_type cell \
    --task_type multi_class \
    --task_level_type cell_level \
    --model_type lucasingle \
    --input_type matrix \
    --input_mode single \
    --time_str 20260518111026 \
    --step $step \
    --input_file ../dataset/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1/cell/multi_class/test2/test2.csv \
    --seq_fasta_filepath ../../data/cellxgene/kidney/seqs.fa \
    --seq_max_length 10242 \
    --seq_emb_dir ../../embedding/lucacellv2/lucacell-2048pos/v1.0/kidney/human_ATAC_gene_seq_embeddings/#../../embedding/lucacellv2/lucacell-2048pos/v1.0/kidney/mouse_ATAC_gene_seq_embeddings/#/mnt/luca/workspace/embedding/gene_seq_embeddings_lucaone_60000000_shuffle/ \
    --cell_max_length 9602 \
    --save_path ../../predicts/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1_shuffle_$step/test2_prediction_results.csv \
    --output_classification_vector_dirpath ../../predicts/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1_shuffle_$step/test2_cell_vectors/ \
    --output_attention_pooling_scores_dirpath ../../predicts/cellxgene_kidney_finetune_lucacellv2_2_rmMouse_1_shuffle_$step/test2_attention/ \
    --print_per_num 100 \
    --ground_truth_idx 6 \
    --gpu_id 0
