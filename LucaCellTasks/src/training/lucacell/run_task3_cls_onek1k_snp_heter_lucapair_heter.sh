#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
seed=1221

# for dataset
DATASET_NAME="onek1k_lucacellv2_expr_snp_cls_noXY_egenes"
DATASET_TYPE="gene_cell"

# for task
TASK_TYPE="regression"
TASK_LEVEL_TYPE="cell_level"
LABEL_TYPE="onek1k_lucacellv2_expr_snp_cls_noXY_egenes"

# for input
## here only embedding matrix-channel
## input_type: vector_vs_vector, vector_vs_matrix/matrix_vs_vector. gene_vs_matrix/matrix_vs_gene, matrix_vs_matrix for pair
INPUT_TYPE="gene_vs_matrix"
INPUT_MODE="pair"
TRUNC_TYPE="right"

# for model
MODEL_TYPE="lucapair_heter"
CONFIG_NAME="lucapair_heter_config.json"
FUSION_TYPE="concat"
dropout_prob=0.1
fc_size=1024
classifier_size=$fc_size
BEST_METRIC_TYPE="sp_statistic"
# binary-class, multi-label: bce, multi-class: cce, regression: l1 or l2
loss_type="l2"


# for cell embedding matrix
embedding_input_size_a=2560
embedding_input_size_b=2560
# for matrix encoder
hidden_size=2560
num_attention_heads=8
num_hidden_layers=2
# none, avg, max, value_attention
matrix_pooling_type="value_attention"

# for seq llm
seq_llm_dirpath="../llm/"
seq_llm_type="lucaone"
seq_llm_version="lucaone"
seq_llm_step=60000000
seq_max_length=10242
seq_vector_dirpath="../../data/onek1k/individual_mRNA_seq_cls_embeddings_hvgs/"
# seq_matrix_dirpath="./../embedding/gene/lucaone-gene/36800000/matrix/"
seq_embedding_vector_type="mean"
seq_embedding_fixed_len_a_time=4096
# seq_meta_fasta="../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/genes.fa"
# seq_meta_fasta="xxxx"
# seq_matrix_embedding_exists
# seq_matrix_add_special_token
# seq_embedding_complete
# seq_embedding_complete_seg_overlap

# for cell llm
cell_llm_dirpath="../llm/"
cell_llm_type="lucacell"
cell_llm_version="lucacell-2048pos/v1.0"
#cell_llm_task_level="express_token_mask"
cell_llm_time_str=20260226165643
cell_llm_step=6400000
cell_max_length_a=4098
cell_max_length_b=4098
# cell_vector_dirpath="xxxx"
cell_matrix_dirpath="../../embedding/lucacellv2/lucacell-2048pos/v1.0/onek1k_noXY_egenes/cell_embeddings_snp_cls_6400000/"
cell_embedding_vector_type="mean"
cell_embedding_fixed_len_a_time=4096
# not_cell_prepend_bos
# not_cell_append_eos
# cell_matrix_embedding_exists
# cell_matrix_add_special_token
# cell_embedding_complete
# cell_embedding_complete_seg_overlap

# for training
## max epochs
num_train_epochs=1000
## accumulation gradient steps
gradient_accumulation_steps=1
# 间隔多少个step在log文件中写入信息（实际上是gradient_accumulation_steps与logging_steps的最小公倍数）
logging_steps=1000
## checkpoint的间隔step数。-1表示按照epoch粒度保存checkpoint
save_steps=-1
## warmup_steps个step到达peak lr
warmup_steps=1000
## 最大迭代step次数(这么多次后，peak lr1变为lr2, 需要根据epoch,样本数量,n_gpu,batch_size,gradient_accumulation_steps进行估算）
## -1自动计算
max_steps=-1
## batch size for one GPU
batch_size=16
## 最大学习速率(peak learning rate)
learning_rate=1e-4
## data loading buffer size
buffer_size=1024
## pos_weight
# weight=xxxx
## loss_reduction
loss_reduction="mean"
## num_labels
num_labels=114
dynamic_weighted_mse_threshold=0.025
dynamic_weighted_mse_low_weight=1
dynamic_weighted_mse_high_weight=5

time_str=$(date "+%Y%m%d%H%M%S")
cd ../../
python run.py \
  --train_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/train/ \
  --val_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/val/ \
  --test_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/test/ \
  --dataset_name $DATASET_NAME \
  --dataset_type $DATASET_TYPE \
  --task_type $TASK_TYPE \
  --task_level_type $TASK_LEVEL_TYPE \
  --model_type $MODEL_TYPE \
  --input_type $INPUT_TYPE \
  --input_mode $INPUT_MODE \
  --label_type $LABEL_TYPE \
  --label_filepath ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
  --output_dir ../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --log_dir ../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --tb_log_dir ../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --config_path ../config/$MODEL_TYPE/$CONFIG_NAME \
  --matrix_pooling_type $matrix_pooling_type \
  --fusion_type $FUSION_TYPE \
  --do_train \
  --do_eval \
  --do_predict \
  --do_metrics \
  --evaluate_during_training \
  --per_gpu_train_batch_size=$batch_size \
  --per_gpu_eval_batch_size=$batch_size  \
  --gradient_accumulation_steps=$gradient_accumulation_steps \
  --learning_rate=$learning_rate \
  --lr_update_strategy step \
  --num_train_epochs=$num_train_epochs \
  --seed $seed \
  --loss_type $loss_type \
  --loss_reduction $loss_reduction \
  --best_metric_type $BEST_METRIC_TYPE \
  --seq_llm_dirpath $seq_llm_dirpath \
  --seq_llm_type $seq_llm_type \
  --seq_llm_version $seq_llm_version \
  --seq_llm_step $seq_llm_step \
  --seq_max_length $seq_max_length \
  --seq_embedding_vector_type $seq_embedding_vector_type \
  --seq_embedding_fixed_len_a_time $seq_embedding_fixed_len_a_time \
  --seq_matrix_add_special_token \
  --seq_embedding_complete \
  --seq_embedding_complete_seg_overlap \
  --seq_vector_dirpath $seq_vector_dirpath \
  --cell_llm_dirpath $cell_llm_dirpath \
  --cell_llm_type $cell_llm_type \
  --cell_llm_version $cell_llm_version \
  --cell_llm_step $cell_llm_step \
  --cell_max_length_a $cell_max_length_a \
  --cell_max_length_b $cell_max_length_b \
  --cell_matrix_dirpath $cell_matrix_dirpath \
  --cell_embedding_vector_type $cell_embedding_vector_type \
  --cell_embedding_fixed_len_a_time $cell_embedding_fixed_len_a_time \
  --cell_matrix_embedding_exists \
  --cell_matrix_add_special_token \
  --cell_embedding_complete \
  --cell_embedding_complete_seg_overlap \
  --embedding_input_size_a $embedding_input_size_a \
  --embedding_input_size_b $embedding_input_size_b \
  --trunc_type=$TRUNC_TYPE \
  --buffer_size $buffer_size \
  --save_all \
  --ignore_index -100 \
  --hidden_size $hidden_size \
  --num_attention_heads $num_attention_heads \
  --num_hidden_layers $num_hidden_layers \
  --dropout_prob $dropout_prob \
  --vector_fc_size null \
  --matrix_fc_size null \
  --classifier_size $classifier_size \
  --emb_activate_func gelu \
  --fc_activate_func gelu \
  --classifier_activate_func gelu \
  --warmup_steps $warmup_steps \
  --beta1 0.9 \
  --beta2 0.99 \
  --weight_decay 0.01 \
  --save_steps $save_steps \
  --max_steps $max_steps \
  --logging_steps $logging_steps \
  --num_labels $num_labels \
  --dynamic_weighted_mse_threshold $dynamic_weighted_mse_threshold \
  --dynamic_weighted_mse_low_weight $dynamic_weighted_mse_low_weight \
  --dynamic_weighted_mse_high_weight $dynamic_weighted_mse_high_weight

#  --seq_matrix_dirpath $seq_matrix_dirpath \
#  --seq_meta_fasta $seq_meta_fasta \
 
