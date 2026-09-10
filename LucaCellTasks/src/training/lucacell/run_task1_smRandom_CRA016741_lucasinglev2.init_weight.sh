#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
seed=1221

# for dataset
DATASET_NAME="CRA016741_finetune_lucacellv2"
DATASET_TYPE="cell"

# for task
TASK_TYPE="multi_class"
TASK_LEVEL_TYPE="cell_level"
LABEL_TYPE="CRA016741_finetune_lucacellv2"

# for input
## here only embedding matrix-channel
## input_type: vector, matrix for single
INPUT_TYPE="matrix"
INPUT_MODE="single"
TRUNC_TYPE="right"

# for model
MODEL_TYPE="lucasingle"
CONFIG_NAME="lucasingle_config.json"
FUSION_TYPE="concat"
dropout_prob=0.1
fc_size=256
classifier_size=$fc_size
BEST_METRIC_TYPE="f1"
# binary-class, multi-label: bce, multi-class: cce, regression: l1 or l2
loss_type="cce"

# for cell embedding matrix
embedding_input_size=2560
# for matrix encoder
hidden_size=2560
num_attention_heads=8
num_hidden_layers=1
# none, avg, max, value_attention
matrix_pooling_type="value_attention"

# for seq llm
seq_llm_dirpath="../llm/"
seq_llm_type="lucaone"
seq_llm_version="lucaone"
seq_llm_step=60000000
seq_max_length=10242
seq_vector_dirpath="../../embedding/lucacellv2/lucacell-2048pos/v1.0/CRA016741_finetune/seq_embeddings1/#../../embedding/lucacellv2/lucacell-2048pos/v1.0/CRA016741_finetune/seq_embeddings/"
# seq_matrix_dirpath=""
seq_embedding_vector_type="mean"
seq_embedding_fixed_len_a_time=10240
seq_meta_fasta="../../data/microbe/PRJCA017256/CRA016741/sampled_renamed_seqs.all.fa"
# seq_matrix_embedding_exists
# seq_matrix_add_special_token
# seq_embedding_complete
# seq_embedding_complete_seg_overlap

# for cell llm
cell_llm_dirpath="../llm/"
cell_llm_type="lucacell"
cell_llm_version="lucacell-2048pos/v1.0"
cell_llm_task_level="express_token_mask"
# cell_llm_time_str=20260226165643
cell_llm_step=6400000
cell_max_length=9602
# cell_vector_dirpath=""
# cell_matrix_dirpath="../../embedding/lucacell/lucacell-2400/v1.0/kidney/all_cell_embeddings/"
cell_embedding_vector_type="mean"
cell_embedding_fixed_len_a_time=9600
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
gradient_accumulation_steps=16
# 间隔多少个step在log文件中写入信息（实际上是gradient_accumulation_steps与logging_steps的最小公倍数）
logging_steps=200
## checkpoint的间隔step数。-1表示按照epoch粒度保存checkpoint
save_steps=-1
## warmup_steps个step到达peak lr
warmup_steps=1000
## 最大迭代step次数(这么多次后，peak lr1变为lr2, 需要根据epoch,样本数量,n_gpu,batch_size,gradient_accumulation_steps进行估算）
## -1自动计算
max_steps=-1
## batch size for one GPU
batch_size=1
## 最大学习速率(peak learning rate)
learning_rate=2e-4
## data loading buffer size
buffer_size=128
seq_buffer_size=1280000
## pos_weight
weight=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
## loss_reduction
loss_reduction="mean"
## num_labels
num_labels=54

init_weight_filepath_for_gene_express_bin='../../LucaCellTasks/llm/models/lucacell/lucacell-2048pos/v1.0/express_token_mask/20260226165643/checkpoint-step6400000/express_bin_tokens_embed.pth'

gene_express_vocab_size=56

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
  --seq_vector_dirpath $seq_vector_dirpath \
  --seq_embedding_vector_type $seq_embedding_vector_type \
  --seq_embedding_fixed_len_a_time $seq_embedding_fixed_len_a_time \
  --seq_matrix_add_special_token \
  --seq_embedding_complete \
  --seq_embedding_complete_seg_overlap \
  --seq_meta_fasta $seq_meta_fasta \
  --cell_llm_dirpath $cell_llm_dirpath \
  --cell_llm_type $cell_llm_type \
  --cell_llm_version $cell_llm_version \
  --cell_llm_step $cell_llm_step \
  --cell_max_length $cell_max_length \
  --cell_embedding_vector_type $cell_embedding_vector_type \
  --cell_embedding_fixed_len_a_time $cell_embedding_fixed_len_a_time \
  --cell_matrix_embedding_exists \
  --cell_matrix_add_special_token \
  --cell_embedding_complete \
  --cell_embedding_complete_seg_overlap \
  --embedding_input_size $embedding_input_size \
  --trunc_type=$TRUNC_TYPE \
  --buffer_size $buffer_size \
  --seq_buffer_size $seq_buffer_size \
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
  --weight $weight \
  --num_labels $num_labels \
  --not_frozen_gene_express_bin_embedding \
  --init_weight_filepath_for_gene_express_bin $init_weight_filepath_for_gene_express_bin \
  --gene_express_vocab_size $gene_express_vocab_size

# --cell_llm_time_str $cell_llm_time_str\
