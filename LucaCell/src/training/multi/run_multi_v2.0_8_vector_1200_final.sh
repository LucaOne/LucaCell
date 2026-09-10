#!/bin/bash
# for dataset
## the dataset name
dataset_name="lucacell-2048pos"
## the dataset version
dataset_version="v1.0"
# mlm_probability
mlm_probability=0.50

# for pretrain tasks
# 预训练任务
pretrain_task_name="express_token_mask"
# 预训练任务的权重
pretrain_task_weight="express_token_mask:1.0"

alphabet_type="nucleotide"

nucleotide_input_type="embedding_vector"

global_nucleotide_seqs_filepath="../meta/hg38.ensGene_genes.fa;../meta/mm10.ensGene_genes.fa"

global_nucleotide_seq_embeddings_dirpath="../../embedding/lucaone/meta/hg38.ensGene_genes/mean_vector/;../../embedding/lucaone/meta/mm10.ensGene_genes/mean_vector/"

global_gene_positions_filepath="../meta/gene_loc_human.csv;../meta/gene_loc_mouse.csv"

## max seq length
max_nucleotide_seq_length=10242
max_gene_seq_length=1202
max_express_sorted_position_embeddings=50
## truncation strategy
TRUNCATION_TYPE="right"

# for dataloader
# 流式加载的缓存大小
buffer_size=1024000
embedding_buffer_size=2048000
# 数据加载器worker数
worker_num=1
# seed 用于复现或者中间训练出错保护现场(因为是伪随机)
seed=1111

# for model
## model type
model_type="lucacell"
## embedding dim
embed_dim=2560
## ffn size
ffn_size=10240
nucleotide_token_encoder_num_attention_heads=0
nucleotide_token_encoder_num_layers=0
## num layers(transformer blocks)
num_layers=40
## num heads(transformer heads)
num_attention_heads=40
## pooling type(for the nucleotide sequence embedding matrix of LucaOne)
nucleotide_token_pooling_type="mean"
best_metric_type="loss"

# for training
## max epochs
num_train_epochs=3
## accumulation gradient steps
gradient_accumulation_steps=32
# 间隔多少个step在log文件中写入loss（实际上是gradient_accumulation_steps与loss_logging_steps的最小公倍数, 这里是4000）
loss_logging_steps=1000
# 间隔多少个step在log文件中写入信息（实际上是gradient_accumulation_steps与logging_steps的最小公倍数, 这里是32000）
logging_steps=1000
# checkpoint的间隔step数
save_steps=100000
# warmup_steps个step到达peak lr，实际上是warmup_steps=warmup_steps/gradient_accumulation_steps
warmup_steps=32000
# 最大迭代step次数(这么多次后，peak lr1变为lr2, 需要根据epoch,样本数量,n_gpu,batch_size,gradient_accumulation_steps进行估算）
# 最后想要变成多大的值比如从lr1->lr2，那么就是(max_epochs*sample_cnt)*lr1/(n_gpu * batch_size * gradient_accumulation_steps*(lr1 - lr2))进行估算
# 85,000,000: (3 * 85000000 * 2e-4) /(8 * 1 * 32 * (2e-4 - 1e-5))=1048519.7368421052
max_steps=1000000
# batch size for one GPU
batch_size=1
# 最大学习速率(peak learning rate)
learning_rate=2e-4

time_str=$(date "+%Y%m%d%H%M%S")

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


cd ../..
python -W ignore -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node=8 \
       run.py \
       --time_str $time_str \
       --tb_log_dir ../tb-logs/$model_type/$dataset_name/$dataset_version/$pretrain_task_name/$time_str \
       --log_dir ../logs/$model_type/$dataset_name/$dataset_version/$pretrain_task_name/$time_str \
       --output_dir ../models/$model_type/$dataset_name/$dataset_version/$pretrain_task_name/$time_str \
       --global_nucleotide_seqs_filepath  $global_nucleotide_seqs_filepath \
       --global_nucleotide_seq_embeddings_dirpath $global_nucleotide_seq_embeddings_dirpath \
       --global_gene_positions_filepath $global_gene_positions_filepath \
       --alphabet_type $alphabet_type \
       --mlm_probability $mlm_probability \
       --embed_dim $embed_dim \
       --ffn_size $ffn_size \
       --nucleotide_token_encoder_num_attention_heads $nucleotide_token_encoder_num_attention_heads \
       --nucleotide_token_encoder_num_layers $nucleotide_token_encoder_num_layers \
       --num_layers $num_layers \
       --num_attention_heads $num_attention_heads \
       --max_nucleotide_seq_length $max_nucleotide_seq_length \
       --max_express_sorted_position_embeddings $max_express_sorted_position_embeddings \
       --max_gene_seq_length $max_gene_seq_length \
       --add_special_tokens \
       --truncation $TRUNCATION_TYPE \
       --no_nucleotide_token_embeddings \
       --no_nucleotide_position_embeddings \
       --no_nucleotide_token_encoder \
       --no_gene_positions_embeddings \
       --no_express_sorted_embeddings \
       --no_gene_type_embeddings \
       --use_rotary_embeddings \
       --use_last_layer_norm \
       --dropout_prob 0.0 \
       --attention_probs_dropout_prob 0.05 \
       --nucleotide_token_pooling_type $nucleotide_token_pooling_type \
       --model_type $model_type \
       --model_config ../config/${model_type}_config.json \
       --train_data_dir ../dataset/$dataset_name/$dataset_version/train/ \
       --val_data_dir ../dataset/$dataset_name/$dataset_version/val/ \
       --test_data_dir ../dataset/$dataset_name/$dataset_version/test/ \
       --ignore_index -100 \
       --buffer_size $buffer_size \
       --embedding_buffer_size $embedding_buffer_size \
       --worker_num $worker_num \
       --seed $seed \
       --per_gpu_train_batch_size $batch_size \
       --per_gpu_eval_batch_size $batch_size \
       --learning_rate $learning_rate \
       --weight_decay 0.01 \
       --max_grad_norm 1.0 \
       --num_train_epochs $num_train_epochs \
       --max_steps $max_steps \
       --warmup_steps $warmup_steps \
       --beta1 0.9 \
       --beta2 0.98 \
       --do_train \
       --do_eval \
       --do_test \
       --do_metrics \
       --evaluate_during_training \
       --eval_start_epoch 1 \
       --loss_logging_steps $loss_logging_steps \
       --logging_steps $logging_steps \
       --save_steps $save_steps \
       --gradient_accumulation_steps $gradient_accumulation_steps \
       --scheduler_type step \
       --best_metric_type $best_metric_type \
       --pretrain_task_name $pretrain_task_name \
       --pretrain_task_weight $pretrain_task_weight \
       --nucleotide_input_type $nucleotide_input_type \
       --express_bin_list_filepath ../meta/express_bin_list.meta \
       --processed_sample_cnt $save_steps \
       --pretrained_model_name lucaone \
       --use_bf16