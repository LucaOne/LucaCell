#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/11/26 21:02
@project: LucaOnePlusTasks
@file: run
@desc: model building main
'''
import sys
import copy, json
import logging
import argparse
from collections import OrderedDict
from datetime import timedelta
from datasets import load_dataset
import torch.distributed as dist
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
from torch.utils.data.dataloader import DataLoader
from datasets.distributed import split_dataset_by_node
try:
    from llm.lucaone.v2_0.alphabet import Alphabet as AlphabetLucaOne
    from llm.lucacell.models.alphabet import Alphabet as AlphabetLucaCell
    from common.metrics import metrics_multi_class, metrics_binary
    from common.multi_label_metrics import *
    from utils import set_seed, save_labels, get_parameter_number, get_labels, load_trained_model, \
        download_trained_checkpoint_lucaone_v2, download_trained_checkpoint_lucacell
    from multi_files_stream_dataloader import *
    from trainer import train
    from evaluator import evaluate
    from tester import test
    from llm.lucacell.get_embedding import load_model as load_lucacell
    from lucasingle.models.LucaSingle import LucaSingle
    from lucapair.models.LucaPairHomo import LucaPairHomo
    from lucapair.models.LucaPairHeter import LucaPairHeter
    from lucapair.models.LucaPairIntraInter import LucaPairIntraInter
    from llm.lucacell.models.lucacell_for_downstream_tasks import LucaCellForDownstreamTasks
    from common.model_config import LucaConfig
    from encoder import Encoder
    from batch_converter import BatchConverter
except ImportError:
    from src.llm.lucaone.v2_0.alphabet import Alphabet as AlphabetLucaOne
    from src.llm.lucacell.models.alphabet import Alphabet as AlphabetLucaCell
    from src.common.metrics import metrics_multi_class, metrics_binary
    from src.common.multi_label_metrics import *
    from src.utils import set_seed, save_labels, get_parameter_number, get_labels, load_trained_model, \
        download_trained_checkpoint_lucaone_v2, download_trained_checkpoint_lucacell
    from src.trainer import train
    from src.evaluator import evaluate
    from src.tester import test
    from src.llm.lucacell.get_embedding import load_model as load_lucacell
    from src.lucasingle.models.LucaSingle import LucaSingle
    from src.lucapair.models.LucaPairHomo import LucaPairHomo
    from src.lucapair.models.LucaPairHeter import LucaPairHeter
    from src.lucapair.models.LucaPairIntraInter import LucaPairIntraInter
    from src.llm.lucacell.models.lucacell_for_downstream_tasks import LucaCellForDownstreamTasks
    from src.common.model_config import LucaConfig
    from src.encoder import Encoder
    from src.batch_converter import BatchConverter
    from src.multi_files_stream_dataloader import *


logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser("Running")
    parser.add_argument(
        "--train_data_dir",
        default=None,
        type=str,
        required=True,
        help="the train dataset dirpath."
    )
    parser.add_argument(
        "--val_data_dir",
        default=None,
        type=str,
        required=True,
        help="the val dataset dirpath."
    )
    parser.add_argument(
        "--test_data_dir",
        default=None,
        type=str,
        help="the train dataset dirpath."
    )
    parser.add_argument(
        "--buffer_size",
        default=1024,
        type=int,
        help="how many samples are loaded into memory at once"
    )
    parser.add_argument(
        "--seq_buffer_size",
        default=10240,
        type=int,
        help="how many samples are loaded into memory at once"
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=True,
        help="dataset name"
    )
    parser.add_argument(
        "--dataset_type",
        default="protein",
        type=str,
        required=True,
        choices=[
            "cell",
            "cell_cell",
            "gene_cell"
        ],
        help="dataset type"
    )
    parser.add_argument(
        "--task_type",
        default="binary_class",
        type=str,
        required=True,
        choices=[
            "multi_label",
            "multi_class",
            "binary_class",
            "regression"
        ],
        help="task type"
    )
    parser.add_argument(
        "--task_level_type",
        default="cell_level",
        type=str,
        required=True,
        choices=[
            "gene_level",
            "span_level",
            "cell_level"
        ],
        help="task level type"
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        choices=[
            "lucacell_finetune",
            "lucasingle",
            "lucapair_homo",
            "lucapair_heter",
            "lucapair_intrainter"
        ],
        help="the model type of selected"
    )
    parser.add_argument(
        "--input_type",
        default=None,
        type=str,
        required=True,
        choices=[
            "vector",
            "matrix",
            "vector_vs_vector",
            "vector_vs_matrix",
            "gene_vs_matrix",
            "matrix_vs_gene",
            "matrix_vs_vector",
            "matrix_vs_matrix"
        ],
        help="the input type of selected"
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="single",
        choices=["single", "pair"],
        help="the input mode"
    )

    parser.add_argument(
        "--label_type",
        default=None,
        type=str,
        required=True,
        help="label type"
    )
    parser.add_argument(
        "--label_filepath",
        default=None,
        type=str,
        required=True,
        help="the label list filepath"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="the output dirpath"
    )

    parser.add_argument(
        "--log_dir",
        default="./logs/",
        type=str,
        required=True,
        help="log dir."
    )
    parser.add_argument(
        "--tb_log_dir",
        default="./tb-logs/",
        type=str,
        required=True,
        help="tensorboard log dir."
    )

    # Other parameters
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        required=True,
        help="the config filepath of the running model"
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="cache dirpath"
    )

    parser.add_argument(
        "--position_embedding_type",
        default="RoPE",
        type=str,
        choices=["absolute", "RoPE"],
        help="the position embedding type."
    )

    parser.add_argument(
        "--matrix_pooling_type",
        type=str,
        default=None,
        choices=[
            "none",
            "first",
            "last",
            "sum",
            "max",
            "avg",
            "attentive",
            "attention",
            "context_attention",
            "weighted_attention",
            "value_attention",
            "transformer"
        ],
        help="pooling type for embedding encoder"
    )
    # fusion type
    parser.add_argument(
        "--fusion_type",
        default="concat",
        type=str,
        required=True,
        choices=["concat", "add"],
        help="the fusion type"
    )

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="whether to run eval on the val set."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to run predict on the test set."
    )
    parser.add_argument(
        "--do_metrics",
        action="store_true",
        help="whether to eval metrics on the val and test set."
    )

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="evaluation during training at each logging step."
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=50,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="Set total number of training steps to perform."
    )
    parser.add_argument(
        "--warmup_steps",
        default=-1,
        type=int,
        help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        type=float,
        help="Adamw beta1."
    )
    parser.add_argument(
        "--beta2",
        default=0.98,
        type=float,
        help="Adamw beta2."
    )
    parser.add_argument(
        "--lr_update_strategy",
        default="step",
        choices=["step", "epoch"],
        type=str,
        help="Learning rate update strategy."
    )
    parser.add_argument(
        "--lr_decay_rate",
        default=0.95,
        type=float,
        help="Learning rate decay rate when lr_update_strategy==epoch."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=-1,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=-1,
        help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
             "See details at https://nvidia.github.io/apex/amp.html"
    )
    # for GPU, 单卡默认为-1，不需要显示的设置
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="main node local rank, for pytorch<1.9."
    )
    parser.add_argument(
        "--local-rank",
        default=-1,
        type=int,
        help="main node local rank, for pytorch>=1.9."
    )

    # multi-label/binary-class
    parser.add_argument(
        "--sigmoid",
        action="store_true",
        help="Classifier add sigmoid if task_type is binary-class or multi-label"
    )

    # loss func
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce",
        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce", "l1", "l2", "dynamic_weighted_mse"],
        help="Loss type"
    )

    # for dynamic_weighted_mse
    parser.add_argument(
        "--dynamic_weighted_mse_threshold",
        type=float,
        default=0.5,
        help="the threshold for dynamic_weighted_mse"
    )
    parser.add_argument(
        "--dynamic_weighted_mse_low_weight",
        type=float,
        default=1.0,
        help="the low_weight for dynamic_weighted_mse"
    )
    parser.add_argument(
        "--dynamic_weighted_mse_high_weight",
        type=float,
        default=5.0,
        help="the high_weight for dynamic_weighted_mse"
    )

    # which metric for model finalization selected
    parser.add_argument(
        "--best_metric_type",
        type=str,
        default="f1",
        choices=[
            "loss",
            "acc",
            "jaccard",
            "prec",
            "recall",
            "f1",
            "fmax",
            "roc_auc",
            "pr_auc",
            "mcc",
            "sp_statistic",
            "ps_statistic",
            "mse",
            "mae",
            "r2"
        ],
        help="which metric for model selected"
    )
    # for BCE Loss
    parser.add_argument(
        "--pos_weight",
        type=str,
        default="1.0",
        help="positive weight for bce"
    )

    # for CE Loss
    parser.add_argument(
        "--weight",
        type=str,
        default=None,
        help="every label weight for multi-class"
    )

    # for focal Loss
    parser.add_argument(
        "--focal_loss_alpha",
        type=float,
        default=0.7,
        help="focal loss alpha"
    )
    parser.add_argument(
        "--focal_loss_gamma",
        type=float,
        default=2.0,
        help="focal loss gamma"
    )
    parser.add_argument(
        "--focal_loss_reduce",
        action="store_true",
        help="mean for one sample(default sum)"
    )

    # for asymmetric Loss
    parser.add_argument(
        "--asl_gamma_neg",
        type=float,
        default=4.0,
        help="negative gamma for asl"
    )
    parser.add_argument(
        "--asl_gamma_pos",
        type=float,
        default=1.0,
        help="positive gamma for asl"
    )

    parser.add_argument(
        "--seq_no_token_embeddings",
        action="store_true",
        help="Whether not to use token_embeddings"
    )
    parser.add_argument(
        "--seq_no_position_embeddings",
        action="store_true",
        help="Whether not to use position_embeddings"
    )
    parser.add_argument(
        "--seq_no_token_type_embeddings",
        action="store_true",
        help="Whether not to use token_type_embeddings"
    )

    parser.add_argument(
        "--seq_position_embedding_type",
        default="RoPE",
        type=str,
        choices=["absolute", "RoPE"],
        help="the position embedding type."
    )
    parser.add_argument(
        "--use_rotary_position_embeddings_for_cross",
        action="store_true",
        help="whether not to use rope for cross attention"
    )

    parser.add_argument(
        "--cell_no_token_embeddings",
        action="store_true",
        help="Whether not to use token_embeddings"
    )
    parser.add_argument(
        "--cell_no_position_embeddings",
        action="store_true",
        help="Whether not to use position_embeddings"
    )
    parser.add_argument(
        "--cell_no_token_type_embeddings",
        action="store_true",
        help="Whether not to use token_type_embeddings"
    )

    parser.add_argument(
        "--cell_position_embedding_type",
        default="RoPE",
        type=str,
        choices=["absolute", "RoPE"],
        help="the position embedding type."
    )
    parser.add_argument(
        "--use_embed_layer_norm",
        action="store_true",
        help="whether not to use emb layer norm"
    )
    # for embedding input
    parser.add_argument(
        "--embedding_input_size",
        default=None,
        type=int,
        help="the length of input embedding dim."
    )
    parser.add_argument(
        "--matrix_encoder",
        action="store_true",
        help="Whether to use matrix encoder"
    )
    parser.add_argument(
        "--matrix_encoder_act",
        action="store_true",
        help="Whether to use matrix encoder activate function"
    )

    parser.add_argument(
        "--trunc_type",
        default="right",
        type=str,
        required=True,
        choices=["left", "right"],
        help="truncate type for whole input"
    )

    # 再次训练加载已经训练好的模型
    parser.add_argument(
        "--model_dirpath",
        default=None,
        type=str,
        help="load the trained model to continue training."
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="save all check-point"
    )
    parser.add_argument(
        "--delete_old",
        action="store_true",
        help="delete old check-point"
    )

    # encoder
    parser.add_argument(
        "--hidden_size",
        default=None,
        type=int,
        help="hidden size for encoder."
    )
    parser.add_argument(
        "--intermediate_size",
        default=None,
        type=int,
        help="hidden size for encoder."
    )
    parser.add_argument(
        "--num_attention_heads",
        default=None,
        type=int,
        help="num attention_heads for encoder"
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=None,
        type=int,
        help="num hidden_layers for encoder"
    )
    parser.add_argument(
        "--dropout_prob",
        default=None,
        type=float,
        help="dropout prob for encoder"
    )

    # classifier
    parser.add_argument(
        "--classifier_size",
        default=None,
        type=int,
        help="hidden size for classifier."
    )

    # seq llm
    parser.add_argument(
        "--seq_llm_dirpath",
        default=None,
        type=str,
        help="llm dir."
    )
    parser.add_argument(
        "--seq_llm_type",
        type=str,
        default="lucaone",
        choices=["lucaone"],
        help="lucaone llm version"
    )
    parser.add_argument(
        "--seq_llm_version",
        type=str,
        default="lucaone-gene",
        choices=["lucaone-gene","lucaone"],
        help="lucaone llm version"
    )
    parser.add_argument(
        "--seq_llm_step",
        type=str,
        default="36800000",
        help="lucaone llm step."
    )
    parser.add_argument(
        "--seq_vector_dirpath",
        type=str,
        default=None,
        help="seq vector dirpath"
    )
    parser.add_argument(
        "--seq_matrix_dirpath",
        type=str,
        default=None,
        help="seq matrix dirpath"
    )
    parser.add_argument(
        "--seq_max_length",
        type=int,
        default=10242,
        help="matrix dirpath"
    )
    parser.add_argument(
        "--not_seq_prepend_bos",
        action="store_true",
        help="not seq_prepend_bos"
    )
    parser.add_argument(
        "--not_seq_append_eos",
        action="store_true",
        help="not seq_append_eos"
    )
    parser.add_argument(
        "--seq_embedding_vector_type",
        type=str,
        default="mean",
        help="seq embedding vector type, default: mean"
    )
    parser.add_argument(
        "--seq_vector_embedding_exists",
        action="store_true",
        help="whether the seq vector embedding exists"
    )
    parser.add_argument(
        "--seq_matrix_embedding_exists",
        action="store_true",
        help="whether the seq matrix embedding exists"
    )
    parser.add_argument(
        "--seq_matrix_add_special_token",
        action="store_true",
        help="add special token([CLS], [SEP]) embedding vector into embedding matrix of sequence"
    )
    parser.add_argument(
        "--seq_embedding_complete",
        action="store_true",
        help="when the seq len > inference_max_len, then the embedding matrix is completed by segment"
    )
    parser.add_argument(
        "--seq_embedding_complete_seg_overlap",
        action="store_true",
        help="overlap segment(overlap sliding window)"
    )
    parser.add_argument(
        "--seq_embedding_fixed_len_a_time",
        type=int,
        default=10240,
        help="When the input sequence is too long for your GPU to complete the inference at once, you can specify the fixed length of the inference at once, default: 10250"
    )
    parser.add_argument(
        "--seq_meta_fasta",
        type=str,
        default=None,
        help="seq_meta_fasta"
    )

    # cell llm
    parser.add_argument(
        "--cell_llm_dirpath",
        default=None,
        type=str,
        help="llm dir."
    )
    parser.add_argument(
        "--cell_llm_type",
        type=str,
        default="lucacell",
        choices=["lucacell","lucacellv2"],
        help="lucacell llm version"
    )
    parser.add_argument(
        "--cell_llm_version",
        type=str,
        default="lucacell-2400/v1.0",
        choices=["lucacell-2400/v1.0","lucacell-2048pos/v1.0"],
        help="lucacell llm version"
    )
    parser.add_argument(
        "--cell_llm_step",
        type=str,
        default="1300000",
        help="lucacell llm step."
    )
    parser.add_argument(
        "--cell_vector_dirpath",
        type=str,
        default=None,
        help="cell vector dirpath"
    )
    parser.add_argument(
        "--cell_matrix_dirpath",
        type=str,
        default=None,
        help="cell matrix dirpath"
    )
    parser.add_argument(
        "--cell_max_length",
        type=int,
        default=4098,
        help="cell matrix dirpath"
    )
    parser.add_argument(
        "--not_cell_prepend_bos",
        action="store_true",
        help="not cell_prepend_bos"
    )
    parser.add_argument(
        "--not_cell_append_eos",
        action="store_true",
        help="not cell_append_eos"
    )
    parser.add_argument(
        "--cell_embedding_vector_type",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="cell embedding vector type, default: mean"
    )
    parser.add_argument(
        "--cell_matrix_embedding_exists",
        action="store_true",
        help="whether the cell embedding exists"
    )
    parser.add_argument(
        "--cell_matrix_add_special_token",
        action="store_true",
        help="add special token([CLS], [SEP]) embedding vector into embedding matrix of cell"
    )
    parser.add_argument(
        "--cell_embedding_complete",
        action="store_true",
        help="when the cell len > inference_max_len, then the embedding matrix is completed by segment"
    )
    parser.add_argument(
        "--cell_embedding_complete_seg_overlap",
        action="store_true",
        help="overlap segment(overlap sliding window)"
    )
    parser.add_argument(
        "--cell_embedding_fixed_len_a_time",
        type=int,
        default=4096,
        help="When the input cell is too long for your GPU to complete the inference at once, you can specify the fixed length of the inference at once, default: 4096"
    )

    # others
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help="ignore index"
    )
    parser.add_argument(
        "--non_ignore",
        action="store_true",
        help="none ignore."
    )
    parser.add_argument(
        "--vector_fc_size",
        default=None,
        type=str,
        help="vector fc size."
    )
    parser.add_argument(
        "--matrix_fc_size",
        default=None,
        type=str,
        help="matrix fc size."
    )
    parser.add_argument(
        "--emb_activate_func",
        default="gelu",
        type=str,
        help="emb activate func."
    )
    parser.add_argument(
        "--fc_activate_func",
        default="gelu",
        type=str,
        help="fc activate func."
    )
    parser.add_argument(
        "--classifier_activate_func",
        default="tanh",
        type=str,
        help="classifier activate func."
    )
    parser.add_argument(
        "--loss_reduction",
        default="mean",
        choices=["none", "meansum", "mean", "meanmean"],
        type=str,
        help="loss reduction"
    )
    parser.add_argument(
        "--cross_atten",
        action="store_true",
        help="use cross attention"
    )
    parser.add_argument(
        "--self_atten",
        action="store_true",
        help="use self attention"
    )

    # for pair
    parser.add_argument(
        "--cell_max_length_a",
        default=None,
        type=int,
        help="the length of input mbedding a more than max length will be truncated, shorter will be padded."
    )
    parser.add_argument(
        "--embedding_input_size_a",
        default=None,
        type=int,
        help="a embedding_input_size"
    )
    parser.add_argument(
        "--cell_max_length_b",
        default=None,
        type=int,
        help="the length of input b embedding more than max length will be truncated, shorter will be padded."
    )
    parser.add_argument(
        "--embedding_input_size_b",
        default=None,
        type=int,
        help="b embedding_input_size"
    )
    parser.add_argument(
        "--not_matrix_encoder_shared",
        action="store_true",
        help="shared the seq encoder for two kind of matrix types"
    )

    parser.add_argument(
        '--worker_num',
        default=0,
        type=int,
        help='worker number for the data loader.'
    )

    parser.add_argument(
        "--not_save_emb_to_disk",
        action="store_true",
        help="if this option is flagged on, then the embeddings will not be saved to disk."
    )
    parser.add_argument(
        "--evaluate_strategy",
        default="epoch",
        choices=["epoch", "step"],
        help="whether to evaluate by epoch or step."
    )
    parser.add_argument(
        "--evaluate_steps",
        type=int,
        default=-1,
        help="evaluate steps."
    )
    parser.add_argument(
        "--fp16_embedding",
        action="store_true",
        help="whether to use fp16 for embedding."
    )

    # 早停
    parser.add_argument(
        '--early_stop_begin_epoch',
        default=None,
        type=int,
        help='early stop begin epoch.'
    )
    parser.add_argument(
        '--early_stop_num_epoch',
        default=None,
        type=int,
        help='early stop num epoch.'
    )
    parser.add_argument(
        '--num_labels',
        default=None,
        required=True,
        type=int,
        help='num labels.'
    )

    # finetune
    parser.add_argument(
        "--finetune_layers",
        default="20",
        type=str,
        help='finetune layer indices. e.g. 20, or 18,19,20'
    )

    # frozen
    parser.add_argument(
        "--not_frozen_gene_express_bin_embedding",
        action="store_true",
        help="Whether to frozen the gene express bin embedding ."
    )
    parser.add_argument(
        "--gene_express_vocab_size",
        default=None,
        type=int,
        help="gene_express_vocab_size ."
    )

    parser.add_argument(
        "--init_weight_filepath_for_gene_express_bin",
        default=None,
        type=str,
        help='the init weight filepath for gene express bins embedding'
    )

    args = parser.parse_args()
    return args


def check_args(args):
    # 如果输入信息中有序列，则需要token embedding
    if "gene_seq" in args.input_type:
        args.seq_no_token_embeddings = False
    else:
        args.seq_no_token_embeddings = True
    if "gene_id" in args.input_type:
        args.cell_no_token_embeddings = False
    else:
        args.cell_no_token_embeddings = True
    # 如果是token级别的任务，那么输入首尾不需要加上两个特俗token
    if args.task_level_type in ["gene_level"]:
        args.not_cell_prepend_bos = True
        args.not_cell_append_eos = True

    # 多标签分类，则label中-100需要设置不使用
    if args.task_type == "multi_label":
        args.non_ignore = True
    # 2分类/多标签分类，需要sigmoid
    if args.task_type in ["multi_label", "binary_class"]:
        args.sigmoid = True
    elif args.task_type in ["multi_class"]:
        args.sigmoid = False
    if not hasattr(args, "time_str") or args.time_str is None:
        now = datetime.now()
        args.time_str = now.strftime('%Y%m%d%H%M%S')

    # for pytorch 1.9+
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.local_rank = local_rank
        print("args.local_rank: %d" % args.local_rank)
    return args


def get_input_cols(args):
    if args.input_mode == "single" and args.input_type == "vector":
        input_col_names = [args.dataset_type, "cell_embedding_vector"]
    elif args.input_mode == "single" and args.input_type == "matrix":
        input_col_names = [args.dataset_type, "cell_embedding_matrix"]
    elif args.input_mode == "pair" and args.input_type == "vector_vs_vector":
        input_col_names = [args.dataset_type, "cell_embedding_vector", "cell_embedding_vector"]
    elif args.input_mode == "pair" and args.input_type == "gene_vs_matrix":
        input_col_names = [args.dataset_type, "gene_embedding_vectors", "cell_embedding_matrix"]
    elif args.input_mode == "pair" and args.input_type == "matrix_vs_gene":
        input_col_names = [args.dataset_type, "cell_embedding_matrix", "gene_embedding_vectors"]
    elif args.input_mode == "pair" and args.input_type == "vector_vs_matrix":
        input_col_names = [args.dataset_type, "cell_embedding_vector", "cell_embedding_matrix"]
    elif args.input_mode == "pair" and args.input_type == "matrix_vs_vector":
        input_col_names = [args.dataset_type, "cell_embedding_matrix", "cell_embedding_vector"]
    elif args.input_mode == "pair" and args.input_type == "matrix_vs_matrix":
        input_col_names = [args.dataset_type, "cell_embedding_matrix", "cell_embedding_matrix"]
    else:
        raise Exception("Not support input_mode=%s" % args.input_mode)
    return input_col_names


def get_label_size(label_filepath):
    '''
    load label size
    :param label_filepath: label list
    :return:
    '''
    if label_filepath:
        cur_labels = get_labels(label_filepath, header=True if label_filepath.endswith(".csv") else False)
        return len(cur_labels)
    else:
        raise Exception("Label path: %s not exists." % label_filepath)


def create_logger(args):
    '''
    create logger
    :param args:
    :return:
    '''
    if args.local_rank in [-1, 0]:
        print("args.local_rank: %d" % args.local_rank)
        # create the output dir
        if os.path.exists(args.output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        else:
            os.makedirs(args.output_dir)
        # create the logs dir
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
        # create tensorboard logs dir
        if not os.path.exists(args.tb_log_dir):
            os.makedirs(args.tb_log_dir)
        if args.seq_vector_dirpath:
            if "#" in args.seq_vector_dirpath:
                dirpaths = args.seq_vector_dirpath.split("#")
            else:
                dirpaths = [args.seq_vector_dirpath]
            for dirpath in dirpaths:
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
        if args.seq_matrix_dirpath:
            if "#" in args.seq_matrix_dirpath:
                dirpaths = args.seq_matrix_dirpath.split("#")
            else:
                dirpaths = [args.seq_matrix_dirpath]
            for dirpath in dirpaths:
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
        if args.cell_vector_dirpath:
            if "#" in args.cell_vector_dirpath:
                dirpaths = args.cell_vector_dirpath.split("#")
            else:
                dirpaths = [args.cell_vector_dirpath]
            for dirpath in dirpaths:
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
        if args.cell_matrix_dirpath:
            if "#" in args.cell_matrix_dirpath:
                dirpaths = args.cell_matrix_dirpath.split("#")
            else:
                dirpaths = [args.cell_matrix_dirpath]
            for dirpath in dirpaths:
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
    else:
        log_fp = None
    return log_fp


def get_model(args):
    '''
    create tokenizer, model config, model
    :param args:
    :return:
    '''
    if args.task_type == "regression":
        num_labels = args.num_labels
        label_list = ["0"] * num_labels
        print("num_labels: %d" % num_labels)
    else:
        label_list = get_labels(args.label_filepath, True if args.label_filepath.endswith(".csv") else False)
        num_labels = len(label_list)
        assert num_labels == args.num_labels
        print("num_labels: %d" % num_labels)
    if args.local_rank in [-1, 0]:
        logger.info("#" * 25 + "Labels Num:" + "#" * 25)
        logger.info("Num Labels: %d" % num_labels)
        save_labels(os.path.join(args.log_dir, "label.txt"), label_list)

    args.label_size = num_labels
    model_config = LucaConfig.from_json_file(args.config_path)
    if args.position_embedding_type == "RoPE":
        model_config.use_rotary_position_embeddings = True
    model_config.use_rotary_position_embeddings_for_cross = args.use_rotary_position_embeddings_for_cross
    if args.gene_express_vocab_size:
        model_config.gene_express_vocab_size = args.gene_express_vocab_size
    assert args.seq_max_length is not None
    model_config.seq_max_length = args.seq_max_length
    if args.input_mode in ["pair"]:
        assert args.cell_max_length is not None or (args.cell_max_length_a is not None and args.cell_max_length_b is not None)
        model_config.cell_max_length = args.cell_max_length
        if args.cell_max_length_a is not None and args.cell_max_length_b is not None:
            model_config.cell_max_length_a = args.cell_max_length_a
            model_config.cell_max_length_b = args.cell_max_length_b
        else:
            model_config.cell_max_length_a = args.cell_max_length
            model_config.cell_max_length_b = args.cell_max_length
        model_config.cell_max_length = max(model_config.cell_max_length_a, model_config.cell_max_length_b)
        args.cell_max_length = model_config.cell_max_length

        assert args.embedding_input_size is not None or (args.embedding_input_size_a is not None and args.embedding_input_size_b is not None)
        model_config.embedding_input_size = args.embedding_input_size
        if args.embedding_input_size_a is not None and args.embedding_input_size_b is not None:
            model_config.embedding_input_size_a = args.embedding_input_size_a
            model_config.embedding_input_size_b = args.embedding_input_size_b
        else:
            model_config.embedding_input_size_a = args.embedding_input_size
            model_config.embedding_input_size_b = args.embedding_input_size
            args.embedding_input_size_a = args.embedding_input_size
            args.embedding_input_size_b = args.embedding_input_size
    else:
        assert args.cell_max_length is not None
        model_config.cell_max_length = args.cell_max_length
        assert args.embedding_input_size is not None
        model_config.embedding_input_size = args.embedding_input_size

    if args.intermediate_size is not None:
        model_config.intermediate_size = args.intermediate_size
    elif args.matrix_encoder:
        model_config.intermediate_size = 4 * args.embedding_input_size
    elif args.hidden_size:
        model_config.intermediate_size = 4 * args.hidden_size
    else:
        model_config.intermediate_size = 4 * model_config.hidden_size

    model_config.num_labels = num_labels
    model_config.matrix_pooling_type = args.matrix_pooling_type
    model_config.use_embed_layer_norm = args.use_embed_layer_norm

    if args.hidden_size:
        model_config.hidden_size = args.hidden_size
    if args.num_attention_heads:
        model_config.num_attention_heads = args.num_attention_heads
    if args.num_hidden_layers:
        model_config.num_hidden_layers = args.num_hidden_layers
    if args.dropout_prob is not None and args.dropout_prob > -1:
        model_config.attention_probs_dropout_prob = args.dropout_prob
        model_config.classifier_dropout_prob = args.dropout_prob
        model_config.hidden_dropout_prob = args.dropout_prob
    if hasattr(args, "layer_norm_type") and args.layer_norm_type:
        model_config.layer_norm_type = args.layer_norm_type
    if hasattr(args, "layer_norm_name") and args.layer_norm_name:
        model_config.layer_norm_name = args.layer_norm_name
    if hasattr(args, "layer_norm_eps") and args.layer_norm_eps:
        model_config.layer_norm_eps = args.layer_norm_eps

    model_config.ignore_index = args.ignore_index
    model_config.seq_no_token_embeddings = args.seq_no_token_embeddings
    model_config.seq_no_token_type_embeddings = args.seq_no_token_type_embeddings
    model_config.seq_no_position_embeddings = args.seq_no_position_embeddings
    if args.seq_position_embedding_type is not None:
        model_config.seq_position_embedding_type = args.seq_position_embedding_type
    model_config.cell_no_token_embeddings = args.cell_no_token_embeddings
    model_config.cell_no_token_type_embeddings = args.cell_no_token_type_embeddings
    model_config.cell_no_position_embeddings = args.cell_no_position_embeddings
    if args.cell_position_embedding_type is not None:
        model_config.cell_position_embedding_type = args.cell_position_embedding_type
    if args.matrix_fc_size and args.matrix_fc_size != "null":
        model_config.matrix_fc_size = [int(v) for v in args.matrix_fc_size.split(",")]
    if args.vector_fc_size and args.vector_fc_size != "null":
        model_config.vector_fc_size = [int(v) for v in args.vector_fc_size.split(",")]
    if args.emb_activate_func and args.emb_activate_func != "null":
        model_config.emb_activate_func = args.emb_activate_func
    if args.fc_activate_func and args.fc_activate_func != "null":
        model_config.fc_activate_func = args.fc_activate_func
    if args.classifier_activate_func and args.classifier_activate_func != "null":
        model_config.classifier_activate_func = args.classifier_activate_func
    #
    args.seq_prepend_bos = True
    if args.not_seq_prepend_bos:
        args.seq_prepend_bos = False
    args.seq_append_eos = True
    if args.not_seq_append_eos:
        args.seq_append_eos = False
    args.cell_prepend_bos = True
    if args.not_cell_prepend_bos:
        args.cell_prepend_bos = False
    args.cell_append_eos = True
    if args.not_cell_append_eos:
        args.cell_append_eos = False
    model_config.self_atten = args.self_atten
    model_config.cross_atten = args.cross_atten

    if args.classifier_size:
        model_config.classifier_size = args.classifier_size
    if args.pos_weight:
        pos_weight = args.pos_weight.split(",")
        if len(pos_weight) == 1:
            pos_weight = float(pos_weight[0])
        else:
            pos_weight = [float(v) for v in pos_weight]
        args.pos_weight = pos_weight
        model_config.pos_weight = pos_weight
    if args.weight:
        model_config.weight = [float(v) for v in args.weight.split(",")]
        args.weight = model_config.weight
    if args.loss_reduction:
        if args.loss_reduction in ["meanmean", "meansum"] \
                and args.task_level_type in ["cell_level"] \
                and args.task_type not in ["multi_label", "multi-label"]:
            args.loss_reduction = "mean"
        model_config.loss_reduction = args.loss_reduction

    if args.seq_llm_type == "lucaone":
        seq_tokenizer_class = AlphabetLucaOne
    else:
        raise Exception("Not support the seq_llm_type=%s" % args.seq_llm_type)
    seq_tokenizer = seq_tokenizer_class.from_predefined("gene_prot")
    if args.not_seq_prepend_bos:
        seq_tokenizer.prepend_bos = False
    if args.not_seq_append_eos:
        seq_tokenizer.append_eos = False
    model_config.cls_token_id = seq_tokenizer.cls_idx
    model_config.sep_token_id = seq_tokenizer.eos_idx
    model_config.pad_token_id = seq_tokenizer.pad_token_id

    cell_tokenizer = AlphabetLucaCell.from_predefined(
        "nucleotide",
        gene_id_list_filepath=None,
        express_bin_list_filepath=None,
        express_sorted_list_filepath=None,
    )
    if args.not_cell_prepend_bos:
        cell_tokenizer.nucleotide_prepend_bos = False
        cell_tokenizer.gene_prepend_bos = False
        cell_tokenizer.express_prepend_bos = False
    if args.not_cell_append_eos:
        cell_tokenizer.nucleotide_append_eos = False
        cell_tokenizer.gene_append_eos = False
        cell_tokenizer.express_append_eos = False
    # model class
    if args.model_type in ["lucacell_finetune"]:
        model_class = LucaCellForDownstreamTasks
    elif args.model_type in ["lucasingle"]:
        model_class = LucaSingle
    elif args.model_type in ["lucapair_homo"]:
        model_class = LucaPairHomo
    elif args.model_type in ["lucapair_heter"]:
        model_class = LucaPairHeter
    elif args.model_type in ["lucapair_intrainter"]:
        model_class = LucaPairIntraInter
    else:
        raise Exception("Not support the model_type=%s" % args.model_type)

    if args.model_type in ["lucacell_finetune"]:
        lucacell_args_info, lucacell_model_config, lucacell_model, lucacell_cell_tokenizer = load_lucacell(
            log_filepath=os.path.join(os.path.dirname(args.model_dirpath.replace("/models/", "/logs/")), "logs.txt"),
            model_dirpath=args.model_dirpath,
            embedding_inference=False
        )
        model = model_class(lucacell_model_config, lucacell_args_info, model_config, args)
        model_state_dict_keys = {}
        for item in model.state_dict().items():
            model_state_dict_keys[item[0]] = item[1]
        new_state_dict = OrderedDict()
        for k, v in lucacell_model.state_dict().items():
            if k.startswith("module."):
                # remove `module.`
                name = k[7:]
            else:
                name = k
            if name in model_state_dict_keys:
                new_state_dict[name] = v
        diff = set(model_state_dict_keys.keys()).difference(set(new_state_dict.keys()))
        if diff:
            print("diff:")
            print(diff)
            for key in diff:
                new_state_dict[key] = model_state_dict_keys[key]
        model.load_state_dict(new_state_dict)
        finetune_layers = args.finetune_layers.split(",")
        finetune_layers = [int(v) - 1 for v in finetune_layers]
        for name, param in model.named_parameters():
            if name in diff:
                param.requires_grad = True
                print("%s: True" % name)
            else:
                if name.startswith("layers."):
                    layer_id = int(name.split(".")[1])
                    if layer_id in finetune_layers:
                        param.requires_grad = True
                        print("%s: True" % name)
                    '''
                    else:
                        param.requires_grad = False
                        print("%s: False" % name)
                    '''
                '''
                else:
                    param.requires_grad = False
                    print("%s: False" % name)
                '''
    else:
        if args.model_type in ["lucapair_intrainter"]:
            model_config.self_encoder_layers = args.num_hidden_layers
            model_config.self_attention_heads = args.num_attention_heads
            model_config.cross_encoder_layers = args.num_hidden_layers
            model_config.cross_attention_heads = args.num_attention_heads
            model_config.encoder_ffn_dim = args.intermediate_size
            model_config.embedding_input_size_a = args.embedding_input_size_a if args.embedding_input_size_a else args.embedding_input_size
            model_config.embedding_input_size_b = args.embedding_input_size_b if args.embedding_input_size_b else args.embedding_input_size
            model_config.hidden_size = args.hidden_size if args.hidden_size else (model_config.embedding_input_size_a if model_config.embedding_input_size_a == model_config.embedding_input_size_b else 1024)
            model_config.dropout = 0.0
            model_config.classifier_dropout = args.dropout_prob
            args.matrix_add_special_token = True
            if hasattr(args, "layer_dropout") and args.layer_dropout > -1:
                model_config.layer_dropout = args.layer_dropout
        if args.model_dirpath and os.path.exists(args.model_dirpath):
            model = load_trained_model(model_config, args, model_class, args.model_dirpath)
        else:
            model = model_class(model_config, args)

    return model_config, model_class, model, seq_tokenizer, cell_tokenizer, label_list


def create_device(args):
    '''
    create device
    :param args:
    :return:
    '''
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.n_gpu = torch.cuda.device_count()
        if args.n_gpu > 1:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            dist.init_process_group(backend="nccl", timeout=timedelta(seconds=54000))
            if args.local_rank == 0:
                print('world size: %d' % dist.get_world_size())
        else:
            device = torch.device("cuda")
    print("create_device:", device)
    return device


def main():
    # get args
    args = get_args()

    # check args
    args = check_args(args)

    # create log dir
    log_fp = create_logger(args)

    # the output type
    args.output_mode = args.task_type
    # For binary_class/multi_label tasks, the sigmoid needs to be added to the last layer
    if args.output_mode in [
        "multi_label", "multi-label",
        "binary_class", "binary-class"
    ]:
        args.sigmoid = True
    elif args.output_mode in [
        "multi_class", "multi-class"
    ]:
        args.sigmoid = False

    # device
    args.device = create_device(args)

    # lucaone for gene seqs embedding
    download_trained_checkpoint_lucaone_v2(
        llm_dir=args.seq_llm_dirpath,
        llm_type=args.seq_llm_type,
        llm_version=args.seq_llm_version,
        llm_step=args.seq_llm_step
    )
    # lucacell for cell embedding
    '''
    download_trained_checkpoint_lucacell(
        llm_dir=args.cell_llm_dirpath,
        llm_type=args.cell_llm_type,
        llm_version=args.cell_llm_version,
        llm_step=args.cell_llm_step
    )
    '''
    # create model
    model_config, model_class, model, seq_tokenizer, cell_tokenizer, label_list = get_model(args)
    seq_llm_dirpath = "%s/models/%s/%s/checkpoint-step%s" % (
        args.seq_llm_dirpath if args.seq_llm_dirpath else "..",
        args.seq_llm_type,
        args.seq_llm_version,
        args.seq_llm_step
    )
    args.seq_llm_dirpath = seq_llm_dirpath

    cell_llm_dirpath = "%s/models/%s/%s/checkpoint-step%s" % (
        args.cell_llm_dirpath if args.cell_llm_dirpath else "..",
        args.cell_llm_type,
        args.cell_llm_version,
        args.cell_llm_step
    )
    args.cell_llm_dirpath = cell_llm_dirpath
    # encoder config
    # encoder_config
    encoder_config = {
        "input_type": args.input_type,
        "trunc_type": args.trunc_type,
        "seq_llm_dirpath": seq_llm_dirpath,
        "seq_llm_type": args.seq_llm_type,
        "seq_llm_version": args.seq_llm_version,
        "seq_llm_step": args.seq_llm_step,
        "seq_tokenizer": seq_tokenizer,
        "seq_max_length": args.seq_max_length,
        "seq_prepend_bos": True,
        "seq_append_eos": True,
        "seq_matrix_add_special_token": args.seq_matrix_add_special_token,
        "seq_vector_dirpath": args.seq_vector_dirpath,
        "seq_matrix_dirpath": args.seq_matrix_dirpath,
        "seq_embedding_vector_type": args.seq_embedding_vector_type,
        "seq_embedding_complete": args.seq_embedding_complete,
        "seq_embedding_complete_seg_overlap": args.seq_embedding_complete_seg_overlap,
        "seq_embedding_fixed_len_a_time": args.seq_embedding_fixed_len_a_time,
        "seq_vector_embedding_exists":args.seq_vector_embedding_exists,
        "seq_matrix_embedding_exists": args.seq_matrix_embedding_exists,
        "seq_meta_fasta": args.seq_meta_fasta,
        "cell_llm_dirpath": cell_llm_dirpath,
        "cell_llm_type": args.cell_llm_type,
        "cell_llm_version": args.cell_llm_version,
        "cell_llm_step": args.cell_llm_step,
        "cell_tokenizer": cell_tokenizer,
        "cell_max_length": args.cell_max_length,
        "cell_prepend_bos": True,
        "cell_append_eos": True,
        "cell_matrix_add_special_token": args.cell_matrix_add_special_token,
        "cell_vector_dirpath": args.cell_vector_dirpath,
        "cell_matrix_dirpath": args.cell_matrix_dirpath,
        "cell_embedding_vector_type": args.cell_embedding_vector_type,
        "cell_embedding_complete": args.cell_embedding_complete,
        "cell_embedding_complete_seg_overlap": args.cell_embedding_complete_seg_overlap,
        "cell_embedding_fixed_len_a_time": args.cell_embedding_fixed_len_a_time,
        "cell_matrix_embedding_exists": args.cell_matrix_embedding_exists,
        "gpu_id": 0 if args.local_rank == "-1" or args.local_rank == -1 else args.local_rank,
        "lucacell_finetune": args.model_type == "lucacell_finetune",
        "not_frozen_gene_express_bin_embedding": args.not_frozen_gene_express_bin_embedding,
        "buffer_size": args.buffer_size,
        "seq_buffer_size": args.seq_buffer_size
    }
    # file row parser
    # 文件记录解析函数
    encoder = Encoder(**encoder_config)
    # pair对数据集
    if args.input_mode == "single":
        parse_row_func = encoder.encode_single_cell
    elif args.input_mode == "pair" and args.input_type in ["matrix_vs_matrix", "vector_vs_vector", "vector_vs_matrix", "matrix_vs_vector"]:
        parse_row_func = encoder.encode_pair_cell
    elif args.input_mode == "pair" and args.input_type in ["gene_vs_matrix", "matrix_vs_gene"]:
        parse_row_func = encoder.encode_pair_gene_cell
    else:
        raise Exception("Not support input_mode=%s, input_type=%s" % (args.input_mode, args.input_type))

    # encoding
    # luca独特的batch转换器
    batch_data_func = BatchConverter(
        task_level_type=args.task_level_type,
        label_size=args.label_size,
        output_mode=args.output_mode,
        seq_tokenizer=seq_tokenizer,
        seq_no_position_embeddings=args.seq_no_position_embeddings,
        seq_no_token_type_embeddings=args.seq_no_token_type_embeddings,
        seq_truncation_length=args.seq_max_length,
        seq_matrix_add_special_token=args.seq_matrix_add_special_token,
        seq_prepend_bos=not args.not_seq_prepend_bos,
        seq_append_eos=not args.not_seq_append_eos,
        cell_tokenizer=cell_tokenizer,
        cell_no_position_embeddings=args.cell_no_position_embeddings,
        cell_no_token_type_embeddings=args.cell_no_token_type_embeddings,
        cell_truncation_length=args.cell_max_length,
        cell_matrix_add_special_token=args.cell_matrix_add_special_token,
        cell_prepend_bos=not args.not_cell_prepend_bos,
        cell_append_eos=not args.not_cell_append_eos,
        ignore_index=args.ignore_index,
        padding_idx=seq_tokenizer.padding_idx,
        unk_idx=seq_tokenizer.unk_idx,
        cls_idx=seq_tokenizer.cls_idx,
        eos_idx=seq_tokenizer.eos_idx,
        mask_idx=seq_tokenizer.mask_idx,
        non_ignore=args.non_ignore,
        input_type=args.input_type,
        trunc_type=args.trunc_type,
        lucacell_finetune=args.model_type == "lucacell_finetune",
        not_frozen_gene_express_bin_embedding=args.not_frozen_gene_express_bin_embedding
    )
    if args.local_rank in [0, -1]:
        print("n_gpu: %d" % args.n_gpu)

        args_dict = {}
        for attr, value in sorted(args.__dict__.items()):
            if attr != "device":
                args_dict[attr] = value
            else:
                args_dict[attr] = str(value)
        log_fp.write(json.dumps(args_dict, ensure_ascii=False, indent=4) + "\n")
        '''
        args_json = json.dumps(vars(args), indent=4, ensure_ascii=False)
        log_fp.write(args_json + "\n")
        '''
        log_fp.write("#" * 50 + "\n")
        log_fp.write("n_gpu: %d\n" % args.n_gpu)
        log_fp.write("#" * 50 + "\n")
        # input types
        input_col_names = get_input_cols(args)
        log_fp.write("Inputs:\n")
        log_fp.write("Input Name List: %s\n" % ",".join(input_col_names))
        log_fp.write("#" * 50 + "\n")

        # output model hyperparameters in logger
        if len(model_config.id2label) > 10:
            str_config = copy.deepcopy(model_config)
            str_config.id2label = {}
            str_config.label2id = {}
        else:
            str_config = copy.deepcopy(model_config)
        log_fp.write("Encoder Config:\n %s\n" % str(encoder_config))
        log_fp.write("#" * 50 + "\n")
        log_fp.write("Model Config:\n %s\n" % str(str_config))
        log_fp.write("#" * 50 + "\n")
        log_fp.write("Mode Architecture:\n %s\n" % str(model))
        log_fp.write("#" * 50 + "\n")
        log_fp.write("Model parameters: %d \n" % sum(p.numel() for p in model.parameters()))
        log_fp.write("#" * 50 + "\n")

        # model size
        model_size_info = get_parameter_number(model)
        log_fp.write(json.dumps(model_size_info, ensure_ascii=False) + "\n")
        log_fp.write("#" * 50 + "\n")
        log_fp.flush()

    # Set seed
    set_seed(args)

    # model to device
    model.to(args.device)

    if args.local_rank not in [-1, 0]:
        dist.barrier()

    if args.local_rank == 0:
        dist.barrier()

    if args.n_gpu <= 1:
        print("n_gpu: %d, use: MultiFilesStreamLoader" % args.n_gpu)
        train_dataloader = MultiFilesStreamLoader(
            args.train_data_dir,
            args.per_gpu_train_batch_size,
            args.buffer_size,
            parse_row_func=parse_row_func,
            batch_data_func=batch_data_func,
            task_level_type=args.task_level_type,
            input_mode=args.input_mode,
            input_type=args.input_type,
            output_mode=args.output_mode,
            label_size=args.label_size,
            dataset_type="train",
            vector_dirpath=args.cell_vector_dirpath,
            matrix_dirpath=args.cell_matrix_dirpath,
            inference=False,
            header=True,
            shuffle=True
        )
    else:
        print("n_gpu: %d, use: DataLoader" % args.n_gpu)
        train_dataset = load_dataset(
            'csv',
            data_dir=args.train_data_dir,
            split='train',
            streaming=True
        )
        if args.input_mode == "pair":
            print("Has Pair: True")
            train_dataset = train_dataset.map(
                lambda x: parse_row_func(
                    x["sample_id_a"],
                    x["sample_id_b"],
                    x["sample_type_a"],
                    x["sample_type_b"],
                    x["sample_gene_id_list_a"],
                    x["sample_gene_id_list_b"],
                    x["sample_gene_seq_type_list_a"],
                    x["sample_gene_seq_type_list_b"],
                    x["sample_gene_seq_list_a"],
                    x["sample_gene_seq_list_b"],
                    x["sample_gene_express_bin_list_a"],
                    x["sample_gene_express_bin_list_b"],
                    x["label"]
                ),
                batched=False
            )
        else:
            print("Has Pair: False")
            train_dataset = train_dataset.map(
                lambda x: parse_row_func(
                    x["sample_id"],
                    x["sample_type"],
                    x["sample_gene_id_list"],
                    x["sample_gene_seq_type_list"],
                    x["sample_gene_seq_list"],
                    x["sample_gene_express_bin_list"],
                    x["label"]
                ),
                batched=False
            )
        train_dataset = split_dataset_by_node(train_dataset, rank=args.local_rank, world_size=dist.get_world_size()) \
            .shuffle(buffer_size=args.buffer_size, seed=args.seed)
        train_dataset = train_dataset.with_format("torch")
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.per_gpu_train_batch_size,
            num_workers=args.worker_num,
            pin_memory=True,
            collate_fn=batch_data_func
        )

    # Training
    max_metric_model_info = None
    if args.do_train:
        logger.info("++++++++++++Training+++++++++++++")
        global_step, tr_loss, max_metric_model_info = train(
            args, train_dataloader, model_config, model, seq_tokenizer, parse_row_func, batch_data_func,
            train_sampler=None, log_fp=log_fp
        )
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # save
    if args.do_train and args.local_rank in [-1, 0]:
        logger.info("++++++++++++Save Model+++++++++++++")
        # Create output directory if needed
        best_output_dir = os.path.join(args.output_dir, "best")
        global_step = max_metric_model_info["global_step"]
        prefix = "checkpoint-{}".format(global_step)
        shutil.copytree(os.path.join(args.output_dir, prefix), best_output_dir)
        logger.info("Saving model checkpoint to %s", best_output_dir)
        torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
        save_labels(os.path.join(best_output_dir, "label.txt"), label_list)

    # evaluate
    if args.do_eval and args.local_rank in [-1, 0] and args.val_data_dir:
        logger.info("++++++++++++Validation+++++++++++++")
        log_fp.write("++++++++++++Validation+++++++++++++\n")
        global_step = max_metric_model_info["global_step"]
        logger.info("best %s global step: %d" % (args.best_metric_type, global_step))
        log_fp.write("best %s global step: %d\n" % (args.best_metric_type, global_step))
        prefix = "checkpoint-{}".format(global_step)
        checkpoint = os.path.join(args.output_dir, prefix)
        if seq_tokenizer is None:
            seq_tokenizer = AlphabetLucaOne.from_pretrained(checkpoint)
        if cell_tokenizer is None:
            cell_tokenizer = AlphabetLucaCell.from_pretrained(checkpoint)

        logger.info("checkpoint path: %s" % checkpoint)
        log_fp.write("checkpoint path: %s\n" % checkpoint)
        model = load_trained_model(model_config, args, model_class, checkpoint)
        model.to(args.device)
        result = evaluate(args, model, parse_row_func, batch_data_func, prefix=prefix, log_fp=log_fp)
        result = dict(("evaluation_" + k + "_{}".format(global_step), v) for k, v in result.items())
        logger.info(json.dumps(result, ensure_ascii=False))
        log_fp.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Testing
    if args.do_predict and args.local_rank in [-1, 0] and args.test_data_dir:
        logger.info("++++++++++++Testing+++++++++++++")
        log_fp.write("++++++++++++Testing+++++++++++++\n")
        global_step = max_metric_model_info["global_step"]
        logger.info("best %s global step: %d" % (args.best_metric_type, global_step))
        log_fp.write("best %s global step: %d\n" % (args.best_metric_type, global_step))
        prefix = "checkpoint-{}".format(global_step)
        checkpoint = os.path.join(args.output_dir, prefix)
        if seq_tokenizer is None:
            seq_tokenizer = AlphabetLucaOne.from_pretrained(checkpoint)
        if cell_tokenizer is None:
            cell_tokenizer = AlphabetLucaCell.from_pretrained(checkpoint)
        logger.info("checkpoint path: %s" % checkpoint)
        log_fp.write("checkpoint path: %s\n" % checkpoint)
        model = load_trained_model(model_config, args, model_class, checkpoint)
        model.to(args.device)
        result = test(args, model, parse_row_func, batch_data_func, prefix=prefix, log_fp=log_fp)
        result = dict(("evaluation_" + k + "_{}".format(global_step), v) for k, v in result.items())
        logger.info(json.dumps(result, ensure_ascii=False))
        log_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
    if args.local_rank in [-1, 0] and log_fp:
        log_fp.close()
    if args.n_gpu > 1:
        dist.barrier()


if __name__ == "__main__":
    main()


