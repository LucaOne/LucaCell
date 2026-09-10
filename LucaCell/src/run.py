#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/26 14:48
@project: LucaOne
@file: run.py
@desc: main for LucaOne
'''
import os, torch
import sys, json
import datetime
import argparse
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from datasets.distributed import split_dataset_by_node
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from encoder import Encoder
    from utils import set_seed, to_device, load_trained_model, get_labels, get_parameter_number, save_model_parameters
    from multi_files_stream_dataloader import MultiFilesStreamLoader
    from trainer import train, train_continue, train_simple_v2
    from models.lucacell import LucaCell
    from models.alphabet import Alphabet
    from models.configuration_lucacell import LucaCellConfig
    from batch_converter import BatchConverter
except ImportError as e:
    from src.encoder import Encoder
    from src.utils import set_seed, to_device, load_trained_model, get_labels, get_parameter_number, save_model_parameters
    from src.multi_files_stream_dataloader import MultiFilesStreamLoader
    from src.trainer import train, train_continue, train_simple_v2
    from src.models.lucacell import LucaCell
    from src.models.alphabet import Alphabet
    from src.models.configuration_lucacell import LucaCellConfig
    from src.batch_converter import BatchConverter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'


def get_args():
    parser = argparse.ArgumentParser(description='LucaCell')
    # for logging
    parser.add_argument(
        "--tb_log_dir",
        type=str,
        default=None,
        required=True,
        help="TensorBoard log every X updates steps."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        required=True,
        help="Log every X updates steps."
    )

    parser.add_argument(
        "--global_nucleotide_seqs_filepath",
        type=str,
        default=None,
        required=True,
        help="global nucleotide seqs filepath"
    )

    parser.add_argument(
        "--global_nucleotide_seq_embeddings_dirpath",
        type=str,
        default=None,
        required=True,
        help="global nucleotide seq embedding dirpath"
    )
    parser.add_argument(
        "--global_gene_positions_filepath",
        type=str,
        default=None,
        required=True,
        help="global gene positions dirpath"
    )

    # for model
    # the modeling time str
    parser.add_argument(
        "--time_str",
        type=str,
        default=None,
        help="the modeling time str"
    )
    parser.add_argument(
        "--alphabet_type",
        type=str,
        default="nucleotide",
        required=True,
        choices=["nucleotide", "gene"],
        help="the alphabet type"
    )
    parser.add_argument(
        "--gene_id_list_filepath",
        type=str,
        default=None,
        help="the gene id list filepath"
    )
    parser.add_argument(
        "--express_bin_list_filepath",
        type=str,
        default=None,
        required=True,
        help="the express value list filepath"
    )
    parser.add_argument(
        "--express_sorted_list_filepath",
        type=str,
        default=None,
        help="the express sorted list filepath"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=None,
        required=True,
        help="the embedding dim"
    )
    parser.add_argument(
        "--ffn_size",
        type=int,
        default=None,
        help="the ffn layer size"
    )
    parser.add_argument(
        "--nucleotide_token_encoder_num_attention_heads",
        type=int,
        default=None,
        required=True,
        help="num attention heads of nucleotide_token_encoder"
    )
    parser.add_argument(
        "--nucleotide_token_encoder_num_layers",
        type=int,
        default=None,
        required=True,
        help="num hidden layers of nucleotide_token_encoder"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        required=True,
        help="num hidden layers"
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int, 
        default=None, 
        required=True, 
        help="num attention heads"
    )
    # for input sequence
    parser.add_argument(
        "--max_nucleotide_seq_length",
        default=1280,
        type=int,
        help="the max length of the nucleotide sequence"
    )
    parser.add_argument(
        "--max_gene_seq_length",
        default=1280,
        type=int,
        help="the max length of the gene id/express sequence"
    )
    parser.add_argument(
        '--add_special_tokens',
        action='store_true',
        help='add special tokens in the start and end position of the input sequence and the gene id/express seq'
    )
    parser.add_argument(
        '--truncation',
        default='right',
        type=str,
        choices=["right", "left"],
        help='the truncation side type'
    )
    parser.add_argument(
        '--no_nucleotide_token_embeddings',
        action='store_true',
        help='whether no nucleotide token embeddings'
    )
    parser.add_argument(
        '--no_nucleotide_token_encoder',
        action='store_true',
        help='whether no nucleotide token encoder'
    )
    parser.add_argument(
        '--no_nucleotide_position_embeddings',
        action='store_true',
        help='whether no nucleotide position embedding'
    )
    parser.add_argument(
        '--no_gene_positions_embeddings',
        action='store_true',
        help='whether no gene position embedding'
    )
    parser.add_argument(
        '--no_express_sorted_embeddings',
        action='store_true',
        help='whether no gene express sorted embedding'
    )
    parser.add_argument(
        '--max_express_sorted_position_embeddings',
        default=100,
        type=int,
        help='the max express sorted size'
    )
    parser.add_argument(
        '--no_gene_type_embeddings',
        action='store_true',
        help='whether no gene type embedding'
    )
    parser.add_argument(
        '--use_rotary_embeddings',
        action='store_true',
        help='whether use rope'
    )
    parser.add_argument(
        '--use_embed_layer_norm',
        action='store_true',
        help='whether use embedding layer norm'
    )
    parser.add_argument(
        '--use_last_layer_norm',
        action='store_true',
        help='whether use last layer norm'
    )
    parser.add_argument(
        '--embed_scale',
        default=1.0,
        type=float,
        help="embed_scale"
    )
    parser.add_argument(
        '--dropout_prob',
        default=0.1,
        type=float,
        help="dropout_prob"
    )
    parser.add_argument(
        '--attention_probs_dropout_prob',
        default=0.1,
        type=float,
        help="attention_probs_dropout_prob"
    )

    # pooling_type
    parser.add_argument(
        "--nucleotide_token_pooling_type",
        type=str,
        default=None,
        choices=[
            "context_attention",
            "weighted_attention",
            "value_attention",
            "mean",
            "max",
            "cls",
            "sep",
            "cls_sep"
        ],
        help="pooling type for nucleotide token encoder"
    )

    parser.add_argument(
        "--nucleotide_input_type",
        type=str,
        default="embedding_matrix",
        choices=["seq", "embedding_vector", "embedding_matrix"],
        help="nucleotide input type"
    )

    # for model selection
    parser.add_argument(
        '--model_type', 
        default="lucacell", 
        type=str,
        choices=["lucacell"], 
        help='the model type'
    )
    parser.add_argument(
        '--model_config', 
        type=str, 
        default=None, 
        help='the model config file path'
    )

    # for dataset
    parser.add_argument(
        "--train_data_dir", 
        default=None, 
        type=str, 
        required=True, 
        help="the train dataset dir path."
    )
    parser.add_argument(
        "--val_data_dir",
        default=None, 
        type=str, 
        required=True, help="the validation dataset dir path."
    )
    parser.add_argument(
        "--test_data_dir",
        default=None,
        type=str,
        required=True,
        help="the testing dataset dir path."
    )

    # the loss info for the pretraining tasks
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help="the ignore index."
    )

    # for stream dataloader
    parser.add_argument(
        '--buffer_size',
        default=10240,
        type=int,
        help='buffer size for the dataset loading'
    )
    parser.add_argument(
        '--embedding_buffer_size',
        default=None,
        type=int,
        help='embedding buffer size for the seq embeddings'
    )

    parser.add_argument(
        '--worker_num',
        default=1,
        type=int,
        help='worker number for the data loader.'
    )

    # for training
    parser.add_argument(
        "--best_metric_type",
        type=str,
        default="f1",
        choices=["loss", "acc", "jaccard", "prec", "recall", "f1", "fmax", "roc_auc", "pr_auc"],
        help="which metric for model selected"
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
    parser.add_argument(
        "--seed",
        default=1111,
        type=int,
        help="random seed value."
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='whether not to use GPU'
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="for fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="the initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="weight decay if we apply some."
    )
    parser.add_argument(
        "--decay_rate",
        default=0.9,
        type=float,
        help="weight decay of learning rate."
    )
    parser.add_argument(
        "--lr_update_steps",
        default=30000,
        type=int,
        help="lr update steps."
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int, 
        help="total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1, 
        type=int, 
        help="set total number of training steps to perform."
    )
    parser.add_argument(
        "--warmup_steps",
        default=-1, 
        type=int,
        help="linear warmup over warmup_steps."
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
        "--do_train", 
        action="store_true",
        help="whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="whether to run predict on the test set."
    )
    parser.add_argument(
        "--do_metrics",
        action="store_true",
        help="whether to run eval metrics on the test set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="where to evaluate during training."
    )
    parser.add_argument(
        "--loss_logging_steps",
        type=int,
        default=100,
        help="Loss log every X updates steps."
    )
    parser.add_argument(
        "--logging_steps",
        type=int, default=10000,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
        help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation steps."
    )
    parser.add_argument(
        "--eval_start_epoch",
        type=int,
        default=-1,
        help="the start epoch to eval."
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="step",
        choices=["step", "epoch"],
        help="lr update scheduler type."
    )

    # for model save
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="the output dir path"
    )
    # for pretraining tasks
    parser.add_argument(
        "--pretrain_task_name",
        type=str,
        default="express_token_mask,express_sorted_mask",
        choices=["express_token_mask,express_sorted_mask", "express_token_mask", "express_sorted_mask"],
        help="the pretrain task name"
    )
    # for loss
    parser.add_argument(
        "--pretrain_task_weight",
        type=str,
        default="express_token_mask:1.0,express_sorted_mask:1.0",
        help="the pretrain task weight"
    )

    # pretrained model path
    parser.add_argument(
        "--model_dirpath",
        default=None,
        type=str,
        help="the pretrained model path"
    )

    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default=None,
        help="whether to use the pretrained model init parameters"
    )

    parser.add_argument(
        "--processed_sample_cnt", 
        default=100000,
        type=int,
        help="processed how many samples to write sample ids")
    parser.add_argument(
        "--trained_checkpoint",
        default=None,
        type=int,
        help="the checkpoint of continue to pretraining"
    )
    parser.add_argument(
        "--trained_epoch",
        default=None,
        type=int,
        help="the epoch of continue to pretraining"
    )
    parser.add_argument(
        "--removed_continue",
        action="store_true",
        help="whether to remove done samples to continue training"
    )
    parser.add_argument(
        "--global_total_loss",
        default=0,
        type=float,
        help="the global loss to continue training"
    )
    parser.add_argument(
        "--cur_epoch_loss",
        default=0,
        type=float,
        help="the epoch loss to continue training"
    )
    parser.add_argument(
        "--mlm_probability",
        default=0.15,
        type=float,
        help="the mask prob"
    )

    parser.add_argument(
        '--use_bf16',
        action='store_true',
        help='whether to use bf16 for training'
    )
    input_args = parser.parse_args()
    return input_args


def check_args(args):
    '''
    check the args
    :param args:
    :return:
    '''
    # for pytorch 1.9+
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.local_rank = local_rank
        print("args.local_rank: %d" % args.local_rank)
    assert args.log_dir is not None
    assert args.tb_log_dir is not None
    assert args.train_data_dir is not None and os.path.exists(args.train_data_dir)
    assert args.output_dir is not None
    assert args.model_config is not None and os.path.exists(args.model_config)
    if not hasattr(args, "time_str") or args.time_str is None:
        now = datetime.datetime.now()
        args.time_str = now.strftime('%Y%m%d%H%M%S')
    if args.pretrain_task_weight:
        strs = args.pretrain_task_weight.split(",")
        args.pretrain_task_weight = {}
        for s in strs:
            ss = s.split(":")
            args.pretrain_task_weight[ss[0]] = float(ss[1])


def get_model(args):
    '''
    create tokenizer, model config, model
    :param args:
    :return:
    '''
    # four type of models
    if args.model_type in ["lucacell"]:
        config_class, model_class = LucaCellConfig, LucaCell
    else:
        raise Exception("Not support model_type=%s" % args.model_type)

    # model config
    model_config = config_class.from_json_file(args.model_config)
    if args.pretrained_model_name is None:
        model_config.pretrained_model_name = args.pretrained_model_name

    args.output_mode = {}
    if args.alphabet_type == "gene":
        args.output_mode["gene_mask"] = "multi_class"
    args.output_mode["express_value_mask"] = "multi_class"
    if not args.no_express_sorted_embeddings:
        args.output_mode["express_sorted_mask"] = "multi_class"

    # model important parameters
    if args.nucleotide_input_type:
        model_config.nucleotide_input_type = args.nucleotide_input_type
    if args.pretrain_task_name:
        model_config.pretrain_task_name = args.pretrain_task_name
    if args.pretrain_task_weight:
        model_config.pretrain_task_weight = args.pretrain_task_weight
    if args.alphabet_type:
        model_config.alphabet_type = args.alphabet_type
    if args.embed_dim:
        model_config.embed_dim = args.embed_dim
    if args.ffn_size:
        model_config.ffn_size = args.ffn_size
    else:
        model_config.ffn_size = 4 * args.embed_dim
    if args.nucleotide_token_encoder_num_attention_heads:
        model_config.nucleotide_token_encoder_num_attention_heads = args.nucleotide_token_encoder_num_attention_heads
    if args.nucleotide_token_encoder_num_layers:
        model_config.nucleotide_token_encoder_num_layers = args.nucleotide_token_encoder_num_layers
    if args.nucleotide_token_pooling_type:
        model_config.nucleotide_token_pooling_type = args.nucleotide_token_pooling_type
    if args.num_attention_heads:
        model_config.num_attention_heads = args.num_attention_heads
    if args.num_layers:
        model_config.num_layers = args.num_layers

    if args.no_nucleotide_token_embeddings is not None:
        model_config.no_nucleotide_token_embeddings = args.no_nucleotide_token_embeddings
    if args.no_nucleotide_token_encoder is not None:
        model_config.no_nucleotide_token_encoder = args.no_nucleotide_token_encoder
        if args.no_nucleotide_token_encoder:
            model_config.nucleotide_token_encoder_num_attention_heads = 0
            args.nucleotide_token_encoder_num_attention_heads = 0
            model_config.nucleotide_token_encoder_num_layers = 0
            args.nucleotide_token_encoder_num_layers = 0
    if args.no_nucleotide_position_embeddings is not None:
        model_config.no_nucleotide_position_embeddings = args.no_nucleotide_position_embeddings
    if args.max_nucleotide_seq_length is not None:
        model_config.max_nucleotide_seq_length = args.max_nucleotide_seq_length
        model_config.max_nucleotide_position_embeddings = args.max_nucleotide_seq_length + 1
    if args.no_gene_positions_embeddings is not None:
        model_config.no_gene_positions_embeddings = args.no_gene_positions_embeddings
    if args.max_gene_seq_length is not None:
        model_config.max_gene_seq_length = args.max_gene_seq_length
        model_config.max_gene_position_embeddings = args.max_gene_seq_length + 1
    if args.no_express_sorted_embeddings is not None:
        model_config.no_express_sorted_embeddings = args.no_express_sorted_embeddings
    if args.max_express_sorted_position_embeddings is not None:
        model_config.max_express_sorted_position_embeddings = args.max_express_sorted_position_embeddings + 6
    if args.no_gene_type_embeddings is not None:
        model_config.no_gene_type_embeddings = args.no_gene_type_embeddings
    if args.ignore_index is not None:
        model_config.ignore_index = args.ignore_index
    if args.use_rotary_embeddings is not None:
        model_config.use_rotary_embeddings = args.use_rotary_embeddings
    if args.use_embed_layer_norm is not None:
        model_config.use_embed_layer_norm = args.use_embed_layer_norm
    if args.use_last_layer_norm is not None:
        model_config.use_last_layer_norm = args.use_last_layer_norm
    if args.pretrained_model_name is not None:
        model_config.pretrained_model_name = args.pretrained_model_name
    if args.dropout_prob is not None and args.dropout_prob > -1:
        model_config.attention_probs_dropout_prob = args.dropout_prob
        model_config.hidden_dropout_prob = args.dropout_prob
    if args.attention_probs_dropout_prob is not None and args.attention_probs_dropout_prob > -1:
        model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    if 0 < args.embed_scale < 1.0:
        model_config.embed_scale = args.embed_scale
    model_config.truncation_type = args.truncation
    model_config.mlm_probability = args.mlm_probability

    # load the pretrained model or create the model
    if args.model_dirpath:
        model = load_trained_model(model_config, args, model_class, args.model_dirpath)
    else:
        # create model
        model = model_class(model_config, args)
    if args.local_rank in [-1, 0]:
        save_model_parameters(model, os.path.join(args.output_dir, "init_parameters"))
    tokenizer = Alphabet.from_predefined(
        model_config.alphabet_type,
        gene_id_list_filepath=args.gene_id_list_filepath,
        express_bin_list_filepath=args.express_bin_list_filepath,
        express_sorted_list_filepath=args.express_sorted_list_filepath
    )
    return model_config, model, tokenizer


def create_logger(args):
    '''
    create logger
    :param args:
    :return:
    '''
    if args.local_rank in [-1, 0]:
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
        for dataset_type in ["train", "dev", "test"]:
            processed_samples_dir_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "processed_samples",
                args.time_str,
                dataset_type
            )
            if not os.path.exists(processed_samples_dir_path):
                os.makedirs(processed_samples_dir_path)
        exception_path = "../exception/%s/" % args.time_str
        if not os.path.exists(exception_path):
            os.makedirs(exception_path)
        debug_path = "../debug/%s/" % args.time_str
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        print("Output dir, logger, tb-logger created succeed.")
    else:
        log_fp = None
    return log_fp


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
            dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=54000))
            if args.local_rank == 0:
                print('world size: %d' % dist.get_world_size())
        else:
            device = torch.device("cuda")
    return device


def main():
    # get args
    args = get_args()
    print("mlm_probability: %f" % args.mlm_probability)

    # check args
    check_args(args)

    # create log dir
    log_fp = create_logger(args)

    # device
    args.device = create_device(args)

    # create model
    model_config, model, tokenizer = get_model(args)

    # encoder config
    encoder_config = {
        "add_special_tokens": args.add_special_tokens,
        "truncation": args.truncation
    }

    # file row parser
    # 文件记录解析函数
    encoder = Encoder(
        config=encoder_config,
        nucleotide_input_type=args.nucleotide_input_type,
        tokenizer=tokenizer,
        max_nucleotide_seq_length=args.max_nucleotide_seq_length,
        max_gene_seq_length=args.max_gene_seq_length,
        ignore_index=args.ignore_index,
        global_nucleotide_seqs_filepath=args.global_nucleotide_seqs_filepath,
        global_nucleotide_seq_embeddings_dirpath=args.global_nucleotide_seq_embeddings_dirpath,
        global_gene_positions_filepath=args.global_gene_positions_filepath,
        embedding_buffer_size=args.embedding_buffer_size
    )
    parse_row_func = encoder.encode

    # lucacell 独特的batch转换器
    batch_data_func = BatchConverter(
        alphabet=tokenizer,
        nucleotide_input_type=args.nucleotide_input_type,
        no_nucleotide_position_embeddings=model_config.no_nucleotide_position_embeddings,
        no_gene_positions_embeddings=model_config.no_gene_positions_embeddings,
        no_express_sorted_embeddings=model_config.no_express_sorted_embeddings,
        no_gene_type_embeddings=model_config.no_gene_type_embeddings,
        truncation_nucleotide_seq_length=args.max_nucleotide_seq_length,
        truncation_gene_seq_length=args.max_gene_seq_length,
        ignore_index=model.ignore_index,
        mlm_probability=args.mlm_probability
    )

    # write logs
    if args.local_rank in [0, -1]:
        print("n_gpu: %d" % args.n_gpu)

        args_dict = {}
        for attr, value in sorted(args.__dict__.items()):
            if attr != "device":
                args_dict[attr] = value
        log_fp.write(json.dumps(args_dict, indent=4, ensure_ascii=False) + "\n")
        '''
        args_json = json.dumps(vars(args), indent=4, ensure_ascii=False)
        log_fp.write(args_json + "\n")
        '''
        log_fp.write("#" * 50 + "\n")
        log_fp.write("n_gpu: %d\n" % args.n_gpu)
        log_fp.write("#" * 50 + "\n")

        log_fp.write("Encoder Config:\n %s\n" % str(encoder_config))
        log_fp.write("#" * 50 + "\n")
        log_fp.write("Model Config:\n %s\n" % str(model_config))
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

    # set seed
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
            dataset_type="train",
            header=True,
            shuffle=True,
            seed=args.seed
        )
    else:
        print("n_gpu: %d, use: DataLoader" % args.n_gpu)
        train_dataset = load_dataset(
            'csv',
            data_dir=args.train_data_dir,
            split='train',
            streaming=True
        )
        train_dataset = train_dataset.map(
            lambda x: parse_row_func(
                x["sample_id"],
                x["sample_type"],
                eval(x["gene_id_list"]) if "gene_id_list" in x else eval(x["feature_ids"]),
                eval(x["gene_express_list"]) if "gene_express_list" in x else eval(x["qcut_bins"]),
                eval(x["non_express_gene_id_list"]) if "non_express_gene_id_list" in x else eval(x["zero_freature_ids"])
            ),
            batched=False,
            remove_columns=["sample_id", "sample_type", "feature_ids", "qcut_bins", "zero_freature_ids"]
        )
        train_dataset = split_dataset_by_node(train_dataset, rank=args.local_rank, world_size=dist.get_world_size()) \
            .shuffle(buffer_size=args.buffer_size, seed=args.seed)
        train_dataset = train_dataset.with_format("torch")
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.per_gpu_train_batch_size,
            # sampler=train_sampler,
            # num_workers=args.worker_num,
            num_workers=args.worker_num,
            pin_memory=True,
            collate_fn=batch_data_func
        )
    if args.do_train:
        if args.trained_checkpoint is not None:
            global_step, avg_loss, max_metric_model_info = train_continue(
                args,
                model,
                model_config,
                dataloader=train_dataloader,
                parse_row_func=parse_row_func,
                batch_data_func=batch_data_func,
                tokenizer=tokenizer,
                train_sampler=None,
                log_fp=log_fp
            )
        else:
            global_step, avg_loss, max_metric_model_info = train_simple_v2(
                args,
                model,
                model_config,
                dataloader=train_dataloader,
                parse_row_func=parse_row_func,
                batch_data_func=batch_data_func,
                tokenizer=tokenizer,
                train_sampler=None,
                log_fp=log_fp
            )


if __name__ == "__main__":
    main()
