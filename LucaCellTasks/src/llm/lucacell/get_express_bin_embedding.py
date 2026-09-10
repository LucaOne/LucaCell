#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/10/9 20:51
@project: LucaCellTasks
@file: get_express_bin_embedding
@desc: xxxx
"""
import os
import sys
import torch
import argparse
import numpy as np
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from args import Args
    from file_operator import fasta_reader, csv_reader, tsv_reader
    from utils import set_seed, to_device, get_labels, get_parameter_number, \
        download_trained_checkpoint_lucaone_v2, download_trained_checkpoint_lucacell, \
        clean_seq_luca, available_gpu_id, load_trained_model, load_trained_logs, \
        calc_emb_filename_by_sample_id, matrix_2_vector, print_args, print_batch_input
    from llm.lucaone.get_embedding import embedding_for_one_sample as predict_seq_embedding_main
    from llm.lucacell.models.lucacell import LucaCell
    from llm.lucacell.models.configuration_lucacell import LucaCellConfig
    from llm.lucacell.models.alphabet import Alphabet
except ImportError as e:
    from src.args import Args
    from src.file_operator import fasta_reader, csv_reader, tsv_reader
    from src.utils import set_seed, to_device, get_labels, get_parameter_number, \
        download_trained_checkpoint_lucaone_v2, download_trained_checkpoint_lucacell, \
        clean_seq_luca, available_gpu_id, load_trained_model, load_trained_logs, \
        calc_emb_filename_by_sample_id, matrix_2_vector, print_args, print_batch_input
    from src.llm.lucaone.v2_0.lucaone_gplm_config import LucaGPLMConfig
    from src.llm.lucaone.get_embedding import embedding_for_one_sample as predict_seq_embedding_main
    from src.llm.lucacell.models.lucacell import LucaCell
    from src.llm.lucacell.models.configuration_lucacell import LucaCellConfig
    from src.llm.lucacell.models.alphabet import Alphabet
from transformers import PretrainedConfig

global_log_filepath, global_model_dirpath, global_args_info, \
global_model_config, global_model_version, global_model, global_tokenizer = None, None, None, None, None, None, None


def load_model(
        log_filepath,
        model_dirpath,
        embedding_inference=True
):
    """
    create tokenizer, model config, model
    :param log_filepath:
    :param model_dirpath:
    :param embedding_inference:
    :return:
    """
    model_dirpath = os.path.abspath(model_dirpath)
    assert model_dirpath is not None and os.path.exists(model_dirpath)
    print("Model dirpath: %s" % model_dirpath)
    args_info = load_trained_logs(log_filepath)
    # create tokenizer
    tokenizer_dir = os.path.join(model_dirpath, "tokenizer")
    gene_id_list_filepath = os.path.join(
        tokenizer_dir,
        os.path.basename(args_info["gene_id_list_filepath"])
    ) if args_info["gene_id_list_filepath"] else None
    express_bin_list_filepath = os.path.join(
        tokenizer_dir,
        os.path.basename(args_info["express_bin_list_filepath"])
    ) if args_info["express_bin_list_filepath"] else None
    express_sorted_list_filepath = os.path.join(
        tokenizer_dir,
        os.path.basename(args_info["express_sorted_list_filepath"])
    ) if args_info["express_sorted_list_filepath"] else None
    tokenizer = Alphabet.from_predefined(
        args_info["alphabet_type"],
        gene_id_list_filepath=gene_id_list_filepath,
        express_bin_list_filepath=express_bin_list_filepath,
        express_sorted_list_filepath=express_sorted_list_filepath,
    )
    # lucacell
    if args_info["model_type"] in ["lucacell"]:
        config_class, model_class = LucaCellConfig, LucaCell
    else:
        raise Exception("Not support model_type=%s" % args_info["model_type"])

    # model config
    model_config: PretrainedConfig = config_class.from_json_file(
        os.path.join(model_dirpath, "config.json")
    )

    # load the pretrained model or create the model
    print("Load pretrained model: %s" % model_dirpath)
    args = Args()
    args.pretrain_task_name = args_info["pretrain_task_name"]
    args.ignore_index = args_info["ignore_index"]
    args.max_gene_seq_length = args_info["max_gene_seq_length"]
    args.max_nucleotide_seq_length = args_info["max_nucleotide_seq_length"]
    args.pretrained_model_name = None
    args.embedding_inference = embedding_inference
    args.gene_id_list_filepath = gene_id_list_filepath
    args.express_bin_list_filepath = express_bin_list_filepath
    args.express_sorted_list_filepath = express_sorted_list_filepath
    model = load_trained_model(model_config, args, model_class, model_dirpath)
    return args_info, model_config, model, tokenizer

def get_args():
    parser = argparse.ArgumentParser(description='LucaCell Embedding')
    # for lucaone checkpoint
    parser.add_argument(
        "--lucaone_version",
        type=str,
        default="lucaone-gene",
        choices=["lucaone", "lucaone-gene"],
        help="the lucaone version"
    )
    parser.add_argument(
        "--lucaone_step",
        type=str,
        default="36800000",
        choices=["60000000", "36800000"],
        help="the lucaone checkpoint-step"
    )
    # for trained llm checkpoint
    parser.add_argument(
        "--llm_dir",
        type=str,
        default="../../",
        help="the llm model dir"
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        default="lucacell",
        choices=["lucacell", "lucacellv2"],
        help="the llm type"
    )
    parser.add_argument(
        "--llm_version",
        type=str,
        default="lucacell-2400/v1.0",
        choices=["lucacell-2400/v1.0", "lucacell-2048pos/v1.0"],
        help="the llm version"
    )
    parser.add_argument(
        "--llm_task_name",
        type=str,
        default="express_token_mask",
        choices=["express_token_mask"],
        help="the llm task name"
    )
    parser.add_argument(
        "--llm_time_str",
        type=str,
        default=None,
        help="the llm running time str"
    )
    parser.add_argument(
        "--llm_step",
        type=int,
        default=None,
        help="the llm checkpoint step."
    )
    input_args = parser.parse_args()
    return input_args


def read_fasta(fasta_path):
    seqs = {}
    for row in fasta_reader(fasta_path):
        seq_id = row[0]
        seq = row[1].upper().strip().replace("U", "T")
        if seq_id[0] == ">":
            seq_id = seq_id[1:]
        seqs[seq_id] = seq
    return seqs


def main(lucacell_args):
    global global_log_filepath, global_model_dirpath, global_args_info, \
        global_model_config, global_model, global_tokenizer
    print("*" * 20 + "LucaCell Model Args:" + "*" * 20)
    print_args(lucacell_args)
    print("*" * 50)

    if lucacell_args.llm_dir is None:
        lucacell_args.llm_dir = "../.."
    lucaone_args = Args()
    lucaone_args.llm_dir = lucacell_args.llm_dir
    lucaone_args.llm_type = "lucaone"
    lucaone_args.llm_version = lucacell_args.lucaone_version if lucacell_args.lucaone_version else "lucaone-gene"
    lucaone_args.llm_step = lucacell_args.lucaone_step if lucacell_args.lucaone_step else "36800000"
    # lucaone for gene seqs embedding
    download_trained_checkpoint_lucaone_v2(
        llm_dir=lucaone_args.llm_dir,
        llm_type=lucaone_args.llm_type,
        llm_version=lucaone_args.llm_version,
        llm_step=lucaone_args.llm_step
    )
    # lucacell for cell embedding
    download_trained_checkpoint_lucacell(
        llm_dir=lucacell_args.llm_dir,
        llm_type=lucacell_args.llm_type,
        llm_version=lucacell_args.llm_version,
        llm_task_name=lucacell_args.llm_task_name,
        llm_time_str=lucacell_args.llm_time_str,
        llm_step=lucacell_args.llm_step
    )
    cur_log_filepath = "%s/logs/%s/%s/%s/%s/logs.txt" % (
        lucacell_args.llm_dir if lucacell_args.llm_dir else "../..",
        lucacell_args.llm_type,
        lucacell_args.llm_version,
        lucacell_args.llm_task_name,
        lucacell_args.llm_time_str
    )
    print("Log filepath: %s" % os.path.abspath(cur_log_filepath))
    cur_model_dirpath = "%s/models/%s/%s/%s/%s/checkpoint-step%s" % (
        lucacell_args.llm_dir if lucacell_args.llm_dir else "../..",
        lucacell_args.llm_type,
        lucacell_args.llm_version,
        lucacell_args.llm_task_name,
        lucacell_args.llm_time_str,
        lucacell_args.llm_step
    )
    print("Model filepath: %s" % os.path.abspath(cur_model_dirpath))
    if global_log_filepath != cur_log_filepath or global_model_dirpath != cur_model_dirpath:
        global_log_filepath = cur_log_filepath
        global_model_dirpath = cur_model_dirpath
        global_args_info, global_model_config, global_model, global_tokenizer = load_model(
            log_filepath=global_log_filepath,
            model_dirpath=global_model_dirpath,
            embedding_inference=True
        )
        saved_path = "%s/express_bin_tokens_embed.pth" % cur_model_dirpath
        torch.save(global_model.express_tokens_embed.weight.data, saved_path)
        print("gene express bin embedding save to: %s" % saved_path)


if __name__ == "__main__":
    run_args = get_args()
    main(run_args)

"""
python get_express_bin_embedding.py \
    --lucaone_version lucaone-gene \
    --lucaone_step 36800000 \
    --llm_dir ../../../  \
    --llm_type lucacell \
    --llm_version lucacell-2400/v1.0 \
    --llm_task_name express_token_mask \
    --llm_time_str 20250726103509 \
    --llm_step xxxxx
"""

