#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/18 15:32
@project: LucaCell
@file: get_embedding
@desc: get embedding from pretrained llm
'''
import os
import sys
import torch
import argparse
import numpy as np
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../src")
try:
    from args import Args
    from file_operator import fasta_reader, csv_reader, tsv_reader
    from utils import set_seed, to_device, get_labels, get_parameter_number, \
        download_trained_checkpoint_lucaone_v2, download_trained_checkpoint_lucacell, \
        clean_seq_luca, available_gpu_id, load_trained_model, load_trained_logs, \
        calc_emb_filename_by_sample_id, matrix_2_vector, print_args, print_batch_input
    from lucaone.get_embedding import embedding_for_one_sample as predict_seq_embedding_main
    from models.lucacell import LucaCell
    from models.configuration_lucacell import LucaCellConfig
    from models.alphabet import Alphabet
except ImportError as e:
    from src.args import Args
    from src.file_operator import fasta_reader, csv_reader, tsv_reader
    from src.utils import set_seed, to_device, get_labels, get_parameter_number, \
        download_trained_checkpoint_lucaone_v2, download_trained_checkpoint_lucacell, \
        clean_seq_luca, available_gpu_id, load_trained_model, load_trained_logs, \
        calc_emb_filename_by_sample_id, matrix_2_vector, print_args, print_batch_input
    from src.lucaone.v2_0.lucaone_gplm_config import LucaGPLMConfig
    from src.lucaone.get_embedding import embedding_for_one_sample as predict_seq_embedding_main
    from src.models.lucacell import LucaCell
    from src.models.configuration_lucacell import LucaCellConfig
    from src.models.alphabet import Alphabet
from transformers import PretrainedConfig

global_log_filepath, global_model_dirpath, global_args_info, \
global_model_config, global_model_version, global_model, global_tokenizer = None, None, None, None, None, None, None


def attns_merge(
        save_type,
        seq_len,
        add_special_token,
        trunc_type,
        attn_list,
):
    if add_special_token:
        seq_len += 2
    shape = attn_list[0].shape
    n_layers, n_heads = shape[0], shape[1]
    if save_type == "tensor":
        merged_attns = torch.zeros(n_layers, n_heads, seq_len, seq_len)
    else:
        merged_attns = np.zeros(n_layers, n_heads, seq_len, seq_len)
    if trunc_type == "right":
        attn_list = attn_list[::-1]
    if add_special_token:
        sum_cls_eos_2_cls_eos = None
    for attn_item in attn_list:
        if add_special_token:
            start = attn_item[1]
            end = attn_item[2]
            merged_attns[:, :, start + 1: end + 1, start + 1: end + 1] = attn_item[:, :, 1:-1, 1:-1]
            # [CLS]->Other(exclude [EOS])
            merged_attns[:, :, 0, start + 1: end + 1] = attn_item[:, :, 0, 1:-1]
            # Other(exclude [EOS])->[CLS]
            merged_attns[:, :, start + 1: end + 1, 0] = attn_item[:, :,  1:-1, 0]
            # Other(exclude [CLS])->[EOS]
            merged_attns[:, :, start + 1: end + 1, end + 1] = attn_item[:, :,  1:-1, end + 1]
            # [EOS]->Other(exclude [CLS])
            merged_attns[:, :, end + 1, end + 1] = attn_item[:, :, end + 1, 1:-1]
            if sum_cls_eos_2_cls_eos is None:
                sum_cls_eos_2_cls_eos = [attn_item[:, :, 0, 0], attn_item[:, :, 0, -1], attn_item[:, :, -1, 0], attn_item[:, :, -1, -1]]
            else:
                sum_cls_eos_2_cls_eos[0] += attn_item[:, :, 0, 0]
                sum_cls_eos_2_cls_eos[1] += attn_item[:, :, 0, -1]
                sum_cls_eos_2_cls_eos[2] += attn_item[:, :, -1, 0]
                sum_cls_eos_2_cls_eos[3] += attn_item[:, :, -1, -1]
        else:
            start = attn_item[1]
            end = attn_item[2]
            merged_attns[:, :, start: end] = attn_item

    if add_special_token:
        # 第一个分片
        '''
        merged_attns[:, :, 0, 0] = attn_item[:, :, 0, 0]
        merged_attns[:, :, 0, seq_len - 1] = attn_item[:, :, 0, -1]
        merged_attns[:, :, seq_len - 1, 0] = attn_item[:, :, -1, 0]
        merged_attns[:, :, seq_len - 1, seq_len - 1] = attn_item[:, :, -1, -1]
        '''
        # 平均值赋值[CLS],[EOS]<->[CLS],[EOS]
        merged_attns[:, :, 0, 0] = sum_cls_eos_2_cls_eos[0]/len(attn_list)
        merged_attns[:, :, 0, seq_len - 1] = sum_cls_eos_2_cls_eos[2]/len(attn_list)
        merged_attns[:, :, seq_len - 1, 0] = sum_cls_eos_2_cls_eos[2]/len(attn_list)
        merged_attns[:, :, seq_len - 1, seq_len - 1] = sum_cls_eos_2_cls_eos[3]/len(attn_list)
    return merged_attns


def complete_embedding_matrix(
        lucaone_args,
        lucacell_args,
        sample_id,
        sample_type,
        sample_gene_id_list,
        sample_gene_seq_list,
        sample_gene_express_list,
        truncation_nucleotide_seq_length,
        truncation_gene_seq_length,
        init_emb,
        init_attns,
        embedding_type,
        matrix_add_special_token,
        save_type="numpy",
        gene_seq_emb_save_path=None,
        use_cpu=False,
        use_bf16=True,
):
    if init_emb is not None \
            and lucacell_args.embedding_complete \
            and ("representations" in embedding_type or "matrix" in embedding_type):
        torch.cuda.empty_cache()
        ori_seq_len = len(sample_gene_express_list)
        attn_list = []
        if init_attns is not None:
            if lucacell_args.trunc_type == "right":
                attn_list.append((init_attns, 0, init_attns.shape[3] - 2))
            else:
                attn_list.append((init_attns, ori_seq_len - init_attns.shape[3] + 2, ori_seq_len))
        # 每次能处理这么长度
        cur_segment_len = init_emb.shape[0]
        if matrix_add_special_token:
            complete_emb = init_emb[1:cur_segment_len - 1]
        else:
            complete_emb = init_emb
        if matrix_add_special_token:
            cur_segment_len = cur_segment_len - 2
        segment_num = int((ori_seq_len + cur_segment_len - 1)/cur_segment_len)
        if segment_num <= 1:
            return init_emb, init_attns
        if lucacell_args.embedding_complete_seg_overlap:
            sliding_window = cur_segment_len // 2
            print("Updated window: %d" % sliding_window)
            # 第一个已经处理，滑动窗口
            if lucacell_args.trunc_type == "right":
                last_end = cur_segment_len
                seg_idx = 0
                for pos_idx in range(cur_segment_len, ori_seq_len - sliding_window, sliding_window):
                    seg_idx += 1
                    last_end = min(pos_idx + sliding_window, ori_seq_len)
                    seg_gene_id_list = sample_gene_id_list[pos_idx - sliding_window:last_end]
                    seg_gene_seq_list = sample_gene_seq_list[pos_idx - sliding_window:last_end]
                    seg_gene_express_list = sample_gene_express_list[pos_idx - sliding_window:last_end]
                    print("segment idx: %d, seg express list len: %d" % (seg_idx, len(seg_gene_express_list)))
                    seg_emb, seg_attn, seg_processed_nucleotide_seq_len, seg_processed_gene_seq_len = predict_embedding(
                        lucaone_args,
                        global_model_dirpath,
                        [sample_id + "_seg_%d" % seg_idx, sample_type, seg_gene_id_list, seg_gene_seq_list, seg_gene_express_list],
                        embedding_type,
                        repr_layers=[-1],
                        truncation_nucleotide_seq_length=truncation_nucleotide_seq_length,
                        truncation_gene_seq_length=truncation_gene_seq_length,
                        device=lucacell_args.device if not use_cpu else torch.device("cpu"),
                        matrix_add_special_token=False,
                        save_type=save_type,
                        need_head_weights=lucacell_args.need_head_weights,
                        gene_seq_emb_save_path=gene_seq_emb_save_path,
                        use_bf16=use_bf16
                    )
                    # 有seq overlap 所以要截取
                    if complete_emb is None:
                        complete_emb = seg_emb[sliding_window:]
                    else:
                        if save_type == "numpy":
                            complete_emb = np.concatenate((complete_emb, seg_emb[sliding_window:]), axis=0)
                        else:
                            complete_emb = torch.cat((complete_emb, seg_emb[sliding_window:]), dim=0)
                    if lucacell_args.need_head_weights:
                        attn_list.append((seg_attn, pos_idx - sliding_window, last_end))
                if last_end < ori_seq_len:
                    seg_idx += 1
                    remain = ori_seq_len - last_end
                    seg_gene_id_list = sample_gene_id_list[ori_seq_len - 2 * sliding_window:ori_seq_len]
                    seg_gene_seq_list = sample_gene_seq_list[ori_seq_len - 2 * sliding_window:ori_seq_len]
                    seg_gene_express_list = sample_gene_express_list[ori_seq_len - 2 * sliding_window:ori_seq_len]
                    seg_emb, seg_attn, seg_processed_nucleotide_seq_len, seg_processed_gene_seq_len = predict_embedding(
                        lucaone_args,
                        global_model_dirpath,
                        [sample_id + "_seg_%d" % seg_idx, sample_type, seg_gene_id_list, seg_gene_seq_list, seg_gene_express_list],
                        embedding_type,
                        repr_layers=[-1],
                        truncation_nucleotide_seq_length=truncation_nucleotide_seq_length,
                        truncation_gene_seq_length=truncation_gene_seq_length,
                        device=lucacell_args.device if not use_cpu else torch.device("cpu"),
                        matrix_add_special_token=False,
                        save_type=save_type,
                        need_head_weights=lucacell_args.need_head_weights,
                        gene_seq_emb_save_path=gene_seq_emb_save_path,
                        use_bf16=use_bf16
                    )
                    # 有seq overlap 所以要截取
                    if complete_emb is None:
                        complete_emb = seg_emb[-remain:]
                    else:
                        if save_type == "numpy":
                            complete_emb = np.concatenate((complete_emb, seg_emb[-remain:]), axis=0)
                        else:
                            complete_emb = torch.cat((complete_emb, seg_emb[-remain:]), dim=0)
                    if lucacell_args.need_head_weights:
                        attn_list.append((seg_attn, ori_seq_len - 2 * sliding_window, ori_seq_len))
            else:
                last_start = -cur_segment_len
                seg_idx = 0
                for pos_idx in range(-cur_segment_len, -ori_seq_len + sliding_window, -sliding_window):
                    seg_idx += 1
                    last_start = max(pos_idx - sliding_window, -ori_seq_len)
                    seg_gene_id_list = sample_gene_id_list[last_start: pos_idx + sliding_window]
                    seg_gene_seq_list = sample_gene_seq_list[last_start: pos_idx + sliding_window]
                    seg_gene_express_list = sample_gene_express_list[last_start: pos_idx + sliding_window]
                    seg_emb, seg_attn, seg_processed_nucleotide_seq_len, seg_processed_gene_seq_len = predict_embedding(
                        lucaone_args,
                        global_model_dirpath,
                        [sample_id + "_seg_%d" % seg_idx, sample_type, seg_gene_id_list, seg_gene_seq_list, seg_gene_express_list],
                        embedding_type,
                        repr_layers=[-1],
                        truncation_nucleotide_seq_length=truncation_nucleotide_seq_length,
                        truncation_gene_seq_length=truncation_gene_seq_length,
                        device=lucacell_args.device if not use_cpu else torch.device("cpu"),
                        matrix_add_special_token=False,
                        save_type=save_type,
                        need_head_weights=lucacell_args.need_head_weights,
                        gene_seq_emb_save_path=gene_seq_emb_save_path,
                        use_bf16=use_bf16
                    )
                    # 有seq overlap 所以要截取
                    if complete_emb is None:
                        complete_emb = seg_emb[:sliding_window]
                    else:
                        if save_type == "numpy":
                            complete_emb = np.concatenate((seg_emb[:sliding_window], complete_emb), axis=0)
                        else:
                            complete_emb = torch.cat((seg_emb[:sliding_window], complete_emb), dim=0)
                    if lucacell_args.need_head_weights:
                        attn_list.append(seg_attn)
                if last_start > -ori_seq_len:
                    seg_idx += 1
                    remain = last_start + ori_seq_len
                    seg_gene_id_list = sample_gene_id_list[-ori_seq_len:-ori_seq_len + 2 * sliding_window]
                    seg_gene_seq_list = sample_gene_seq_list[-ori_seq_len:-ori_seq_len + 2 * sliding_window]
                    seg_gene_express_list = sample_gene_express_list[-ori_seq_len:-ori_seq_len + 2 * sliding_window]
                    seg_emb, seg_attn, seg_processed_nucleotide_seq_len, seg_processed_gene_seq_len = predict_embedding(
                        lucaone_args,
                        global_model_dirpath,
                        [sample_id + "_seg_%d" % seg_idx, sample_type, seg_gene_id_list, seg_gene_seq_list, seg_gene_express_list],
                        embedding_type,
                        repr_layers=[-1],
                        truncation_nucleotide_seq_length=truncation_nucleotide_seq_length,
                        truncation_gene_seq_length=truncation_gene_seq_length,
                        device=lucacell_args.device if not use_cpu else torch.device("cpu"),
                        matrix_add_special_token=False,
                        save_type=save_type,
                        need_head_weights=lucacell_args.need_head_weights,
                        gene_seq_emb_save_path=gene_seq_emb_save_path,
                        use_bf16=use_bf16
                    )
                    # 有seq overlap 所以要截取
                    if complete_emb is None:
                        complete_emb = seg_emb[:remain]
                    else:
                        if save_type == "numpy":
                            complete_emb = np.concatenate((seg_emb[:remain], complete_emb), axis=0)
                        else:
                            complete_emb = torch.cat((seg_emb[:remain], complete_emb), dim=0)
                    if lucacell_args.need_head_weights:
                        attn_list.append((seg_attn, 0, 2 * sliding_window))
                if lucacell_args.need_head_weights:
                    attn_list = attn_list[::-1]
        else:
            # 第一个已经处理，最后一个单独处理（需要向左/向右扩充至cur_segment_len长度）
            if lucacell_args.trunc_type == "right":
                begin_seq_idx = 0
            else:
                begin_seq_idx = ori_seq_len - (segment_num - 1) * cur_segment_len
            for seg_idx in range(1, segment_num - 1):
                seg_gene_id_list = sample_gene_id_list[begin_seq_idx + seg_idx * cur_segment_len: begin_seq_idx + (seg_idx + 1) * cur_segment_len]
                seg_gene_seq_list = sample_gene_seq_list[begin_seq_idx + seg_idx * cur_segment_len: begin_seq_idx + (seg_idx + 1) * cur_segment_len]
                seg_gene_express_list = sample_gene_express_list[begin_seq_idx + seg_idx * cur_segment_len: begin_seq_idx + (seg_idx + 1) * cur_segment_len]
                seg_emb, seg_attn, seg_processed_nucleotide_seq_len, seg_processed_gene_seq_len = predict_embedding(
                    lucaone_args,
                    global_model_dirpath,
                    [sample_id + "_seg_%d" % seg_idx, sample_type, seg_gene_id_list, seg_gene_seq_list, seg_gene_express_list],
                    embedding_type,
                    repr_layers=[-1],
                    truncation_nucleotide_seq_length=truncation_nucleotide_seq_length,
                    truncation_gene_seq_length=truncation_gene_seq_length,
                    device=lucacell_args.device if not use_cpu else torch.device("cpu"),
                    matrix_add_special_token=False,
                    save_type=save_type,
                    gene_seq_emb_save_path=gene_seq_emb_save_path,
                    use_bf16=use_bf16
                )
                '''
                if lucacell_args.trunc_type == "right":
                    if save_type == "numpy":
                        complete_emb = np.concatenate((complete_emb, seg_emb), axis=0)
                    else:
                        complete_emb = torch.cat((complete_emb, seg_emb), dim=0)
                else:
                    if save_type == "numpy":
                        complete_emb = np.concatenate((seg_emb, complete_emb), axis=0)
                    else:
                        complete_emb = torch.cat((seg_emb, complete_emb), dim=0)
                '''
                if save_type == "numpy":
                    complete_emb = np.concatenate((complete_emb, seg_emb), axis=0)
                else:
                    complete_emb = torch.cat((complete_emb, seg_emb), dim=0)
                if lucacell_args.need_head_weights:
                    attn_list.append((seg_attn, begin_seq_idx + seg_idx * cur_segment_len, begin_seq_idx + (seg_idx + 1) * cur_segment_len))
            if lucacell_args.trunc_type == "right":
                # 处理最后一个
                last_seg_gene_id_list = sample_gene_id_list[-cur_segment_len:]
                last_seg_gene_seq_list = sample_gene_seq_list[-cur_segment_len:]
                last_seg_gene_express_list = sample_gene_express_list[-cur_segment_len:]
                really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                last_seg_emb, last_seg_attn, last_seg_processed_nucleotide_seq_len, last_seg_processed_gene_seq_len = predict_embedding(
                    lucaone_args,
                    global_model_dirpath,
                    [sample_id + "_seg_%d" % (segment_num - 1), sample_type, last_seg_gene_id_list, last_seg_gene_seq_list, last_seg_gene_express_list],
                    embedding_type,
                    repr_layers=[-1],
                    truncation_nucleotide_seq_length=truncation_nucleotide_seq_length,
                    truncation_gene_seq_length=truncation_gene_seq_length,
                    device=lucacell_args.device if not use_cpu else torch.device("cpu"),
                    matrix_add_special_token=False,
                    save_type=save_type,
                    need_head_weights=lucacell_args.need_head_weights,
                    gene_seq_emb_save_path=gene_seq_emb_save_path,
                    use_bf16=use_bf16
                )
                last_seg_emb = last_seg_emb[-really_len:, :]
                if save_type == "numpy":
                    complete_emb = np.concatenate((complete_emb, last_seg_emb), axis=0)
                else:
                    complete_emb = torch.cat((complete_emb, last_seg_emb), dim=0)
                if lucacell_args.need_head_weights:
                    attn_list.append((last_seg_attn, ori_seq_len - cur_segment_len, ori_seq_len))
            else:
                # 处理第一个
                first_seg_gene_id_list = sample_gene_id_list[:cur_segment_len]
                first_seg_gene_seq_list = sample_gene_seq_list[:cur_segment_len]
                first_seg_gene_express_list = sample_gene_express_list[:cur_segment_len]
                really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                first_seg_emb, first_seg_attn, first_seg_processed_nucleotide_seq_len, first_seg_processed_seq_len = predict_embedding(
                    lucaone_args,
                    global_model_dirpath,
                    [sample_id + "_seg_0", sample_type, first_seg_gene_id_list, first_seg_gene_seq_list, first_seg_gene_express_list],
                    embedding_type,
                    repr_layers=[-1],
                    truncation_nucleotide_seq_length=truncation_nucleotide_seq_length,
                    truncation_gene_seq_length=truncation_gene_seq_length,
                    device=lucacell_args.device if not use_cpu else torch.device("cpu"),
                    matrix_add_special_token=False,
                    save_type=save_type,
                    need_head_weights=lucacell_args.need_head_weights,
                    gene_seq_emb_save_path=gene_seq_emb_save_path,
                    use_bf16=use_bf16
                )
                first_seg_emb = first_seg_emb[:really_len, :]
                if save_type == "numpy":
                    complete_emb = np.concatenate((first_seg_emb, complete_emb), axis=0)
                else:
                    complete_emb = torch.cat((first_seg_emb, complete_emb), dim=0)
                if lucacell_args.need_head_weights:
                    attn_list = [(first_seg_attn, 0, cur_segment_len)] + attn_list
        if lucacell_args.need_head_weights:
            init_attns = attns_merge(save_type, ori_seq_len, matrix_add_special_token, lucacell_args.trunc_type, attn_list)
        print("gene list len: %d, embedding matrix len: %d" % (
            ori_seq_len,
            complete_emb.shape[0] + (2 if matrix_add_special_token else 0)
        ))
        assert complete_emb.shape[0] == ori_seq_len
        if matrix_add_special_token:
            if save_type == "numpy":
                complete_emb = np.concatenate((init_emb[0:1, :], complete_emb, init_emb[-1:, :]), axis=0)
            else:
                complete_emb = torch.cat((init_emb[0:1, :], complete_emb, init_emb[-1:, :]), dim=0)
        init_emb = complete_emb
    return init_emb, init_attns


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


def encoder(
        lucaone_args,
        lucacell_args,
        lucacell_config,
        sample_id,
        sample_gene_id_list,
        sample_gene_seq_list,
        sample_gene_express_list,
        tokenizer,
        gene_seq_emb_save_path,
        device
):
    truncation_gene_seq_length = lucacell_args["max_gene_seq_length"] - 2
    truncation_type_for_gene_seq = lucacell_config.truncation if hasattr(lucacell_config, "truncation") else lucacell_args["truncation"]
    assert lucacell_args["model_type"] in ["lucacell"]
    assert len(sample_gene_id_list) == len(sample_gene_express_list)
    gene_seq_emb_list = []
    if len(sample_gene_id_list) > truncation_gene_seq_length:
        if truncation_type_for_gene_seq == "right":
            sample_gene_id_list = sample_gene_id_list[:truncation_gene_seq_length]
            sample_gene_express_list = sample_gene_express_list[:truncation_gene_seq_length]
            if sample_gene_seq_list and len(sample_gene_seq_list) > 0:
                sample_gene_seq_list = sample_gene_seq_list[:truncation_gene_seq_length]
            sample_gene_express_list = sample_gene_express_list[:truncation_gene_seq_length]
        else:
            sample_gene_id_list = sample_gene_id_list[-truncation_gene_seq_length:]
            if sample_gene_seq_list and len(sample_gene_seq_list) > 0:
                sample_gene_seq_list = sample_gene_seq_list[-truncation_gene_seq_length:]
            sample_gene_express_list = sample_gene_express_list[-truncation_gene_seq_length:]
        processed_gene_seq_len = truncation_gene_seq_length + 2
    else:
        processed_gene_seq_len = len(sample_gene_id_list) + 2
    truncation_nucleotide_seq_length = lucacell_args["max_nucleotide_seq_length"] - 2
    processed_nucleotide_seq_len = 0
    lucaone_model = None
    for gene_idx, gene_id in enumerate(sample_gene_id_list):
        seq_embed_vector = None
        if gene_seq_emb_save_path:
            gene_seq_emb_vector_filename = calc_emb_filename_by_sample_id(gene_id, "vector")
            for dirpath in gene_seq_emb_save_path:
                gene_seq_emb_vector_filepath = os.path.join(dirpath, gene_seq_emb_vector_filename)
                if os.path.exists(gene_seq_emb_vector_filepath):
                    seq_embed_vector = torch.load(gene_seq_emb_vector_filepath, weights_only=True)
                    break
        while seq_embed_vector is None:
            print("gene_id=%s 's embedding not exists in advance: %s" % (gene_id, str(gene_seq_emb_save_path)))
            '''
            if len(gene_seq) > 2500:
                cur_device = torch.device("cpu")
            else:
                cur_device = device
            seq_embed_matrix, cur_processed_nucleotide_seq = predict_seq_embedding(
                llm_dirpath=lucaone_config.llm_dirpath,
                sample=[gene_id, "gene", gene_seq],
                trunc_type=truncation_type_for_nucleotide_seq,
                embedding_type="matrix",
                repr_layers=[-1],
                truncation_seq_length=truncation_nucleotide_seq_length,
                device=cur_device,
                matrix_add_special_token=True,
                save_type="tensor",
                use_bf16=use_bf16
            )
            cur_processed_nucleotide_seq_len = len(cur_processed_nucleotide_seq) + 2
            '''
            lucaone_args.seq_id = gene_id
            lucaone_args.seq_type = "gene"
            lucaone_args.seq = sample_gene_seq_list[gene_idx]
            seq_embed_matrix, cur_processed_nucleotide_seq_len, lucaone_model = predict_seq_embedding_main(
                lucaone_args
            )
            if seq_embed_matrix is None:
                truncation_nucleotide_seq_length = int(truncation_nucleotide_seq_length * 0.95)
                print("sample_id: %s, gene_id: %d, seq truncation to: %d" % (sample_id, gene_id, truncation_nucleotide_seq_length))
            else:
                processed_nucleotide_seq_len = max(cur_processed_nucleotide_seq_len, processed_nucleotide_seq_len)
                seq_embed_vector = torch.mean(seq_embed_matrix[1:-1], dim=0)
                # print("gene_seq_emb_vector_filepath: %s" % gene_seq_emb_vector_filepath)
                torch.save(seq_embed_vector, gene_seq_emb_vector_filepath)
                break
        gene_seq_emb_list.append(seq_embed_vector)
    gene_seq_embs = torch.stack(gene_seq_emb_list, dim=0)
    if lucaone_model:
        lucaone_model.to("cpu")
    torch.cuda.empty_cache()
    express_encoded = tokenizer.express_encode(sample_gene_express_list, add_special_token=True)
    return {
        "sample_ids": [sample_id],
        "nucleotide_input_ids": None,
        "nucleotide_input_embeds": gene_seq_embs.unsqueeze(dim=0).to(device),
        "gene_input_ids": None,
        "gene_type_ids": None,
        "express_input_ids": torch.tensor([express_encoded], dtype=torch.long).to(device),
        "express_sorted_position_ids": None,
        "labels": None
    }, processed_nucleotide_seq_len, processed_gene_seq_len


def get_embedding(
        lucaone_args,
        lucacell_args,
        lucacell_config,
        tokenizer,
        model,
        sample_id,
        sample_gene_id_list,
        sample_gene_seq_list,
        sample_gene_express_list,
        need_head_weights,
        gene_seq_emb_save_path,
        device
):
    model.to("cpu")
    if lucacell_args["model_type"] in ["lucacell"]:
        batch, processed_nucleotide_seq_len, processed_gene_seq_len = encoder(
            lucaone_args,
            lucacell_args,
            lucacell_config,
            sample_id,
            sample_gene_id_list,
            sample_gene_seq_list,
            sample_gene_express_list,
            tokenizer,
            gene_seq_emb_save_path,
            device
        )
        if need_head_weights:
            batch["need_head_weights"] = True
        else:
            batch["need_head_weights"] = False
        batch["need_weights"] = False
        batch["output_hidden_states"] = True
        batch["return_dict"] = True
        batch["repr_layers"] = list(range(lucacell_args["num_layers"] + 1))
    else:
        raise Exception("Not support the model_tye=%s" % lucacell_args["model_type"])
    print("Cell LLM embedding device:", device)
    model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            if "use_bf16" in lucacell_args and lucacell_args["use_bf16"]:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model(**batch)
            else:
                output = model(**batch)
            return output, processed_nucleotide_seq_len, processed_gene_seq_len
    except Exception as e:
        print("Get embedding:", e)
        return None, None, None


def predict_embedding(
        lucaone_args,
        llm_dirpath,
        sample,
        embedding_type,
        repr_layers=[-1],
        truncation_nucleotide_seq_length=4096,
        truncation_gene_seq_length=4096,
        device=None,
        matrix_add_special_token=False,
        save_type="numpy",
        need_head_weights=False,
        gene_seq_emb_save_path=None,
        use_bf16=True,
):
    """
    use sequence to predict cell embedding matrix or vector(bos)
    :param lucaone_args: lucaone running args
    :param llm_dirpath: llm dir path
    :param sample: [sample_id, sample_type, sample_gene_seq_list, sample_gene_express_list]
    :param embedding_type: bos or representations
    :param repr_layers: [-1]
    :param truncation_nucleotide_seq_length: [4096, 2048, 1984, 1792, 1536, 1280, 1152, 1024]
    :param truncation_gene_seq_length: [4096, 2048, 1984, 1792, 1536, 1280, 1152, 1024]
    :param device:
    :param matrix_add_special_token:
    :param save_type:
    :param use_bf16:
    :param need_head_weights:
    :param gene_seq_emb_save_path:
    :return: embedding, processed_seq_len
    """
    global global_log_filepath, global_model_dirpath, global_args_info, \
        global_model_config, global_model_version, global_model, global_tokenizer

    assert "bos" in embedding_type \
           or "representations" in embedding_type \
           or "matrix" in embedding_type \
           or "vector" in embedding_type

    sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_list = sample[0], sample[1], sample[2], sample[3], sample[4]
    cur_log_filepath = os.path.join(os.path.dirname(llm_dirpath).replace("models", "logs"), "logs.txt")
    cur_model_dirpath = llm_dirpath
    if global_log_filepath != cur_log_filepath or global_model_dirpath != cur_model_dirpath:
        global_log_filepath = cur_log_filepath
        global_model_dirpath = cur_model_dirpath
        global_args_info, global_model_config, global_model, global_tokenizer = \
            load_model(
                log_filepath=global_log_filepath,
                model_dirpath=global_model_dirpath,
                embedding_inference=True
            )
    global_args_info["max_nucleotide_seq_length"] = truncation_nucleotide_seq_length + 2
    global_args_info["max_gene_seq_length"] = truncation_gene_seq_length + 2
    global_args_info["use_bf16"] = use_bf16

    if device is None and not torch.cuda.is_available():
        device = torch.device("cpu")
    elif device is None and torch.cuda.is_available():
        device = torch.device("cuda")
    emb, processed_nucleotide_seq_len, processed_gene_seq_len = get_embedding(
        lucaone_args,
        global_args_info,
        global_model_config,
        global_tokenizer,
        global_model,
        sample_id,
        sample_gene_id_list,
        sample_gene_seq_list,
        sample_gene_express_list,
        need_head_weights,
        gene_seq_emb_save_path,
        device
    )
    if emb is None:
        return None, None, None, None

    embeddings = emb.hidden_states
    attentions = None
    if need_head_weights:
        attentions = emb.attentions[0, :, :, :, :]
    if "representations" in embedding_type or "matrix" in embedding_type:
        if use_bf16:
            embeddings = embeddings.float()
        if matrix_add_special_token:
            embedding = embeddings[0, 0: processed_gene_seq_len, :].to(device="cpu").clone()
        else:
            embedding = embeddings[0, 1: processed_gene_seq_len - 1, :].to(device="cpu").clone()
    elif "bos" in embedding_type or "vector" in embedding_type:
        if use_bf16:
            embeddings = embeddings.float()
        embedding = embeddings[0, 0, :].to(device="cpu").clone()
    else:
        raise Exception("Not support embedding_type=%s" % embedding_type)
    if save_type == "numpy":
        embedding = embedding.numpy()
    return embedding, attentions, processed_nucleotide_seq_len, processed_gene_seq_len


def get_args():
    parser = argparse.ArgumentParser(description='LucaCell Embedding')
    # for lucaone checkpoint
    parser.add_argument(
        "--lucaone_version",
        type=str,
        default="lucaone",
        choices=["lucaone"],
        help="the lucaone version"
    )
    parser.add_argument(
        "--lucaone_step",
        type=str,
        default="60000000",
        choices=["60000000"],
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
        choices=["lucacell"],
        help="the llm type"
    )
    parser.add_argument(
        "--llm_version",
        type=str,
        default="lucacell-2048pos/v1.0",
        choices=["lucacell-2048pos/v1.0"],
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

    # for one sample
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="the sample id"
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default=None,
        help="the sample type"
    )
    parser.add_argument(
        "--sample_gene_seq_list",
        type=str,
        default=None,
        help="the sample gene seq list(Comma separated)"
    )
    parser.add_argument(
        "--sample_gene_express_list",
        type=str,
        default=None,
        help="the sample gene express bin list(Comma separated)"
    )

    # for many
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="the input file（format: csv or tsv)"
    )
    # for input csv
    parser.add_argument(
        "--sample_id_idx",
        type=int,
        default=None,
        help="sample id col idx(0 start)"
    )
    parser.add_argument(
        "--sample_type_idx",
        type=int,
        default=None,
        help="sample type col idx(0 start)"
    )
    parser.add_argument(
        "--sample_gene_seq_meta_fasta",
        type=str,
        default=None,
        help="set the gene seqs fasta file when gene seqs not in the input file(Comma separated for multi fasta files)"
    )
    parser.add_argument(
        "--sample_gene_seq_list_idx",
        type=int,
        default=None,
        help="sample gene seqs col idx(0 start)"
    )
    parser.add_argument(
        "--sample_gene_id_list_idx",
        type=int,
        default=None,
        help="sample gene ids col idx(0 start)"
    )
    parser.add_argument(
        "--sample_gene_express_list_idx",
        type=int,
        default=None,
        help="sample gene express col idx(0 start)"
    )
    parser.add_argument(
        "--sample_non_express_gene_id_list_idx",
        type=int,
        default=None,
        help="sample non express gene ids list col idx(0 start)"
    )
    parser.add_argument(
        "--sample_non_express_gene_seq_list_idx",
        type=int,
        default=None,
        help="sample non express gene seqs col idx(0 start)"
    )
    parser.add_argument(
        "--need_head_weights",
        action="store_true",
        help="whether to save attention scores matrix, shape: n_layers x n_heads x L x L"
    )

    # for saved path
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="embedding file save dir path"
    )

    # for embedding
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="matrix",
        choices=["matrix", "vector"],
        help="the llm embedding type."
    )
    parser.add_argument(
        "--save_type",
        type=str,
        default="numpy",
        choices=["tensor", "numpy"],
        help="the embedding save type(tensor or numpy)."
    )
    parser.add_argument(
        "--vector_type",
        type=str,
        default="mean",
        choices=["mean", "max", "cls"],
        help="the llm vector embedding type."
    )
    parser.add_argument(
        "--trunc_type",
        type=str,
        default="right",
        choices=["left", "right"],
        help="llm trunc type."
    )
    parser.add_argument(
        "--truncation_nucleotide_seq_length",
        type=int,
        default=10240,
        help="the llm truncation nucleotide seq length(not contain [CLS] and [SEP]."
    )
    parser.add_argument(
        "--truncation_gene_seq_length",
        type=int,
        default=4096,
        help="the llm truncation gene seq length(not contain [CLS] and [SEP]."
    )
    parser.add_argument(
        "--matrix_add_special_token",
        action="store_true",
        help="whether to add special token embedding vector in seq representation matrix"
    )
    parser.add_argument(
        "--embedding_complete",
        action="store_true",
        help="when the seq len > inference_max_len, then the embedding matrix is completed by segment")

    parser.add_argument(
        "--embedding_complete_seg_overlap",
        action="store_true",
        help="segment overlap"
    )
    parser.add_argument(
        "--embedding_fixed_len_a_time_for_seq",
        type=int,
        default=None,
        help="the embedding fixed length of once inference for longer gene sequqence"
    )
    parser.add_argument(
        "--embedding_fixed_len_a_time_for_cell",
        type=int,
        default=None,
        help="the embedding fixed length of once inference for longer gene list size"
    )
    parser.add_argument(
        "--gene_seq_emb_dirpath",
        type=str,
        default=None,
        help="the gene embedding exists in advance"
    )
    parser.add_argument(
        "--global_gene_positions_filepath",
        type=str,
        default=None,
        required=True,
        help="global gene positions dirpath"
    )
    parser.add_argument(
        '--use_non_express_gene_list',
        action='store_true',
        help='whether to use non express gene list'
    )

    # for running
    parser.add_argument(
        '--use_bf16',
        action='store_true',
        help='whether to use bf16 for training'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help="the gpu id to use."
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


def load_global_gene_positions(global_gene_positions_filepath):
    global_gene_positions_filepath = global_gene_positions_filepath.split(";")
    global_gene_positions = {}
    for filepath in global_gene_positions_filepath:
        for row in csv_reader(filepath):
            gene_id, position = row[0], int(row[1])
            if gene_id[0] == ">":
                gene_id = gene_id[1:]
            global_gene_positions[gene_id] = position
    return global_gene_positions


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
    lucaone_args.llm_version = lucacell_args.lucaone_version if lucacell_args.lucaone_version else "lucaone"
    lucaone_args.llm_step = lucacell_args.lucaone_step if lucacell_args.lucaone_step else "60000000"
    lucaone_args.embedding_type = "matrix"
    lucaone_args.save_type = "tensor"
    lucaone_args.vector_type = None
    lucaone_args.trunc_type = "right"
    lucaone_args.truncation_seq_length = lucacell_args.truncation_nucleotide_seq_length
    lucaone_args.matrix_add_special_token = True
    lucaone_args.embedding_complete = True
    lucaone_args.embedding_complete_seg_overlap = True
    lucaone_args.embedding_fixed_len_a_time = lucacell_args.embedding_fixed_len_a_time_for_seq
    lucaone_args.gpu_id = lucacell_args.gpu_id
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
    if lucacell_args.gpu_id >= 0:
        gpu_id = lucacell_args.gpu_id
    else:
        gpu_id = -1
        print("gpu_id: ", gpu_id)
    lucacell_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    # model.to(lucacell_args.device)
    need_head_weights = lucacell_args.need_head_weights
    assert (lucacell_args.input_file is not None and os.path.exists(lucacell_args.input_file)) or lucacell_args.seq is not None
    print("LucaCell args device: %s" % lucacell_args.device)
    embedding_type = lucacell_args.embedding_type
    vector_type = lucacell_args.vector_type
    if embedding_type == "vector" and vector_type == "cls":
        matrix_add_special_token = True
    else:
        matrix_add_special_token = lucacell_args.matrix_add_special_token
    cell_emb_save_path = lucacell_args.save_path
    print("Emb of Cell(Gene Seq + Express) save dir: %s" % os.path.abspath(cell_emb_save_path))
    if not os.path.exists(cell_emb_save_path):
        os.makedirs(cell_emb_save_path)
    gene_seq_emb_exists_in_advance = False
    if lucacell_args.gene_seq_emb_dirpath is not None:
        print("Emb of Seq(Gene Seq LucaOne) exists in advance: %s" % lucacell_args.gene_seq_emb_dirpath)
        gene_seq_emb_save_path = lucacell_args.gene_seq_emb_dirpath.split("#")
        for dirpath in gene_seq_emb_save_path:
            assert os.path.exists(dirpath)
        gene_seq_emb_exists_in_advance = True
    else:
        gene_seq_emb_save_path = os.path.join(cell_emb_save_path, "gene_seq_embeddings")
        print("Emb of Seq(Gene Seq LucaOne) save dir: %s" % os.path.abspath(gene_seq_emb_save_path))
        if not os.path.exists(gene_seq_emb_save_path):
            os.makedirs(gene_seq_emb_save_path)
        gene_seq_emb_save_path = [gene_seq_emb_save_path]

    global_gene_positions = load_global_gene_positions(lucacell_args.global_gene_positions_filepath)
    # 如果输入的文件中没有序列信息，则通过meta指定的fasta文件中加载
    if lucacell_args.input_file and os.path.exists(lucacell_args.input_file):
        gene_seq_from_fasta_flag = False
        gene_seq_fasta = {}
        assert lucacell_args.sample_gene_id_list_idx is not None
        if lucacell_args.sample_gene_seq_list_idx is None and not gene_seq_emb_exists_in_advance:
            gene_seq_from_fasta_flag = True
            for fasta_file in lucacell_args.sample_gene_seq_meta_fasta.split(","):
                gene_seq_fasta.update(read_fasta(fasta_file))
        done = 0
        file_reader = fasta_reader
        if lucacell_args.input_file.endswith(".csv"):
            file_reader = csv_reader
        elif lucacell_args.input_file.endswith(".tsv"):
            file_reader = tsv_reader

        assert not lucacell_args.use_non_express_gene_list or lucacell_args.sample_non_express_gene_id_list_idx > 0
        for row in file_reader(lucacell_args.input_file):
            torch.cuda.empty_cache()
            sample_non_express_gene_id_list = None
            sample_non_express_gene_seq_list = None
            if lucacell_args.use_non_express_gene_list:
                sample_non_express_gene_id_list = eval(row[lucacell_args.sample_non_express_gene_id_list_idx])
            # 基因序列需要从meta-fasta中通过gene_id获取
            if gene_seq_from_fasta_flag:
                sample_id, sample_type, sample_gene_id_list, sample_gene_express_list = \
                    row[lucacell_args.sample_id_idx].strip(), \
                    row[lucacell_args.sample_type_idx], \
                    eval(row[lucacell_args.sample_gene_id_list_idx]), \
                    eval(row[lucacell_args.sample_gene_express_list_idx])
                sample_gene_seq_list = [gene_seq_fasta[gene_id] for gene_id in sample_gene_id_list]
                sample_non_express_gene_seq_list = [gene_seq_fasta[gene_id] for gene_id in sample_non_express_gene_id_list] if sample_non_express_gene_id_list else None
            else:
                if gene_seq_emb_exists_in_advance:
                    # 基因序列在输入文件中有
                    sample_id, sample_type, sample_gene_id_list, sample_gene_express_list = \
                        row[lucacell_args.sample_id_idx].strip(), \
                            row[lucacell_args.sample_type_idx], \
                            eval(row[lucacell_args.sample_gene_id_list_idx]), \
                            eval(row[lucacell_args.sample_gene_express_list_idx])
                    sample_gene_seq_list = []
                    sample_non_express_gene_seq_list = [] if sample_non_express_gene_id_list else None
                else:
                    # 基因序列在输入文件中有
                    sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_list = \
                        row[lucacell_args.sample_id_idx].strip(), \
                        row[lucacell_args.sample_type_idx], \
                        eval(row[lucacell_args.sample_gene_id_list_idx]), \
                        eval(row[lucacell_args.sample_gene_seq_list_idx]), \
                        eval(row[lucacell_args.sample_gene_express_list_idx])
                    sample_non_express_gene_seq_list = eval(row[lucacell_args.sample_non_express_gene_seq_list_idx]) if sample_non_express_gene_id_list else None
            emb_filename = calc_emb_filename_by_sample_id(
                sample_id=sample_id,
                embedding_type=embedding_type
            )

            embedding_filepath = os.path.join(cell_emb_save_path, emb_filename)
            if need_head_weights:
                attn_filename = calc_emb_filename_by_sample_id(
                    sample_id=sample_id,
                    embedding_type="attention"
                )
                attn_dirpath = os.path.join(cell_emb_save_path, "attentions")
                if not os.path.exists(attn_dirpath):
                    os.makedirs(attn_dirpath)
                attn_filepath = os.path.join(attn_dirpath, attn_filename)
            if not os.path.exists(embedding_filepath):
                new_sample_gene_id_list = [(gene_id, global_gene_positions[gene_id]) for gene_id in sample_gene_id_list]
                if sample_gene_seq_list and len(sample_gene_seq_list) > 0:
                    new_sample_gene_seq_list = [(sample_gene_seq_list[gene_idx], global_gene_positions[gene_id]) for gene_idx, gene_id in enumerate(sample_gene_id_list)]
                else:
                    new_sample_gene_seq_list = []
                new_sample_gene_express_list = [(sample_gene_express_list[gene_idx], global_gene_positions[gene_id]) for gene_idx, gene_id in enumerate(sample_gene_id_list)]
                if lucacell_args.use_non_express_gene_list and sample_non_express_gene_id_list:
                    new_sample_non_express_gene_id_list = [(gene_id, global_gene_positions[gene_id]) for gene_id in sample_non_express_gene_id_list]
                    new_sample_non_express_gene_seq_list = [(sample_non_express_gene_seq_list[gene_idx], global_gene_positions[gene_id]) for gene_idx, gene_id in enumerate(sample_non_express_gene_id_list)]
                    new_sample_non_express_gene_express_list = [('[E_NON]', global_gene_positions[gene_id]) for gene_id in sample_non_express_gene_id_list]
                    new_sample_gene_id_list += new_sample_non_express_gene_id_list
                    new_sample_gene_seq_list += new_sample_non_express_gene_seq_list
                    new_sample_gene_express_list += new_sample_non_express_gene_express_list
                sample_gene_id_list = sorted(new_sample_gene_id_list, key=lambda x: x[1])
                sample_gene_id_list = [x[0] for x in sample_gene_id_list]
                if new_sample_gene_seq_list  and len(new_sample_gene_seq_list) > 0:
                    sample_gene_seq_list = sorted(new_sample_gene_seq_list, key=lambda x: x[1])
                    sample_gene_seq_list = [x[0] for x in sample_gene_seq_list]
                else:
                    sample_gene_seq_list = []
                sample_gene_express_list = sorted(new_sample_gene_express_list, key=lambda x: x[1])
                sample_gene_express_list = [x[0] for x in sample_gene_express_list]
                input_gene_list_len = len(sample_gene_id_list)
                if lucacell_args.embedding_complete:
                    truncation_gene_seq_length = input_gene_list_len
                else:
                    truncation_gene_seq_length = min(input_gene_list_len, lucacell_args.truncation_gene_seq_length)
                while True:
                    # 设置了一次性推理长度
                    if lucacell_args.embedding_fixed_len_a_time_for_cell and lucacell_args.embedding_fixed_len_a_time_for_cell > 0:
                        emb, attns, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding(
                            lucaone_args,
                            global_model_dirpath,
                            [sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_list],
                            embedding_type="matrix",
                            repr_layers=[-1],
                            truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                            truncation_gene_seq_length=lucacell_args.embedding_fixed_len_a_time_for_cell,
                            device=lucacell_args.device,
                            matrix_add_special_token=matrix_add_special_token,
                            save_type=lucacell_args.save_type,
                            need_head_weights=need_head_weights,
                            gene_seq_emb_save_path=gene_seq_emb_save_path,
                            use_bf16=lucacell_args.use_bf16
                        )
                        # 如果指定的设备运行失败，则使用CPU
                        use_cpu = False
                        if emb is None:
                            emb, attns, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding(
                                lucaone_args,
                                global_model_dirpath,
                                [sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_list],
                                embedding_type="matrix",
                                repr_layers=[-1],
                                truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                                truncation_gene_seq_length=lucacell_args.embedding_fixed_len_a_time_for_cell,
                                device=torch.device("cpu"),
                                matrix_add_special_token=matrix_add_special_token,
                                save_type=lucacell_args.save_type,
                                need_head_weights=need_head_weights,
                                gene_seq_emb_save_path=gene_seq_emb_save_path,
                                use_bf16=lucacell_args.use_bf16
                            )
                            use_cpu = True
                        if emb is not None and input_gene_list_len > lucacell_args.embedding_fixed_len_a_time_for_cell:
                            emb, attns = complete_embedding_matrix(
                                lucaone_args,
                                lucacell_args,
                                sample_id,
                                sample_type,
                                sample_gene_id_list,
                                sample_gene_seq_list,
                                sample_gene_express_list,
                                truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                                truncation_gene_seq_length=lucacell_args.embedding_fixed_len_a_time_for_cell,
                                init_emb=emb,
                                init_attns=attns,
                                embedding_type="matrix",
                                matrix_add_special_token=matrix_add_special_token,
                                save_type=lucacell_args.save_type,
                                gene_seq_emb_save_path=gene_seq_emb_save_path,
                                use_cpu=use_cpu,
                                use_bf16=lucacell_args.use_bf16
                            )
                        if use_cpu:
                            print("use_cpu: %r" % use_cpu)
                    else:
                        emb, attns, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding(
                            lucaone_args,
                            global_model_dirpath,
                            [sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_list],
                            embedding_type="matrix",
                            repr_layers=[-1],
                            truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                            truncation_gene_seq_length=truncation_gene_seq_length,
                            device=lucacell_args.device,
                            matrix_add_special_token=matrix_add_special_token,
                            save_type=lucacell_args.save_type,
                            need_head_weights=need_head_weights,
                            gene_seq_emb_save_path=gene_seq_emb_save_path,
                            use_bf16=lucacell_args.use_bf16
                        )
                        use_cpu = False
                        if emb is None:
                            emb, attns, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding(
                                lucaone_args,
                                global_model_dirpath,
                                [sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_list],
                                embedding_type="matrix",
                                repr_layers=[-1],
                                truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                                truncation_gene_seq_length=truncation_gene_seq_length,
                                device=torch.device("cpu"),
                                matrix_add_special_token=matrix_add_special_token,
                                save_type=lucacell_args.save_type,
                                need_head_weights=need_head_weights,
                                gene_seq_emb_save_path=gene_seq_emb_save_path,
                                use_bf16=lucacell_args.use_bf16
                            )
                            use_cpu = True
                        # embedding完全长
                        if emb is not None and input_gene_list_len > truncation_gene_seq_length:
                            emb, attns = complete_embedding_matrix(
                                lucaone_args,
                                lucacell_args,
                                sample_id,
                                sample_type,
                                sample_gene_id_list,
                                sample_gene_seq_list,
                                sample_gene_express_list,
                                truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                                truncation_gene_seq_length=truncation_gene_seq_length,
                                init_emb=emb,
                                init_attns=attns,
                                embedding_type="matrix",
                                matrix_add_special_token=matrix_add_special_token,
                                save_type=lucacell_args.save_type,
                                gene_seq_emb_save_path=gene_seq_emb_save_path,
                                use_cpu=use_cpu,
                                use_bf16=lucacell_args.use_bf16
                            )
                        if use_cpu:
                            print("use_cpu: %r" % use_cpu)
                    if emb is not None:
                        if embedding_type == "vector":
                            emb = matrix_2_vector(emb, matrix_add_special_token, vector_type, lucacell_args.save_type)
                        torch.save(emb, embedding_filepath)
                        if need_head_weights:
                            torch.save(attns, attn_filepath)
                        break
                    print("%s embedding error, max_len from %d truncate to %d" % (
                        sample_id,
                        truncation_gene_seq_length,
                        int(truncation_gene_seq_length * 0.95)
                    ))
                    truncation_gene_seq_length = int(truncation_gene_seq_length * 0.95)
                torch.cuda.empty_cache()
            else:
                print("%s exists." % embedding_filepath)
            done += 1
            if done % 1000 == 0:
                print("Embedding done: %d" % done)
        print("Embedding over, done: %d" % done)
    elif lucacell_args.sample_gene_id_list and lucacell_args.sample_gene_seq_list and lucacell_args.sample_gene_express_list:
        print("Input gene express list length: %d" % len(lucacell_args.sample_gene_express_list))
        if lucacell_args.sample_id is None:
            lucacell_args.sample_id = "Unknown"
        lucacell_args.sample_gene_id_list = lucacell_args.sample_gene_id_list.upper().split(",")
        lucacell_args.sample_gene_seq_list = lucacell_args.sample_gene_seq_list.upper().split(",")
        lucacell_args.sample_gene_express_list = lucacell_args.sample_gene_express_list.upper().split(",")
        input_gene_list_len = len(lucacell_args.sample_gene_express_list)
        if lucacell_args.embedding_complete:
            truncation_gene_seq_length = input_gene_list_len
        else:
            truncation_gene_seq_length = min(input_gene_list_len, lucacell_args.truncation_gene_seq_length)
        while True:
            # 设置了一次性推理长度
            if lucacell_args.embedding_fixed_len_a_time_for_cell and lucacell_args.embedding_fixed_len_a_time_for_cell > 0:
                emb, attns, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding(
                    lucaone_args,
                    global_model_dirpath,
                    [lucacell_args.sample_id, lucacell_args.sample_type, lucacell_args.sample_gene_id_list, lucacell_args.sample_gene_seq_list, lucacell_args.sample_gene_express_list],
                    embedding_type="matrix",
                    repr_layers=[-1],
                    truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                    truncation_gene_seq_length=lucacell_args.embedding_fixed_len_a_time_for_cell,
                    device=lucacell_args.device,
                    matrix_add_special_token=matrix_add_special_token,
                    save_type=lucacell_args.save_type,
                    need_head_weights=need_head_weights,
                    gene_seq_emb_save_path=gene_seq_emb_save_path,
                    use_bf16=lucacell_args.use_bf16
                )
                use_cpu = False
                if emb is None:
                    emb, attns, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding(
                        lucaone_args,
                        global_model_dirpath,
                        [lucacell_args.sample_id, lucacell_args.sample_type, lucacell_args.sample_gene_id_list, lucacell_args.sample_gene_seq_list, lucacell_args.sample_gene_express_list],
                        embedding_type="matrix",
                        repr_layers=[-1],
                        truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                        truncation_gene_seq_length=lucacell_args.embedding_fixed_len_a_time_for_cell,
                        device=torch.device("cpu"),
                        matrix_add_special_token=matrix_add_special_token,
                        save_type=lucacell_args.save_type,
                        need_head_weights=need_head_weights,
                        gene_seq_emb_save_path=gene_seq_emb_save_path,
                        use_bf16=lucacell_args.use_bf16
                    )
                    use_cpu = True
                # embedding完全长
                if emb is not None and input_gene_list_len > lucacell_args.embedding_fixed_len_a_time_for_cell:
                    emb, attns = complete_embedding_matrix(
                        lucaone_args,
                        lucacell_args,
                        lucacell_args.sample_id,
                        lucacell_args.sample_type,
                        lucacell_args.sample_gene_id_list,
                        lucacell_args.sample_gene_seq_list,
                        lucacell_args.sample_gene_express_list,
                        truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                        truncation_gene_seq_length=lucacell_args.embedding_fixed_len_a_time_for_cell,
                        init_emb=emb,
                        init_attns=attns,
                        embedding_type="matrix",
                        matrix_add_special_token=matrix_add_special_token,
                        save_type=lucacell_args.save_type,
                        gene_seq_emb_save_path=gene_seq_emb_save_path,
                        use_cpu=use_cpu
                    )
                if use_cpu:
                    print("use_cpu: %r" % use_cpu)
            else:
                emb, attns, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding(
                    lucaone_args,
                    global_model_dirpath,
                    [lucacell_args.sample_id, lucacell_args.sample_type, lucacell_args.sample_gene_id_list, lucacell_args.sample_gene_seq_list, lucacell_args.sample_gene_express_list],
                    embedding_type="matrix",
                    repr_layers=[-1],
                    truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                    truncation_gene_seq_length=truncation_gene_seq_length,
                    device=lucacell_args.device,
                    matrix_add_special_token=matrix_add_special_token,
                    save_type=lucacell_args.save_type,
                    need_head_weights=need_head_weights,
                    gene_seq_emb_save_path=gene_seq_emb_save_path,
                    use_bf16=lucacell_args.use_bf16
                )
                use_cpu = False
                if emb is None:
                    emb, attns, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding(
                        lucaone_args,
                        global_model_dirpath,
                        [lucacell_args.sample_id, lucacell_args.sample_type, lucacell_args.sample_gene_id_list, lucacell_args.sample_gene_seq_list, lucacell_args.sample_gene_express_list],
                        embedding_type="matrix",
                        repr_layers=[-1],
                        truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                        truncation_gene_seq_length=truncation_gene_seq_length,
                        device=torch.device("cpu"),
                        matrix_add_special_token=matrix_add_special_token,
                        save_type=lucacell_args.save_type,
                        need_head_weights=need_head_weights,
                        gene_seq_emb_save_path=gene_seq_emb_save_path,
                        use_bf16=lucacell_args.use_bf16
                    )
                    use_cpu = True
                # embedding完全长
                if emb is not None and input_gene_list_len > truncation_gene_seq_length:
                    emb, attns = complete_embedding_matrix(
                        lucaone_args,
                        lucacell_args,
                        lucacell_args.sample_id,
                        lucacell_args.sample_type,
                        lucacell_args.sample_gene_id_list,
                        lucacell_args.sample_gene_seq_list,
                        lucacell_args.sample_gene_express_list,
                        truncation_nucleotide_seq_length=lucacell_args.truncation_nucleotide_seq_length,
                        truncation_gene_seq_length=truncation_gene_seq_length,
                        init_emb=emb,
                        init_attns=attns,
                        embedding_type="matrix",
                        matrix_add_special_token=matrix_add_special_token,
                        save_type=lucacell_args.save_type,
                        gene_seq_emb_save_path=gene_seq_emb_save_path,
                        use_cpu=use_cpu
                    )
                if use_cpu:
                    print("use_cpu: %r" % use_cpu)
            if emb is not None:
                print("processed seq length: %d, truncation seq length: %d" % (processed_gene_seq_len, truncation_gene_seq_length))
                break
            print("%s embedding error, max_len from %d truncate to %d" % (
                lucacell_args.sample_id,
                truncation_gene_seq_length,
                int(truncation_gene_seq_length * 0.95)
            ))
            truncation_gene_seq_length = int(truncation_gene_seq_length * 0.95)
        print("Sample: %s" % lucacell_args.sample_id)
        print("Gene: %s" % lucacell_args.sample_gene_seq_list)
        print("Express: %s" % lucacell_args.sample_gene_express_list)
        print("Embedding: ")
        if embedding_type == "vector":
            emb = matrix_2_vector(emb, matrix_add_special_token, vector_type, lucacell_args.save_type)
        print(emb)
    else:
        raise Exception("Input error, please --help.")


if __name__ == "__main__":
    run_args = get_args()
    main(run_args)
