#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/21 16:10
@project: LucaOnePlusTasks
@file: encoder
@desc: encoder for LucaOnePlusTasks
'''
import os
import torch
import sys
import numpy as np
sys.path.append(".")
sys.path.append("../")
sys.path.append("../src")
try:
    from args import Args
    from file_operator import fasta_reader
    from utils import calc_emb_filename_by_seq_id, calc_emb_filename_by_sample_id, matrix_2_vector
    from llm.lucaone.get_embedding import predict_embedding as predict_embedding_lucaone
    from llm.lucacell.get_embedding import predict_embedding as predict_embedding_lucacell
    from llm.lucacell.get_embedding import complete_embedding_matrix as complete_embedding_matrix_for_cell
except ImportError as e:
    from src.args import Args
    from src.file_operator import fasta_reader
    from src.utils import calc_emb_filename_by_seq_id, calc_emb_filename_by_sample_id, matrix_2_vector
    from src.llm.lucaone.get_embedding import predict_embedding as predict_embedding_lucaone
    from src.llm.lucacell.get_embedding import predict_embedding as predict_embedding_lucacell
    from src.llm.lucacell.get_embedding import complete_embedding_matrix as complete_embedding_matrix_for_cell


EMBEDDING_MAX_SEQ_LENGTH = 10240
EMBEDDING_MAX_CELL_LENGTH = 10240


def complete_embedding_matrix_for_seq(
        seq_id,
        seq_type,
        seq,
        truncation_seq_length,
        init_emb,
        llm_dirpath,
        trunc_type,
        embedding_type,
        matrix_add_special_token,
        embedding_complete,
        embedding_complete_seg_overlap,
        predict_embedding_func,
        device,
        use_cpu=False,
        save_type="numpy"
):
    if init_emb is not None and embedding_complete and ("representations" in embedding_type or "matrix" in embedding_type):
        torch.cuda.empty_cache()
        ori_seq_len = len(seq)
        # 每次能处理这么长度
        # print("init_emb:", init_emb.shape)
        cur_segment_len = init_emb.shape[0]
        print("cur_segment_len: %d, ori_seq_len: %d" %(cur_segment_len, ori_seq_len))
        if matrix_add_special_token:
            first_emb = init_emb[1:cur_segment_len - 1]
        else:
            first_emb = init_emb
        if matrix_add_special_token:
            cur_segment_len = cur_segment_len - 2
        # print("cur_segment_len: %d" % cur_segment_len)
        init_cur_segment_len = cur_segment_len
        segment_num = int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)
        if segment_num <= 1:
            return init_emb
        append_emb = None
        if embedding_complete_seg_overlap:
            sliding_window = init_cur_segment_len // 2
            print("Embedding Complete Seg Overlap: %r, ori seq len: %d, segment len: %d, init sliding windown: %d" % (
                embedding_complete_seg_overlap, ori_seq_len, init_cur_segment_len, sliding_window
            ))
            while True:
                print("updated window: %d" % sliding_window)
                try:
                    # 第一个已经处理，滑动窗口
                    if trunc_type == "right":
                        last_end = init_cur_segment_len
                        seg_idx = 0
                        for pos_idx in range(init_cur_segment_len, ori_seq_len - sliding_window, sliding_window):
                            seg_idx += 1
                            last_end = min(pos_idx + sliding_window, ori_seq_len)
                            seg_seq = seq[pos_idx - sliding_window:last_end]
                            print("segment idx: %d, seg seq len: %d" % (seg_idx, len(seg_seq)))
                            if predict_embedding_func == predict_embedding_lucaone:
                                seg_emb, seg_processed_seq_len = predict_embedding_func(
                                    llm_dirpath,
                                    sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                    trunc_type=trunc_type,
                                    embedding_type=embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=truncation_seq_length,
                                    matrix_add_special_token=False,
                                    device=device if not use_cpu else torch.device("cpu"),
                                    save_type=save_type
                                )
                            else:
                                seg_emb, seg_processed_seq_len = predict_embedding_func(
                                    llm_dirpath,
                                    [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                    trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=truncation_seq_length,
                                    matrix_add_special_token=False,
                                    device=device if not use_cpu else torch.device("cpu"),
                                    save_type=save_type
                                )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[sliding_window:]
                            else:
                                if save_type == "numpy":
                                    append_emb = np.concatenate((append_emb, seg_emb[sliding_window:]), axis=0)
                                else:
                                    append_emb = torch.cat((append_emb, seg_emb[sliding_window:]), dim=0)
                        if last_end < ori_seq_len:
                            seg_idx += 1
                            remain = ori_seq_len - last_end
                            seg_seq = seq[ori_seq_len - 2 * sliding_window:ori_seq_len]
                            seg_emb, seg_processed_seq_len = predict_embedding_func(
                                llm_dirpath,
                                [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=False,
                                device=device if not use_cpu else torch.device("cpu"),
                                save_type=save_type
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[-remain:]
                            else:
                                if save_type == "numpy":
                                    append_emb = np.concatenate((append_emb, seg_emb[-remain:]), axis=0)
                                else:
                                    append_emb = torch.cat((append_emb, seg_emb[-remain:]), dim=0)
                    else:
                        last_start = -init_cur_segment_len
                        seg_idx = 0
                        for pos_idx in range(-init_cur_segment_len, -ori_seq_len + sliding_window, -sliding_window):
                            seg_idx += 1
                            last_start = max(pos_idx - sliding_window, -ori_seq_len)
                            seg_seq = seq[last_start: pos_idx + sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding_func(
                                llm_dirpath,
                                [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=False,
                                device=device if not use_cpu else torch.device("cpu"),
                                save_type=save_type
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[:sliding_window]
                            else:
                                if save_type == "numpy":
                                    append_emb = np.concatenate((seg_emb[:sliding_window], append_emb), axis=0)
                                else:
                                    append_emb = torch.cat((seg_emb[:sliding_window], append_emb), dim=0)
                        if last_start > -ori_seq_len:
                            seg_idx += 1
                            remain = last_start + ori_seq_len
                            seg_seq = seq[-ori_seq_len:-ori_seq_len + 2 * sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding_func(
                                llm_dirpath,
                                [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=False,
                                device=device if not use_cpu else torch.device("cpu"),
                                save_type=save_type
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[:remain]
                            else:
                                if save_type == "numpy":
                                    append_emb = np.concatenate((seg_emb[:remain], append_emb), axis=0)
                                else:
                                    append_emb = torch.cat((seg_emb[:remain], append_emb), dim=0)
                except Exception as e:
                    append_emb = None
                if append_emb is not None:
                    break
                print("fail, change sliding window: %d -> %d" % (sliding_window, int(sliding_window * 0.95)))
                sliding_window = int(sliding_window * 0.95)
        else:
            while True:
                print("ori seq len: %d, segment len: %d" % (ori_seq_len, cur_segment_len))
                try:
                    # 第一个已经处理，最后一个单独处理（需要向左/向右扩充至cur_segment_len长度）
                    if trunc_type == "right":
                        begin_seq_idx = 0
                    else:
                        begin_seq_idx = ori_seq_len - (segment_num - 1) * cur_segment_len
                    for seg_idx in range(1, segment_num - 1):
                        seg_seq = seq[begin_seq_idx + seg_idx * cur_segment_len: begin_seq_idx + (seg_idx + 1) * cur_segment_len]
                        # print("segment idx: %d, seg_seq(%d): %s" % (seg_idx, len(seg_seq), seg_seq))
                        print("segment idx: %d, seg seq len: %d" % (seg_idx, len(seg_seq)))
                        seg_emb, seg_processed_seq_len = predict_embedding_func(
                            llm_dirpath,
                            [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                            trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=False,
                            device=device if not use_cpu else torch.device("cpu"),
                            save_type=save_type
                        )
                        if append_emb is None:
                            append_emb = seg_emb
                        else:
                            if save_type == "numpy":
                                append_emb = np.concatenate((append_emb, seg_emb), axis=0)
                            else:
                                append_emb = torch.cat((append_emb, seg_emb), dim=0)
                    if trunc_type == "right":
                        # 处理最后一个
                        last_seg_seq = seq[-cur_segment_len:]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("last seg seq: %s" % last_seg_seq)
                        print("last seg seq len: %d, really len: %d" % (len(last_seg_seq), really_len))
                        last_seg_emb, last_seg_processed_seq_len = predict_embedding_func(
                            llm_dirpath,
                            [seq_id + "_seg_%d" % (segment_num - 1), seq_type, last_seg_seq],
                            trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=False,
                            device=device if not use_cpu else torch.device("cpu"),
                            save_type=save_type
                        )
                        last_seg_emb = last_seg_emb[-really_len:, :]
                        if save_type == "numpy":
                            append_emb = np.concatenate((append_emb, last_seg_emb), axis=0)
                        else:
                            append_emb = torch.cat((append_emb, last_seg_emb), dim=0)
                    else:
                        # 处理第一个
                        first_seg_seq = seq[:cur_segment_len]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("first seg seq: %s" % first_seg_seq)
                        print("first seg seq len: %d, really len: %d" % (len(first_seg_seq), really_len))
                        first_seg_emb, first_seg_processed_seq_len = predict_embedding_func(
                            llm_dirpath,
                            [seq_id + "_seg_0", seq_type, first_seg_seq],
                            trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=False,
                            device=device if not use_cpu else torch.device("cpu"),
                            save_type=save_type
                        )
                        first_seg_emb = first_seg_emb[:really_len, :]
                        if save_type == "numpy":
                            append_emb = np.concatenate((first_seg_emb, append_emb), axis=0)
                        else:
                            append_emb = torch.cat((first_seg_emb, append_emb), dim=0)
                except Exception as e:
                    append_emb = None
                if append_emb is not None:
                    break
                print("fail, change segment len: %d -> %d, change seg num: %d -> %d" % (
                    cur_segment_len,
                    int(cur_segment_len * 0.95), segment_num,
                    int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)
                ))
                cur_segment_len = int(cur_segment_len * 0.95)
                segment_num = int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)
            append_emb = append_emb[init_cur_segment_len - cur_segment_len:]
        if trunc_type == "right":
            if save_type == "numpy":
                complete_emb = np.concatenate((first_emb, append_emb), axis=0)
            else:
                complete_emb = torch.cat((first_emb, append_emb), dim=0)
        else:
            if save_type == "numpy":
                complete_emb = np.concatenate((append_emb, first_emb), axis=0)
            else:
                complete_emb = torch.cat((append_emb, first_emb), dim=0)
        print("seq len: %d, seq embedding matrix len: %d" % (ori_seq_len, complete_emb.shape[0] + (2 if matrix_add_special_token else 0)))
        print("-" * 50)
        assert complete_emb.shape[0] == ori_seq_len
        if matrix_add_special_token:
            if save_type == "numpy":
                complete_emb = np.concatenate((init_emb[0:1, :], complete_emb, init_emb[-1:, :]), axis=0)
            else:
                complete_emb = torch.cat((init_emb[0:1, :], complete_emb, init_emb[-1:, :]), dim=0)
        init_emb = complete_emb
    return init_emb


class Encoder(object):
    def __init__(
            self,
            input_type,
            trunc_type,
            seq_llm_dirpath,
            seq_llm_type,
            seq_tokenizer,
            seq_max_length,
            seq_prepend_bos,
            seq_append_eos,
            seq_matrix_add_special_token,
            seq_vector_dirpath,
            seq_matrix_dirpath,
            seq_embedding_vector_type,
            seq_embedding_fixed_len_a_time,
            seq_meta_fasta,
            cell_llm_dirpath,
            cell_llm_type,
            cell_tokenizer,
            cell_max_length,
            cell_prepend_bos,
            cell_append_eos,
            cell_matrix_add_special_token,
            cell_vector_dirpath,
            cell_matrix_dirpath,
            cell_embedding_vector_type,
            cell_embedding_fixed_len_a_time,
            **kwargs
    ):
        print("-" * 25 + "LucaCell Downstream Tasks Encoder" + "-" * 25)
        self.input_type = input_type
        self.trunc_type = trunc_type

        self.seq_llm_dirpath = seq_llm_dirpath
        self.seq_llm_type = seq_llm_type
        if "seq_llm_version" in kwargs and kwargs["seq_llm_version"]:
            self.seq_llm_version = kwargs["seq_llm_version"]
        if "seq_llm_step" in kwargs and kwargs["seq_llm_step"]:
            self.seq_llm_step = kwargs["seq_llm_step"]
        self.seq_tokenizer = seq_tokenizer
        self.seq_matrix_add_special_token = seq_matrix_add_special_token
        self.seq_embedding_vector_type = seq_embedding_vector_type
        self.seq_max_length = seq_max_length
        self.seq_prepend_bos = seq_prepend_bos
        self.seq_append_eos = seq_append_eos
        # vector
        if seq_vector_dirpath and "#" in seq_vector_dirpath:
            self.seq_vector_dirpath = list(seq_vector_dirpath.split("#"))
        elif seq_vector_dirpath:
            self.seq_vector_dirpath = [seq_vector_dirpath]
        else:
            self.seq_vector_dirpath = None
        # matrix
        if seq_matrix_dirpath and "#" in seq_matrix_dirpath:
            self.seq_matrix_dirpath = list(seq_matrix_dirpath.split("#"))
        elif seq_matrix_dirpath:
            self.seq_matrix_dirpath = [seq_matrix_dirpath]
        else:
            self.seq_matrix_dirpath = None
        self.seq_embedding_fixed_len_a_time = seq_embedding_fixed_len_a_time
        self.seq_meta_fasta = seq_meta_fasta
        self.gene_id_2_seq = {}
        if self.seq_meta_fasta:
            for row in fasta_reader(self.seq_meta_fasta):
                gene_id = row[0]
                if gene_id[0] == ">":
                    gene_id = gene_id[1:]
                gene_seq = row[1].strip().upper()
                self.gene_id_2_seq[gene_id] = gene_seq
            print("gene_id_2_seq: %d" % len(self.gene_id_2_seq))
        self.cell_llm_dirpath = cell_llm_dirpath
        self.cell_llm_type = cell_llm_type
        self.cell_tokenizer = cell_tokenizer
        self.cell_matrix_add_special_token = cell_matrix_add_special_token
        self.cell_embedding_vector_type = cell_embedding_vector_type
        self.cell_max_length = cell_max_length
        self.cell_prepend_bos = cell_prepend_bos
        self.cell_append_eos = cell_append_eos
        # vector
        if cell_vector_dirpath and "#" in cell_vector_dirpath:
            self.cell_vector_dirpath = list(cell_vector_dirpath.split("#"))
        elif cell_vector_dirpath:
            self.cell_vector_dirpath = [cell_vector_dirpath]
        else:
            self.cell_vector_dirpath = None
        # matrix
        if cell_matrix_dirpath and "#" in cell_matrix_dirpath:
            self.cell_matrix_dirpath = list(cell_matrix_dirpath.split("#"))
        elif cell_matrix_dirpath:
            self.cell_matrix_dirpath = [cell_matrix_dirpath]
        else:
            self.cell_matrix_dirpath = None
        self.cell_embedding_fixed_len_a_time = cell_embedding_fixed_len_a_time

        if self.seq_matrix_add_special_token:
            self.seq_prepend_bos = True
            self.seq_append_eos = True
        if self.cell_matrix_add_special_token:
            self.cell_prepend_bos = True
            self.cell_append_eos = True
        print("Encoder: seq_prepend_bos=%r, seq_append_eos=%r" % (self.seq_prepend_bos, self.seq_append_eos))
        print("Encoder: cell_prepend_bos=%r, cell_append_eos=%r" % (self.cell_prepend_bos, self.cell_append_eos))
        if "seq_embedding_complete" in kwargs and kwargs["seq_embedding_complete"]:
            self.seq_embedding_complete = kwargs["seq_embedding_complete"]
            print("Encoder: seq_embedding_complete=%r" % self.seq_embedding_complete)
        else:
            self.seq_embedding_complete = False
        if "seq_embedding_complete_seg_overlap" in kwargs and kwargs["seq_embedding_complete_seg_overlap"]:
            self.seq_embedding_complete_seg_overlap = kwargs["seq_embedding_complete_seg_overlap"]
            print("Encoder: seq_embedding_complete_seg_overlap=%r" % self.seq_embedding_complete_seg_overlap)
        else:
            self.seq_embedding_complete_seg_overlap = False
        if "seq_vector_embedding_exists" in kwargs and kwargs["seq_vector_embedding_exists"]:
            self.seq_vector_embedding_exists = kwargs["seq_vector_embedding_exists"]
        else:
            self.seq_vector_embedding_exists = False
        if "seq_matrix_embedding_exists" in kwargs and kwargs["seq_matrix_embedding_exists"]:
            self.seq_matrix_embedding_exists = kwargs["seq_matrix_embedding_exists"]
        else:
            self.seq_matrix_embedding_exists = False

        if "cell_embedding_complete" in kwargs and kwargs["cell_embedding_complete"]:
            self.cell_embedding_complete = kwargs["cell_embedding_complete"]
            print("Encoder: cell_embedding_complete=%r" % self.cell_embedding_complete)
        else:
            self.cell_embedding_complete = False
        if "cell_embedding_complete_seg_overlap" in kwargs and kwargs["cell_embedding_complete_seg_overlap"]:
            self.cell_embedding_complete_seg_overlap = kwargs["cell_embedding_complete_seg_overlap"]
            print("Encoder: cell_embedding_complete_seg_overlap=%r" % self.cell_embedding_complete_seg_overlap)
        else:
            self.cell_embedding_complete_seg_overlap = False
        if "cell_matrix_embedding_exists" in kwargs and kwargs["cell_matrix_embedding_exists"]:
            self.cell_matrix_embedding_exists = kwargs["cell_matrix_embedding_exists"]
        else:
            self.cell_matrix_embedding_exists = False
        if "not_frozen_gene_express_bin_embedding" in kwargs and kwargs["not_frozen_gene_express_bin_embedding"]:
            self.not_frozen_gene_express_bin_embedding = kwargs["not_frozen_gene_express_bin_embedding"]
        else:
            self.not_frozen_gene_express_bin_embedding = False
        if "lucacell_finetune" in kwargs and kwargs["lucacell_finetune"]:
            self.lucacell_finetune = kwargs["lucacell_finetune"]
        else:
            self.lucacell_finetune = False
        if "gpu_id" in kwargs:
            self.gpu_id = kwargs["gpu_id"]
        else:
            self.gpu_id = -1
        if self.gpu_id > -1 and torch.cuda.is_available():
            device = torch.device("cuda", self.gpu_id)
        else:
            device = torch.device("cpu")
        print("Encoder: device=", device)
        self.device = device
        self.gene_id_2_emb_filename = {}
        self.gene_seq_embedding_buffer = {}
        self.cell_id_2_emb_filename = {}
        self.cell_embedding_buffer = {}
        if "buffer_size" in kwargs:
            self.embedding_buffer_size = kwargs["buffer_size"]
        else:
            self.embedding_buffer_size = 0
        if "seq_buffer_size" in kwargs:
            self.seq_embedding_buffer_size = kwargs["seq_buffer_size"]
        else:
            self.seq_embedding_buffer_size = 0

        print("Encoder: seq_matrix_add_special_token=%r, "
              "seq_embedding_complete=%r, "
              "seq_embedding_complete_seg_overlap=%r, "
              "seq_embedding_fixed_len_a_time=%d, "
              "seq_vector_embedding_exists=%r, "
              "seq_matrix_embedding_exists=%r" %
              (
                  self.seq_matrix_add_special_token,
                  self.seq_embedding_complete,
                  self.seq_embedding_complete_seg_overlap,
                  self.seq_embedding_fixed_len_a_time if self.seq_embedding_fixed_len_a_time else -1,
                  self.seq_vector_embedding_exists,
                  self.seq_matrix_embedding_exists)
              )
        print("Encoder: cell_matrix_add_special_token=%r, "
              "cell_embedding_complete=%r, "
              "cell_embedding_complete_seg_overlap=%r, "
              "cell_embedding_fixed_len_a_time=%d, "
              "cell_matrix_embedding_exists=%r" %
              (
                  self.cell_matrix_add_special_token,
                  self.cell_embedding_complete,
                  self.cell_embedding_complete_seg_overlap,
                  self.cell_embedding_fixed_len_a_time if self.cell_embedding_fixed_len_a_time else -1,
                  self.cell_matrix_embedding_exists)
              )
        lucacell_args = Args()
        lucacell_args.embedding_complete = self.cell_embedding_complete
        lucacell_args.embedding_complete_seg_overlap = self.cell_embedding_complete_seg_overlap
        lucacell_args.trunc_type = self.trunc_type
        lucacell_args.device = self.device
        lucacell_args.need_head_weights = False
        self.lucacell_args = lucacell_args
        print("*" * 50)

    def put_gene_seq_embedding_into_buffer(self, gene_id, embedding_info):
        if self.seq_embedding_buffer_size > 0:
            if len(self.gene_seq_embedding_buffer) >= self.seq_embedding_buffer_size:
                self.gene_seq_embedding_buffer = {}
            self.gene_seq_embedding_buffer[gene_id] = embedding_info

    def put_cell_embedding_into_buffer(self, cell_id, embedding_info):
        if self.embedding_buffer_size > 0:
            if len(self.cell_embedding_buffer) >= self.embedding_buffer_size:
                self.cell_embedding_buffer = {}
            self.cell_embedding_buffer[cell_id] = embedding_info

    def __get_gene_seq_embedding__(self, seq_id, seq_type, seq, embedding_type, vector_type="mean"):
        embedding_info = None
        if seq_id in self.gene_seq_embedding_buffer:
            return self.gene_seq_embedding_buffer[seq_id]

        if seq_id in self.gene_id_2_emb_filename:
            emb_filename = self.gene_id_2_emb_filename[seq_id]
            try:
                dirpath_list = self.seq_vector_dirpath if embedding_type == "vector" else self.seq_matrix_dirpath
                for dirpath in dirpath_list:
                    emb_filepath = os.path.join(dirpath, emb_filename)
                    if os.path.exists(emb_filepath):
                        embedding_info = torch.load(emb_filepath, map_location="cpu", weights_only=True)
                        self.put_gene_seq_embedding_into_buffer(seq_id, embedding_info)
                        return embedding_info
            except Exception as e:
                print(e)
                embedding_info = None
        elif embedding_type == "vector" and self.seq_vector_dirpath is not None \
                or embedding_type == "matrix" and self.seq_matrix_dirpath is not None:
            emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
            try:
                dirpath_list = self.seq_vector_dirpath if embedding_type == "vector" else self.seq_matrix_dirpath
                for dirpath in dirpath_list:
                    emb_filepath = os.path.join(dirpath, emb_filename)
                    if os.path.exists(emb_filepath):
                        embedding_info = torch.load(emb_filepath, map_location="cpu", weights_only=True)
                        self.gene_id_2_emb_filename[seq_id] = emb_filename
                        self.put_gene_seq_embedding_into_buffer(seq_id, embedding_info)
                        return embedding_info
            except Exception as e:
                print(e)
                embedding_info = None

        if embedding_info is None:
            if self.seq_matrix_embedding_exists or self.seq_vector_embedding_exists:
                with open("seq_matrix_embedding_not_exists.txt", "a+") as wfp:
                    print("seq_id: %s" % seq_id)
                    wfp.write("seq_id: %s\n" % seq_id)
                    wfp.flush()

        if embedding_info is None:
            if self.seq_matrix_embedding_exists or self.seq_vector_embedding_exists:
                print("seq_id: %s 's seq embedding file not exists in advance" % seq_id)
                sys.exit(-1)
            # LucaOne for gene seq embedding
            if seq is None or len(seq) == 0:
                seq = self.gene_id_2_seq[seq_id]
            cur_seq_len = len(seq)
            if hasattr(self, "seq_embedding_complete") and self.seq_embedding_complete:
                truncation_seq_length = min(cur_seq_len, EMBEDDING_MAX_SEQ_LENGTH)
            else:
                truncation_seq_length = self.seq_max_length - int(self.seq_prepend_bos) - int(self.seq_append_eos)
                truncation_seq_length = min(cur_seq_len, truncation_seq_length)

            while True:
                # 设置了一次性推理长度
                if self.seq_embedding_fixed_len_a_time and self.seq_embedding_fixed_len_a_time > 0:
                    embedding_info, processed_seq_len = predict_embedding_lucaone(
                        self.seq_llm_dirpath,
                        [seq_id, seq_type, seq],
                        self.trunc_type,
                        "matrix",
                        repr_layers=[-1],
                        truncation_seq_length=self.seq_embedding_fixed_len_a_time,
                        matrix_add_special_token=self.seq_matrix_add_special_token,
                        device=self.device,
                        save_type="tensor"
                    )
                    use_cpu = False
                    if embedding_info is None:
                        embedding_info, processed_seq_len = predict_embedding_lucaone(
                            self.seq_llm_dirpath,
                            [seq_id, seq_type, seq],
                            self.trunc_type,
                            "matrix",
                            repr_layers=[-1],
                            truncation_seq_length=self.seq_embedding_fixed_len_a_time,
                            matrix_add_special_token=self.seq_matrix_add_special_token,
                            device=torch.device("cpu"),
                            save_type="tensor"
                        )
                        use_cpu = True
                    if embedding_info is not None and hasattr(self, "seq_embedding_complete") and self.seq_embedding_complete \
                            and cur_seq_len > self.seq_embedding_fixed_len_a_time:
                        embedding_info = complete_embedding_matrix_for_seq(
                            seq_id,
                            seq_type,
                            seq,
                            self.seq_embedding_fixed_len_a_time,
                            embedding_info,
                            self.seq_llm_dirpath,
                            self.trunc_type,
                            "matrix",
                            self.seq_matrix_add_special_token,
                            self.seq_embedding_complete,
                            self.seq_embedding_complete_seg_overlap,
                            device=self.device,
                            predict_embedding_func=predict_embedding_lucaone,
                            use_cpu=use_cpu,
                            save_type="tensor"
                        )
                else:
                    embedding_info, processed_seq_len = predict_embedding_lucaone(
                        self.seq_llm_dirpath,
                        [seq_id, seq_type, seq],
                        self.trunc_type,
                        "matrix",
                        repr_layers=[-1],
                        truncation_seq_length=truncation_seq_length,
                        matrix_add_special_token=self.seq_matrix_add_special_token,
                        device=self.device,
                        save_type="tensor"
                    )
                    use_cpu = False
                    if embedding_info is None:
                        embedding_info, processed_seq_len = predict_embedding_lucaone(
                            self.seq_llm_dirpath,
                            [seq_id, seq_type, seq],
                            self.trunc_type,
                            "matrix",
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=self.seq_matrix_add_special_token,
                            device=torch.device("cpu"),
                            save_type="tensor"
                        )
                        use_cpu = True
                    if embedding_info is not None and hasattr(self, "seq_embedding_complete") and self.seq_embedding_complete \
                            and cur_seq_len > truncation_seq_length:
                        embedding_info = complete_embedding_matrix_for_seq(
                            seq_id,
                            seq_type,
                            seq,
                            truncation_seq_length,
                            embedding_info,
                            self.seq_llm_dirpath,
                            self.trunc_type,
                            "matrix",
                            self.seq_matrix_add_special_token,
                            self.seq_embedding_complete,
                            self.seq_embedding_complete_seg_overlap,
                            device=self.device,
                            predict_embedding_func=predict_embedding_lucaone,
                            use_cpu=use_cpu,
                            save_type="tensor"
                        )
                if use_cpu:
                    print("use_cpu: %r" % use_cpu)
                if embedding_info is not None:
                    if embedding_type == "vector":
                        embedding_info = matrix_2_vector(embedding_info, self.seq_matrix_add_special_token, vector_type, "tensor")
                    break
                truncation_seq_length = (truncation_seq_length + int(self.seq_prepend_bos) + int(self.seq_append_eos)) * 0.95 \
                                        - int(self.seq_prepend_bos) - int(self.seq_append_eos)
                truncation_seq_length = int(truncation_seq_length)
                print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
            if embedding_type == "vector" and self.seq_vector_dirpath is not None \
                    or embedding_type == "matrix" and self.seq_matrix_dirpath is not None:
                emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
                dirpath_list = self.seq_vector_dirpath if embedding_type == "vector" else self.seq_matrix_dirpath
                dirpath = dirpath_list[0]
                emb_filepath = os.path.join(dirpath, emb_filename)
                # print("seq_len: %d" % len(seq))
                # print("emb shape:", embedding_info.shape)
                torch.save(embedding_info, emb_filepath)
                self.gene_id_2_emb_filename[seq_id] = emb_filename
                self.put_gene_seq_embedding_into_buffer(seq_id, embedding_info)
        return embedding_info

    def encode_single_gene_seq(
            self,
            seq_id,
            seq_type,
            seq,
            embedding_type,
            vector_filename=None,
            matrix_filename=None,
            label=None
    ):
        seq_type = seq_type.strip().lower()
        # for embedding vector
        vector, matrix = None, None
        if embedding_type == "vector":
            if vector_filename is None:
                if seq is None:
                    raise Exception("seq is none and vector_filename is none")
                elif seq_type == "molecule":
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    vector = self.__get_gene_seq_embedding__(seq_id, seq_type, seq, "vector", self.seq_embedding_vector_type)
            elif isinstance(vector_filename, str):
                for vector_dir in self.seq_vector_dirpath:
                    vector_filepath = os.path.join(vector_dir, vector_filename)
                    if os.path.exists(vector_filepath):
                        vector = torch.load(vector_filepath, map_location="cpu", weights_only=True)
                        break
            elif isinstance(vector_filename, np.ndarray):
                vector = vector_filename
            else:
                raise Exception("vector is not filepath-str and np.ndarray")

        else:
            if matrix_filename is None:
                if seq is None:
                    raise Exception("seq is none and matrix_filename is none")
                elif seq_type == "molecule":
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    matrix = self.__get_gene_seq_embedding__(seq_id, seq_type, seq, "matrix")
            elif isinstance(matrix_filename, str):
                for matrix_dir in self.seq_matrix_dirpath:
                    matrix_filepath = os.path.join(matrix_dir, matrix_filename)
                    if os.path.exists(matrix_filepath):
                        matrix = torch.load(matrix_filepath, map_location="cpu", weights_only=True)
                        break
            elif isinstance(matrix_filename, np.ndarray):
                matrix = matrix_filename
            else:
                raise Exception("matrix is not filepath-str and np.ndarray")

        seq = seq.upper()
        if seq_type and "rna" in seq_type:
            seq = seq.replace("T", "U")
        return {
            "seq_id": seq_id,
            "seq": seq,
            "seq_type": seq_type,
            "vector": vector,
            "matrix": matrix,
            "label": label
        }

    def encode_pair_gene_seq(
            self,
            seq_id_a,
            seq_id_b,
            seq_type_a,
            seq_type_b,
            seq_a,
            seq_b,
            embedding_type_a,
            embedding_type_b,
            vector_filename_a=None,
            vector_filename_b=None,
            matrix_filename_a=None,
            matrix_filename_b=None,
            label=None
    ):
        seq_type_a = seq_type_a.strip().lower()
        seq_type_b = seq_type_b.strip().lower()
        # for embedding vector
        vector_a, matrix_a = None, None
        if embedding_type_a == "vector":
            if vector_filename_a is None:
                if seq_a is None:
                    raise Exception("seq_a is none and vector_filename_a is none")
                elif seq_type_a == "molecule":
                    raise Exception("now not support embedding of the seq_type_a=%s" % seq_type_a)
                else:
                    vector_a = self.__get_gene_seq_embedding__(seq_id_a, seq_type_a, seq_a, "vector", self.seq_embedding_vector_type)
            elif isinstance(vector_filename_a, str):
                for vector_dir in self.seq_vector_dirpath:
                    vector_filepath_a = os.path.join(vector_dir, vector_filename_a)
                    if os.path.exists(vector_filepath_a):
                        vector_a = torch.load(vector_filepath_a, map_location="cpu", weights_only=True)
                        break
            elif isinstance(vector_filename_a, np.ndarray):
                vector_a = vector_filename_a
            else:
                raise Exception("vector_a is not filepath-str and np.ndarray")
        else:
            if matrix_filename_a is None:
                if seq_a is None:
                    raise Exception("seq_a is none and matrix_filename_a is none")
                else:
                    matrix_a = self.__get_gene_seq_embedding__(seq_id_a, seq_type_a, seq_a, "matrix")
            elif isinstance(matrix_filename_a, str):
                for matrix_dir in self.seq_matrix_dirpath:
                    matrix_filepath_a = os.path.join(matrix_dir, matrix_filename_a)
                    if os.path.exists(matrix_filepath_a):
                        matrix_a = torch.load(matrix_filepath_a, map_location="cpu", weights_only=True)
                        break
            elif isinstance(matrix_filename_a, np.ndarray):
                matrix_a = matrix_filename_a
            else:
                raise Exception("matrix_a is not filepath-str and np.ndarray")

        # for embedding matrix
        vector_b, matrix_b = None, None
        if embedding_type_b == "vector":
            if vector_filename_b is None:
                if seq_b is None:
                    raise Exception("seq_b is none and vector_filename_b is none")
                elif seq_type_b == "molecule":
                    raise Exception("now not support embedding of the seq_type_b=%s" % seq_type_b)
                else:
                    vector_b = self.__get_gene_seq_embedding__(seq_id_b, seq_type_b, seq_b, "vector", self.seq_embedding_vector_type)
            elif isinstance(vector_filename_b, str):
                for vector_dir in self.seq_vector_dirpath:
                    vector_filepath_b = os.path.join(vector_dir, vector_filename_b)
                    if os.path.exists(vector_filepath_b):
                        vector_b = torch.load(vector_filepath_b, map_location="cpu", weights_only=True)
                        break
            elif isinstance(vector_filename_b, np.ndarray):
                vector_b = vector_filename_b
            else:
                raise Exception("vector_b is not filepath-str and np.ndarray")
        else:
            if matrix_filename_b is None:
                if seq_b is None:
                    raise Exception("seq_b is none and matrix_filename_b is none")
                else:
                    matrix_b = self.__get_gene_seq_embedding__(seq_id_b, seq_type_b, seq_b, "matrix")
            elif isinstance(matrix_filename_b, str):
                for matrix_dir in self.seq_matrix_dirpath:
                    matrix_filepath_b = os.path.join(matrix_dir, matrix_filename_b)
                    if os.path.exists(matrix_filepath_b):
                        matrix_b = torch.load(matrix_filepath_b, map_location="cpu", weights_only=True)
                        break
            elif isinstance(matrix_filename_b, np.ndarray):
                matrix_b = matrix_filename_b
            else:
                raise Exception("matrix_b is not filepath-str and np.ndarray")

        seq_a = seq_a.upper()
        if seq_type_a and "rna" in seq_type_a:
            seq_a = seq_a.replace("T", "U")

        seq_b = seq_b.upper()
        if seq_type_b and "rna" in seq_type_b:
            seq_b = seq_b.replace("T", "U")

        return {
            "seq_id_a": seq_id_a,
            "seq_a": seq_a,
            "seq_type_a": seq_type_a,
            "vector_a": vector_a,
            "matrix_a": matrix_a,
            "seq_id_b": seq_id_b,
            "seq_b": seq_b,
            "seq_type_b": seq_type_b,
            "vector_b": vector_b,
            "matrix_b": matrix_b,
            "label": label
        }

    def __get_cell_embedding__(
            self,
            sample_id,
            sample_type,
            sample_gene_id_list,
            sample_gene_seq_type_list,
            sample_gene_seq_list,
            sample_gene_express_bin_list,
            embedding_type
    ):
        if sample_id in self.cell_embedding_buffer:
            return self.cell_embedding_buffer[sample_id]

        embedding_info = None
        if sample_id in self.cell_id_2_emb_filename:
            emb_filename = self.cell_id_2_emb_filename[sample_id]
            try:
                dirpath_list = self.cell_vector_dirpath if embedding_type == "vector" else self.cell_matrix_dirpath
                for dirpath in dirpath_list:
                    emb_filepath = os.path.join(dirpath, emb_filename)
                    if os.path.exists(emb_filepath):
                        embedding_info = torch.load(emb_filepath, map_location="cpu", weights_only=True)
                        self.put_cell_embedding_into_buffer(sample_id, embedding_info)
                        return embedding_info
            except Exception as e:
                print(e)
                embedding_info = None
        elif embedding_type == "vector" and self.cell_vector_dirpath is not None \
                or embedding_type == "matrix" and self.cell_matrix_dirpath is not None:
            emb_filename = calc_emb_filename_by_sample_id(sample_id=sample_id, embedding_type=embedding_type)
            try:
                dirpath_list = self.cell_vector_dirpath if embedding_type == "vector" else self.cell_matrix_dirpath
                for dirpath in dirpath_list:
                    emb_filepath = os.path.join(dirpath, emb_filename)
                    if os.path.exists(emb_filepath):
                        embedding_info = torch.load(emb_filepath, map_location="cpu", weights_only=True)
                        self.cell_id_2_emb_filename[sample_id] = emb_filename
                        self.put_cell_embedding_into_buffer(sample_id, embedding_info)
                        return embedding_info
            except Exception as e:
                print(e)
                embedding_info = None

        if embedding_info is None:
            if self.cell_matrix_embedding_exists:
                with open("cell_matrix_embedding_not_exists.txt", "a+") as wfp:
                    print("sample_id: %s" % sample_id)
                    wfp.write("sample_id: %s\n" % sample_id)
                    wfp.flush()

        if embedding_info is None:
            if self.cell_matrix_embedding_exists:
                print("sample_id: %s 's cell embedding file not exists in advance" % sample_id)
                sys.exit(-1)
            # LucaOne for gene seq embedding
            cur_cell_len = len(sample_gene_express_bin_list)
            if hasattr(self, "cell_embedding_complete") and self.cell_embedding_complete:
                truncation_cell_length = min(cur_cell_len, EMBEDDING_MAX_CELL_LENGTH)
            else:
                truncation_cell_length = self.cell_max_length - int(self.cell_prepend_bos) - int(self.cell_append_eos)
                truncation_cell_length = min(cur_cell_len, truncation_cell_length)
            while True:
                # 设置了一次性推理长度
                if self.cell_embedding_fixed_len_a_time and self.cell_embedding_fixed_len_a_time > 0:
                    emb, _, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding_lucacell(
                        lucaone_args=None,
                        llm_dirpath=self.cell_llm_dirpath,
                        sample=[sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_bin_list],
                        embedding_type="matrix",
                        repr_layers=[-1],
                        truncation_nucleotide_seq_length=self.seq_max_length,
                        truncation_gene_seq_length=self.cell_embedding_fixed_len_a_time,
                        device=self.device,
                        matrix_add_special_token=self.cell_matrix_add_special_token,
                        save_type="tensor",
                        need_head_weights=False,
                        gene_seq_emb_save_path=self.seq_vector_dirpath
                    )
                    # 如果指定的设备运行失败，则使用CPU
                    use_cpu = False
                    if emb is None:
                        emb, _, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding_lucacell(
                            lucaone_args=None,
                            llm_dirpath=self.cell_llm_dirpath,
                            sample=[sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_bin_list],
                            embedding_type="matrix",
                            repr_layers=[-1],
                            truncation_nucleotide_seq_length=self.seq_max_length,
                            truncation_gene_seq_length=self.cell_embedding_fixed_len_a_time,
                            device=torch.device("cpu"),
                            matrix_add_special_token=self.cell_matrix_add_special_token,
                            save_type="tensor",
                            need_head_weights=False,
                            gene_seq_emb_save_path=self.seq_vector_dirpath
                        )
                        use_cpu = True
                    if emb is not None and cur_cell_len > self.cell_embedding_fixed_len_a_time:
                        emb, _ = complete_embedding_matrix_for_cell(
                            lucaone_args=None,
                            lucacell_args=self.lucacell_args,
                            sample_id=sample_id,
                            sample_type=sample_type,
                            sample_gene_id_list=sample_gene_id_list,
                            sample_gene_seq_list=sample_gene_seq_list,
                            sample_gene_express_bin_list=sample_gene_express_bin_list,
                            truncation_nucleotide_seq_length=self.seq_max_length,
                            truncation_gene_seq_length=self.cell_embedding_fixed_len_a_time,
                            init_emb=emb,
                            init_attns=None,
                            embedding_type="matrix",
                            matrix_add_special_token=self.cell_matrix_add_special_token,
                            save_type="tensor",
                            gene_seq_emb_save_path=self.seq_vector_dirpath,
                            use_cpu=use_cpu
                        )
                    if use_cpu:
                        print("use_cpu: %r" % use_cpu)
                else:
                    emb, _, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding_lucacell(
                        lucaone_args=None,
                        llm_dirpath=self.cell_llm_dirpath,
                        sample=[sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_bin_list],
                        embedding_type="matrix",
                        repr_layers=[-1],
                        truncation_nucleotide_seq_length=self.seq_max_length,
                        truncation_gene_seq_length=self.cell_max_length,
                        device=self.device,
                        matrix_add_special_token=self.cell_matrix_add_special_token,
                        save_type="tensor",
                        need_head_weights=False,
                        gene_seq_emb_save_path=self.seq_vector_dirpath
                    )
                    use_cpu = False
                    if emb is None:
                        emb, _, processed_nucleotide_seq_len, processed_gene_seq_len = predict_embedding_lucacell(
                            lucaone_args=None,
                            llm_dirpath=self.cell_llm_dirpath,
                            sample=[sample_id, sample_type, sample_gene_id_list, sample_gene_seq_list, sample_gene_express_bin_list],
                            embedding_type="matrix",
                            repr_layers=[-1],
                            truncation_nucleotide_seq_length=self.seq_max_length,
                            truncation_gene_seq_length=self.cell_max_length,
                            device=torch.device("cpu"),
                            matrix_add_special_token=self.cell_matrix_add_special_token,
                            save_type="tensor",
                            need_head_weights=False,
                            gene_seq_emb_save_path=self.seq_vector_dirpath
                        )
                        use_cpu = True
                    # embedding完全长
                    if emb is not None and cur_cell_len > truncation_cell_length:
                        emb, _ = complete_embedding_matrix_for_cell(
                            lucaone_args=None,
                            lucacell_args=self.lucacell_args,
                            sample_id=sample_id,
                            sample_type=sample_type,
                            sample_gene_id_list=sample_gene_id_list,
                            sample_gene_seq_list=sample_gene_seq_list,
                            sample_gene_express_bin_list=sample_gene_express_bin_list,
                            truncation_nucleotide_seq_length=self.seq_max_length,
                            truncation_gene_seq_length=truncation_cell_length,
                            init_emb=emb,
                            init_attns=None,
                            embedding_type="matrix",
                            matrix_add_special_token=self.cell_matrix_add_special_token,
                            save_type="tensor",
                            gene_seq_emb_save_path=self.seq_vector_dirpath,
                            use_cpu=use_cpu
                        )
                    if use_cpu:
                        print("use_cpu: %r" % use_cpu)
                if emb is not None:
                    if embedding_type == "vector":
                        embedding_info = matrix_2_vector(emb, self.cell_matrix_add_special_token, self.cell_embedding_vector_type, "tensor")
                    else:
                        embedding_info = emb
                    break
                truncation_cell_length = (truncation_cell_length + int(self.cell_prepend_bos) + int(self.cell_append_eos)) * 0.95 \
                                        - int(self.cell_prepend_bos) - int(self.cell_append_eos)
                truncation_cell_length = int(truncation_cell_length)
                print("%s embedding error, truncation_cell_length: %d->%d" % (sample_id, cur_cell_len, truncation_cell_length))
            if embedding_type == "vector" and self.cell_vector_dirpath is not None \
                    or embedding_type == "matrix" and self.cell_matrix_dirpath is not None:
                emb_filename = calc_emb_filename_by_sample_id(sample_id=sample_id, embedding_type=embedding_type)
                dirpath_list = self.cell_vector_dirpath if embedding_type == "vector" else self.cell_matrix_dirpath
                dirpath = dirpath_list[0]
                emb_filepath = os.path.join(dirpath, emb_filename)
                # print("cell_len: %d" % len(seq))
                # print("emb shape:", embedding_info.shape)
                torch.save(embedding_info, emb_filepath)
                self.cell_id_2_emb_filename[sample_id] = emb_filename
                self.put_cell_embedding_into_buffer(sample_id, embedding_info)
        return embedding_info

    def encode_single_cell(
                self,
                sample_id,
                sample_type,
                sample_gene_id_list,
                sample_gene_seq_type_list,
                sample_gene_seq_list,
                sample_gene_express_bin_list,
                vector_filename=None,
                matrix_filename=None,
                label=None
        ):
        if self.not_frozen_gene_express_bin_embedding or self.lucacell_finetune:
            sample_gene_seq_vector_list = []
            for gene_idx, gene_id in enumerate(sample_gene_id_list):
                sample_gene_seq_vector_list.append(
                    self.__get_gene_seq_embedding__(
                        gene_id,
                        "gene",
                        sample_gene_seq_list[gene_idx] if sample_gene_seq_list else None,
                        embedding_type="vector",
                        vector_type="mean"
                    )
                )
            return {
                "sample_id": sample_id,
                "sample_type": sample_type,
                "sample_gene_id_list": None,
                "sample_gene_seq_type_list": None,
                "sample_gene_seq_list": None,
                "sample_gene_express_bin_list": sample_gene_express_bin_list,
                "sample_gene_seq_vector_list": sample_gene_seq_vector_list,
                "sample_gene_seq_matrix_list": None,
                "sample_cell_vector": None,
                "sample_cell_matrix": None,
                "label": label
            }
        else:
            vector = None
            if "vector" in self.input_type:
                # for embedding vector
                if vector_filename is None:
                    vector = self.__get_cell_embedding__(
                        sample_id,
                        sample_type,
                        sample_gene_id_list,
                        sample_gene_seq_type_list,
                        sample_gene_seq_list,
                        sample_gene_express_bin_list,
                        "vector"
                    )
                elif isinstance(vector_filename, str):
                    for vector_dir in self.seq_vector_dirpath:
                        vector_filepath = os.path.join(vector_dir, vector_filename)
                        if os.path.exists(vector_filepath):
                            vector = torch.load(vector_filepath, map_location="cpu", weights_only=True)
                            break
                elif isinstance(vector_filename, np.ndarray) or isinstance(vector_filename, torch.Tensor):
                    vector = vector_filename
                else:
                    raise Exception("vector is not filepath-str and np.ndarray")
            matrix = None
            if "matrix" in self.input_type:
                # for embedding matrix
                if matrix_filename is None:
                    matrix = self.__get_cell_embedding__(
                        sample_id,
                        sample_type,
                        sample_gene_id_list,
                        sample_gene_seq_type_list,
                        sample_gene_seq_list,
                        sample_gene_express_bin_list,
                        "matrix"
                    )
                elif isinstance(matrix_filename, str):
                    for matrix_dir in self.seq_matrix_dirpath:
                        matrix_filepath = os.path.join(matrix_dir, matrix_filename)
                        if os.path.exists(matrix_filepath):
                            matrix = torch.load(matrix_filepath, map_location="cpu", weights_only=True)
                            break
                elif isinstance(matrix_filename, np.ndarray) or isinstance(matrix_filename, torch.Tensor):
                    matrix = matrix_filename
                else:
                    raise Exception("matrix is not filepath-str and np.ndarray")
            return {
                "sample_id": sample_id,
                "sample_type": sample_type,
                "sample_gene_id_list": None,
                "sample_gene_seq_type_list": None,
                "sample_gene_seq_list": None,
                "sample_gene_express_bin_list": None,
                "sample_gene_seq_vector_list": None,
                "sample_gene_seq_matrix_list": None,
                "sample_cell_vector": vector,
                "sample_cell_matrix": matrix,
                "label": label
            }

    def encode_pair_cell(
            self,
            sample_id_a,
            sample_id_b,
            sample_type_a,
            sample_type_b,
            sample_gene_id_list_a,
            sample_gene_id_list_b,
            sample_gene_seq_type_list_a,
            sample_gene_seq_type_list_b,
            sample_gene_seq_list_a,
            sample_gene_seq_list_b,
            sample_gene_express_bin_list_a,
            sample_gene_express_bin_list_b,
            vector_filename_a=None,
            matrix_filename_a=None,
            vector_filename_b=None,
            matrix_filename_b=None,
            label=None
    ):
        if self.not_frozen_gene_express_bin_embedding or self.lucacell_finetune:
            sample_gene_seq_vector_list_a = []
            for gene_idx, gene_id in enumerate(sample_gene_id_list_a):
                sample_gene_seq_vector_list_a.append(
                    self.__get_gene_seq_embedding__(
                        gene_id,
                        "gene",
                        sample_gene_seq_list_a[gene_idx] if sample_gene_seq_list_a else None,
                        embedding_type="vector",
                        vector_type="mean")
                )
            sample_gene_seq_vector_list_b = []
            for gene_idx, gene_id in enumerate(sample_gene_id_list_b):
                sample_gene_seq_vector_list_b.append(
                    self.__get_gene_seq_embedding__(
                        gene_id,
                        "gene",
                        sample_gene_seq_list_b[gene_idx] if sample_gene_seq_list_b else None,
                        embedding_type="vector",
                        vector_type="mean"
                    )
                )

            return {
                "sample_id_a": sample_id_a,
                "sample_type_a": sample_type_a,
                "sample_gene_id_list_a": None,
                "sample_gene_seq_type_list_a": None,
                "sample_gene_seq_list_a": None,
                "sample_gene_express_bin_list_a": sample_gene_express_bin_list_a,
                "sample_gene_seq_vector_list_a": sample_gene_seq_vector_list_a,
                "sample_gene_seq_matrix_list_a": None,
                "sample_cell_vector_a": None,
                "sample_cell_matrix_a": None,
                "sample_id_b": sample_id_b,
                "sample_type_b": sample_type_b,
                "sample_gene_id_list_b": None,
                "sample_gene_seq_type_list_b": None,
                "sample_gene_seq_list_b": None,
                "sample_gene_express_bin_list_b": sample_gene_express_bin_list_b,
                "sample_gene_seq_vector_list_b": sample_gene_seq_vector_list_b,
                "sample_gene_seq_matrix_list_b": None,
                "sample_cell_vector_b": None,
                "sample_cell_matrix_b": None,
                "label": label
            }
        else:
            # for embedding vector
            vector_a, vector_b = None, None
            if "vector" in self.input_type:
                if vector_filename_a is None:
                    vector_a = self.__get_cell_embedding__(
                        sample_id_a,
                        sample_type_a,
                        sample_gene_id_list_a,
                        sample_gene_seq_type_list_a,
                        sample_gene_seq_list_a,
                        sample_gene_express_bin_list_a,
                        "vector"
                    )
                elif isinstance(vector_filename_a, str):
                    for vector_dir in self.seq_vector_dirpath:
                        vector_filepath = os.path.join(vector_dir, vector_filename_a)
                        if os.path.exists(vector_filepath):
                            vector_a = torch.load(vector_filepath, map_location="cpu", weights_only=True)
                            break
                elif isinstance(vector_filename_a, np.ndarray) or isinstance(vector_filename_a, torch.Tensor):
                    vector_a = vector_filename_a
                else:
                    raise Exception("vector_a is not filepath-str and np.ndarray")

                if vector_filename_b is None:
                    vector_b = self.__get_cell_embedding__(
                        sample_id_b,
                        sample_type_b,
                        sample_gene_id_list_b,
                        sample_gene_seq_type_list_b,
                        sample_gene_seq_list_b,
                        sample_gene_express_bin_list_b,
                        "vector"
                    )
                elif isinstance(vector_filename_b, str):
                    for vector_dir in self.seq_vector_dirpath:
                        vector_filepath = os.path.join(vector_dir, vector_filename_b)
                        if os.path.exists(vector_filepath):
                            vector_b = torch.load(vector_filepath, map_location="cpu", weights_only=True)
                            break
                elif isinstance(vector_filename_b, np.ndarray) or isinstance(vector_filename_b, torch.Tensor):
                    vector_b = vector_filename_b
                else:
                    raise Exception("vector_b is not filepath-str and np.ndarray")

            # for embedding matrix
            matrix_a, matrix_b = None, None
            if "matrix" in self.input_type:
                if matrix_filename_a is None:
                    matrix_a = self.__get_cell_embedding__(
                        sample_id_a,
                        sample_type_a,
                        sample_gene_id_list_a,
                        sample_gene_seq_type_list_a,
                        sample_gene_seq_list_a,
                        sample_gene_express_bin_list_a,
                        "matrix"
                    )
                elif isinstance(matrix_filename_a, str):
                    for matrix_dir in self.seq_matrix_dirpath:
                        matrix_filepath = os.path.join(matrix_dir, matrix_filename_a)
                        if os.path.exists(matrix_filepath):
                            matrix_a = torch.load(matrix_filepath, map_location="cpu", weights_only=True)
                            break
                elif isinstance(matrix_filename_a, np.ndarray) or isinstance(matrix_filename_a, torch.Tensor):
                    matrix_a = matrix_filename_a
                else:
                    raise Exception("matrix_a is not filepath-str and np.ndarray")

                if matrix_filename_b is None:
                    matrix_b = self.__get_cell_embedding__(
                        sample_id_b,
                        sample_type_b,
                        sample_gene_id_list_b,
                        sample_gene_seq_type_list_b,
                        sample_gene_seq_list_b,
                        sample_gene_express_bin_list_b,
                        "matrix"
                    )
                elif isinstance(matrix_filename_b, str):
                    for matrix_dir in self.seq_matrix_dirpath:
                        matrix_filepath = os.path.join(matrix_dir, matrix_filename_b)
                        if os.path.exists(matrix_filepath):
                            matrix_b = torch.load(matrix_filepath, map_location="cpu", weights_only=True)
                            break
                elif isinstance(matrix_filename_b, np.ndarray) or isinstance(matrix_filename_b, torch.Tensor):
                    matrix_b = matrix_filename_b
                else:
                    raise Exception("matrix_b is not filepath-str and np.ndarray")

            return {
                "sample_id_a": sample_id_a,
                "sample_type_a": sample_type_a,
                "sample_gene_id_list_a": None,
                "sample_gene_seq_type_list_a": None,
                "sample_gene_seq_list_a": None,
                "sample_gene_express_bin_list_a": None,
                "sample_gene_seq_vector_list_a": None,
                "sample_gene_seq_matrix_list_a": None,
                "sample_cell_vector_a": vector_a,
                "sample_cell_matrix_a": matrix_a,
                "sample_id_b": sample_id_b,
                "sample_type_b": sample_type_b,
                "sample_gene_id_list_b": None,
                "sample_gene_seq_type_list_b": None,
                "sample_gene_seq_list_b": None,
                "sample_gene_express_bin_list_b": None,
                "sample_gene_seq_vector_list_b": None,
                "sample_gene_seq_matrix_list_b": None,
                "sample_cell_vector_b": vector_b,
                "sample_cell_matrix_b": matrix_b,
                "label": label
        }

    def encode_pair_gene_cell(
            self,
            sample_id_a,
            sample_id_b,
            sample_type_a,
            sample_type_b,
            sample_gene_id_list_a,
            sample_gene_id_list_b,
            sample_gene_seq_type_list_a,
            sample_gene_seq_type_list_b,
            sample_gene_seq_list_a,
            sample_gene_seq_list_b,
            sample_gene_express_bin_list_a,
            sample_gene_express_bin_list_b,
            vector_filename_a=None,
            matrix_filename_a=None,
            vector_filename_b=None,
            matrix_filename_b=None,
            label=None
    ):
        vector_a = []
        for gene_idx, gene_id in enumerate(sample_gene_id_list_a):
            if sample_gene_seq_type_list_a:
                cur_seq_type = sample_gene_seq_type_list_a[gene_idx] if isinstance(sample_gene_seq_type_list_a, list) else sample_gene_seq_type_list_a
            else:
                cur_seq_type = "gene"
            if sample_gene_seq_list_a:
                cur_gene_seq = sample_gene_seq_list_a[gene_idx]
            else:
                cur_gene_seq = None
            cur_vector = self.__get_gene_seq_embedding__(gene_id, cur_seq_type, cur_gene_seq, "vector", self.seq_embedding_vector_type)
            vector_a.append(cur_vector)
        vector_a = torch.stack(vector_a, dim=0)

        # for embedding vector
        vector_b = None
        if "vector" in self.input_type:
            if vector_filename_b is None:
                vector_b = self.__get_cell_embedding__(
                    sample_id_b,
                    sample_type_b,
                    sample_gene_id_list_b,
                    sample_gene_seq_type_list_b,
                    sample_gene_seq_list_b,
                    sample_gene_express_bin_list_b,
                    "vector"
                )
            elif isinstance(vector_filename_b, str):
                for vector_dir in self.seq_vector_dirpath:
                    vector_filepath = os.path.join(vector_dir, vector_filename_b)
                    if os.path.exists(vector_filepath):
                        vector_b = torch.load(vector_filepath, map_location="cpu", weights_only=True)
                        break
            elif isinstance(vector_filename_b, np.ndarray) or isinstance(vector_filename_b, torch.Tensor):
                vector_b = vector_filename_b
            else:
                raise Exception("vector_b is not filepath-str and np.ndarray")

        # for embedding matrix
        matrix_b = None
        if "matrix" in self.input_type:
            if matrix_filename_b is None:
                matrix_b = self.__get_cell_embedding__(
                    sample_id_b,
                    sample_type_b,
                    sample_gene_id_list_b,
                    sample_gene_seq_type_list_b,
                    sample_gene_seq_list_b,
                    sample_gene_express_bin_list_b,
                    "matrix"
                )
            elif isinstance(matrix_filename_b, str):
                for matrix_dir in self.seq_matrix_dirpath:
                    matrix_filepath = os.path.join(matrix_dir, matrix_filename_b)
                    if os.path.exists(matrix_filepath):
                        matrix_b = torch.load(matrix_filepath, map_location="cpu", weights_only=True)
                        break
            elif isinstance(matrix_filename_b, np.ndarray) or isinstance(matrix_filename_b, torch.Tensor):
                matrix_b = matrix_filename_b
            else:
                raise Exception("matrix_b is not filepath-str and np.ndarray")
        return {
            "sample_id_a": sample_id_a,
            "sample_type_a": sample_type_a,
            "sample_gene_id_list_a": None,
            "sample_gene_seq_type_list_a": None,
            "sample_gene_seq_list_a": None,
            "sample_gene_express_bin_list_a": None,
            "sample_gene_seq_vector_list_a": vector_a,
            "sample_gene_seq_matrix_list_a": None,
            "sample_cell_vector_a": None,
            "sample_cell_matrix_a": None,
            "sample_id_b": sample_id_b,
            "sample_type_b": sample_type_b,
            "sample_gene_id_list_b": None,
            "sample_gene_seq_type_list_b": None,
            "sample_gene_seq_list_b": None,
            "sample_gene_express_bin_list_b": None,
            "sample_gene_seq_vector_list_b": None,
            "sample_gene_seq_matrix_list_b": None,
            "sample_cell_vector_b": vector_b,
            "sample_cell_matrix_b": matrix_b,
            "label": label
        }




