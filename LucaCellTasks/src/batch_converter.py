#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 15:14
@project: LucaOnePlusTasks
@file: batch_converter
@desc: batch converter for LucaOnePlusTasks
'''
import sys

import numpy
import torch
import random
from typing import Sequence
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
from typing import Tuple, List
try:
    from llm.lucaone.v2_0.alphabet  import Alphabet as AlphabetLucaOne
    from llm.lucacell.models.alphabet import Alphabet as AlphabetLucaCell
except ImportError:
    from src.llm.lucaone.v2_0.alphabet import Alphabet as AlphabetLucaOne
    from src.llm.lucacell.models.alphabet import Alphabet as AlphabetLucaCell


class BatchConverter(object):
    def __init__(
            self,
            task_level_type,
            label_size,
            output_mode,
            seq_tokenizer,
            seq_no_position_embeddings,
            seq_no_token_type_embeddings,
            seq_truncation_length,
            seq_matrix_add_special_token,
            seq_prepend_bos,
            seq_append_eos,
            cell_tokenizer,
            cell_no_position_embeddings,
            cell_no_token_type_embeddings,
            cell_truncation_length,
            cell_matrix_add_special_token,
            cell_prepend_bos,
            cell_append_eos,
            ignore_index: int = -100,
            padding_idx: int = 0,
            unk_idx: int = 1,
            cls_idx: int = 2,
            eos_idx: int = 3,
            mask_idx: int = 4,
            non_ignore: bool = False,
            mlm_probability=0.15,
            **kwargs
    ):
        print("-" * 25 + "LucaOnePlus Downstream Tasks BatchConverter" + "-" * 25)
        print("BatchConverter, kwargs:")
        print(kwargs)
        self.task_level_type = task_level_type
        self.label_size = label_size
        self.output_mode = output_mode
        self.seq_tokenizer = seq_tokenizer
        self.seq_no_position_embeddings = seq_no_position_embeddings
        self.seq_no_token_type_embeddings = seq_no_token_type_embeddings
        self.seq_truncation_length = seq_truncation_length
        self.seq_matrix_add_special_token = seq_matrix_add_special_token
        self.seq_prepend_bos = seq_prepend_bos
        self.seq_append_eos = seq_append_eos

        self.cell_tokenizer = cell_tokenizer
        self.cell_no_position_embeddings = cell_no_position_embeddings
        self.cell_no_token_type_embeddings = cell_no_token_type_embeddings
        self.cell_truncation_length = cell_truncation_length
        self.cell_matrix_add_special_token = cell_matrix_add_special_token
        self.cell_prepend_bos = cell_prepend_bos
        self.cell_append_eos = cell_append_eos

        if task_level_type in ["gene_level"]:
            self.cell_prepend_bos = False
            self.cell_append_eos = False

        self.ignore_index = ignore_index
        self.non_ignore = non_ignore
        self.mlm_probability = mlm_probability

        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.cls_idx = cls_idx
        self.eos_idx = eos_idx
        self.mask_idx = mask_idx

        self.seq_append_len = int(self.seq_prepend_bos) + int(self.seq_append_eos)
        self.cell_append_len = int(self.cell_prepend_bos) + int(self.cell_append_eos)

        # 减去特殊字符之后的长度
        self.seq_truncation_length -= self.seq_append_len
        self.cell_truncation_length -= self.cell_append_len

        if hasattr(cell_tokenizer, "all_special_token_idx_list"):
            self.all_special_token_idx_list = self.cell_tokenizer.all_special_token_idx_list
        else:
            self.all_special_token_idx_list = [self.padding_idx, self.unk_idx, self.cls_idx, self.eos_idx, self.mask_idx]

        self.input_type = None
        if "input_type" in kwargs and kwargs["input_type"]:
            self.input_type = kwargs["input_type"]
            print("BatchConverter: self.input_type=%s" % self.input_type)
        self.trunc_type = "right"
        if "trunc_type" in kwargs and kwargs["trunc_type"]:
            self.trunc_type = kwargs["trunc_type"]
            print("BatchConverter: self.trunc_type=%s" % self.trunc_type)

        if "lucacell_finetune" in kwargs and kwargs["lucacell_finetune"]:
            self.lucacell_finetune = kwargs["lucacell_finetune"]
        else:
            self.lucacell_finetune = False
        if "not_frozen_gene_express_bin_embedding" in kwargs and kwargs["not_frozen_gene_express_bin_embedding"]:
            self.not_frozen_gene_express_bin_embedding = kwargs["not_frozen_gene_express_bin_embedding"]
        else:
            self.not_frozen_gene_express_bin_embedding = False
        print("*" * 50)

    def __parse_label__(self, max_length, task_level_type, label_size, output_mode, label):
        if task_level_type == "gene_level" and isinstance(label, str) and "[" not in label:
            label = [int(v) for v in label]
        elif isinstance(label, str):
                label = eval(label)
        # 需要是padding长度
        cur_len = max_length
        if task_level_type == "gene_level":
            if output_mode in ["multi_label", "multi-label"]:
                # N * seq_len * label_size
                new_label = []
                for _ in range(cur_len):
                    tmp = []
                    for _ in range(label_size):
                        tmp.append(0 if self.non_ignore else self.ignore_index)
                    new_label.append(tmp)
            else:
                # N * seq_len
                new_label = []
                for _ in range(cur_len):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            if label is not None and len(label) > 0:
                begin_idx = 0
                end_idx = cur_len
                if self.cell_prepend_bos:
                    begin_idx = 1
                if self.cell_append_eos:
                    end_idx = cur_len - 1
                for idx, item in enumerate(label):
                    idx += begin_idx
                    if idx >= end_idx:
                        break
                    if output_mode in ["multi_label", "multi-label"]:
                        for v in item:
                            new_label[idx][v] = 1
                    else:
                        new_label[idx] = item
        elif task_level_type == "span_level":
            if output_mode in ["multi_label", "multi-label"]:
                # N * seq_len * label_size
                new_label = []
                for _ in range(cur_len):
                    tmp = []
                    for _ in range(label_size):
                        tmp.append(0 if self.non_ignore else self.ignore_index)
                    new_label.append(tmp)
            else:
                # N * seq_len
                new_label = []
                for _ in range(cur_len):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            if label is not None and len(label) > 0:
                begin_idx = 0
                end_idx = cur_len
                if self.cell_prepend_bos:
                    begin_idx = 1
                if self.cell_append_eos:
                    end_idx = cur_len - 1
                for item in label:
                    for idx in range(item[0], item[1] + 1, 1):
                        idx += begin_idx
                        if idx >= end_idx:
                            break
                        if output_mode in ["multi_label", "multi-label"]:
                            new_label[idx][item[2]] = 1
                        else:
                            new_label[idx] = item[2]
        elif task_level_type == "cell_level":
            if output_mode in ["multi_label", "multi-label"]:
                # N * label_size
                new_label = []
                for _ in range(label_size):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            else:
                # N * 1
                new_label = [0 if self.non_ignore else self.ignore_index]
            if output_mode in ["multi_label", "multi-label"]:
                if label is not None and len(label) > 0:
                    for v in label:
                        new_label[int(v)] = 1
            elif output_mode == "regression":
                if label and isinstance(label, list):
                    if len(label) == 1:
                        new_label = [float(label[0])]
                    else:
                        new_label = [float(v) for v in label]
                else:
                    new_label = [float(label)]
            else:
                if label is not None and len(str(label)) > 0:
                    if isinstance(label, str):
                        new_label = [int(label)]
                    elif isinstance(label, list):
                        new_label = [int(label[0])]
                    else:
                        new_label = [label]
        else:
            raise Exception("Not support task_level_type=%s" % task_level_type)
        return new_label

    def __mask_express_tokens__(self, input_ids, express_bin_list_len):
        mask_idx = self.mask_idx

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # 特殊字符处为1
        special_tokens_mask = [
            1 if v in self.all_special_token_idx_list else 0 for v in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # 将特殊字符处填充为0.0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 非特殊字符的位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 特殊字符处为-100
        labels[~masked_indices] = self.ignore_index  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with alphabet.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.seq_tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        if torch.any(labels != self.ignore_index):
            return input_ids, labels
        else:
            # non [MASK]， random one position, convect to [MASK]
            rand_idx = random.randint(int(self.cell_prepend_bos), express_bin_list_len + int(self.cell_prepend_bos) - 1)
            labels[rand_idx] = input_ids[rand_idx]
            input_ids[rand_idx] = mask_idx
            return input_ids, labels

    def __gene_seqs_seq_encode__(self, batch_size, batch_seq_types, batch_seqs):
        '''
        seq_encoded_list不加特殊token，input_ids根据设置是否加上特殊token占位
        :param batch_size:
        :param batch_seq_types:
        :param batch_seqs:
        :return:
        '''
        gene_seqs_encoded = []
        gene_seqs_max_len = 0
        gene_seqs_max_num = 0
        for batch_idx, seqs in enumerate(batch_seqs):
            cur_batch_seq_types = batch_seq_types[batch_idx]
            cur_seq_encoded = [self.seq_tokenizer.encode(
                seq_type=cur_batch_seq_types[seq_idx] if isinstance(cur_batch_seq_types, list) else cur_batch_seq_types,
                seq=seq.upper()) for seq_idx, seq in enumerate(seqs)
            ]
            # 该长度已经减去了需要增加的特殊字符的个数
            if self.seq_truncation_length:
                if self.trunc_type == "right":
                    cur_seq_encoded = [encoded[:self.seq_truncation_length] for encoded in cur_seq_encoded]
                else:
                    cur_seq_encoded = [encoded[-self.seq_truncation_length:] for encoded in cur_seq_encoded]
            if self.cell_truncation_length:
                if self.trunc_type == "right":
                    cur_seq_encoded = cur_seq_encoded[:self.cell_truncation_length]
                else:
                    cur_seq_encoded = cur_seq_encoded[-self.cell_truncation_length:]
            gene_seqs_max_len = max(gene_seqs_max_len, max(len(seq_encoded) for seq_encoded in cur_seq_encoded))
            gene_seqs_max_num = max(gene_seqs_max_num, len(cur_seq_encoded))
            gene_seqs_encoded.append(cur_seq_encoded)

        gene_seqs_max_len = gene_seqs_max_len + self.seq_append_len
        gene_seqs_max_num = gene_seqs_max_num + self.cell_append_len
        # for input
        gene_seqs_input_ids = torch.empty(
            (
                batch_size,
                gene_seqs_max_num,
                gene_seqs_max_len,
            ),
            dtype=torch.int64,
        )
        gene_seqs_input_ids.fill_(self.padding_idx)

        gene_seqs_position_ids = None
        if not self.seq_no_position_embeddings:
            gene_seqs_position_ids = torch.empty(
                (
                    batch_size,
                    gene_seqs_max_num,
                    gene_seqs_max_len,
                ),
                dtype=torch.int64,
            )
            gene_seqs_position_ids.fill_(self.padding_idx)

        gene_seqs_token_type_ids = None
        if not self.seq_no_position_embeddings:
            gene_seqs_token_type_ids = torch.empty(
                (
                    batch_size,
                    gene_seqs_max_num,
                    gene_seqs_max_len,
                ),
                dtype=torch.int64,
            )
            gene_seqs_token_type_ids.fill_(self.padding_idx)
        gene_seqs_attention_masks = torch.empty(
            (
                batch_size,
                gene_seqs_max_num,
                gene_seqs_max_len,
            ),
            dtype=torch.int64,
        )
        gene_seqs_attention_masks.fill_(0)

        return gene_seqs_encoded, \
               gene_seqs_input_ids, \
               gene_seqs_position_ids, \
               gene_seqs_token_type_ids, \
               gene_seqs_attention_masks, \
               gene_seqs_max_num, \
               gene_seqs_max_len

    def __gene_seqs_vector_encode__(self, batch_size, vectors):
        gene_seqs_vector_max_num = 0
        vectors_encoded = []
        for batch_idx, cur_vectors in enumerate(vectors):
            if self.cell_truncation_length:
                if self.trunc_type == "right":
                    cur_vectors = cur_vectors[:self.cell_truncation_length]
                else:
                    cur_vectors = cur_vectors[-self.cell_truncation_length:]
            vectors_encoded.append(cur_vectors)
            gene_seqs_vector_max_num = max(gene_seqs_vector_max_num, len(cur_vectors))


        gene_seqs_vector_max_num = gene_seqs_vector_max_num + self.cell_append_len
        embedding_vector_dim = vectors[0][0].shape[0]
        if self.lucacell_finetune:
            filled_vectors = torch.empty(
                (
                    batch_size,
                    gene_seqs_vector_max_num - self.cell_append_len,
                    embedding_vector_dim
                ),
                dtype=torch.float32,
            )
            filled_vectors.fill_(0.0)
        else:
            filled_vectors = torch.empty(
                (
                    batch_size,
                    gene_seqs_vector_max_num,
                    embedding_vector_dim
                ),
                dtype=torch.float32,
            )
            filled_vectors.fill_(0.0)

        vectors_attention_masks = torch.empty(
            (
                batch_size,
                gene_seqs_vector_max_num
            ),
            dtype=torch.int64,
        )
        vectors_attention_masks.fill_(0)
        return vectors_encoded, filled_vectors, vectors_attention_masks, gene_seqs_vector_max_num

    def __gene_seqs_matrix_encode__(self, batch_size, matrices):
        '''
        filled_matrices根据设置是否加上两个特殊符号token的占位行
        :param batch_size:
        :param matrices:
        :return:
        '''
        gene_seqs_matrix_max_len = 0
        gene_seqs_matrix_max_num = 0
        matrices_encoded = []
        for batch_idx, cur_matrices in enumerate(matrices):
            # 该长度已经减去了需要增加的特殊字符的个数
            if self.seq_truncation_length:
                if self.trunc_type == "right":
                    if self.seq_matrix_add_special_token:
                        cur_matrices = [torch.cat((matrix[0], matrix[1:self.seq_truncation_length+1], matrix[-1]), dim=0) for matrix in cur_matrices]
                    else:
                        cur_matrices = [matrix[:self.seq_truncation_length] for matrix in cur_matrices]
                else:
                    if self.seq_matrix_add_special_token:
                        cur_matrices = [torch.cat((matrix[0], matrix[-self.seq_truncation_length - 1: -1], matrix[-1]), dim=0) for matrix in cur_matrices]
                    else:
                        cur_matrices = [matrix[-self.seq_truncation_length:] for matrix in cur_matrices]
            if self.cell_truncation_length:
                if self.trunc_type == "right":
                    cur_matrices = cur_matrices[:self.cell_truncation_length]
                else:
                    cur_matrices = cur_matrices[-self.cell_truncation_length:]
            gene_seqs_matrix_max_len = max(gene_seqs_matrix_max_len, max(len(matrix) for matrix in cur_matrices))
            gene_seqs_matrix_max_num = max(gene_seqs_matrix_max_num, len(cur_matrices))
            matrices_encoded.append(cur_matrices)

        gene_seqs_matrix_max_len = gene_seqs_matrix_max_len + self.seq_append_len
        gene_seqs_matrix_max_num = gene_seqs_matrix_max_num + self.cell_append_len
        embedding_vector_dim = matrices[0][0].shape[1]
        # for input
        filled_matrices = torch.empty(
            (
                batch_size,
                gene_seqs_matrix_max_num,
                gene_seqs_matrix_max_len,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_matrices.fill_(0.0)
        attention_masks = torch.empty(
            (
                batch_size,
                gene_seqs_matrix_max_num,
                gene_seqs_matrix_max_len
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)
        return matrices_encoded, filled_matrices, attention_masks, gene_seqs_matrix_max_num, gene_seqs_matrix_max_len

    def __gene_express_bins_encode__(self, batch_size, batch_gene_express_bin_list):
        if self.cell_truncation_length:
            if self.trunc_type == "right":
                batch_gene_express_bin_list = [gene_express_bin_list[:self.cell_truncation_length] for gene_express_bin_list in batch_gene_express_bin_list]
            else:
                batch_gene_express_bin_list = [gene_express_bin_list[-self.cell_truncation_length:] for gene_express_bin_list in batch_gene_express_bin_list]
        gene_express_list_encoding = self.cell_tokenizer.express_encode(batch_gene_express_bin_list)
        max_gene_express_len = max(len(express) for express in batch_gene_express_bin_list)
        max_gene_express_len = max_gene_express_len + self.cell_append_len
        express_bin_input_ids = torch.empty(
            (
                batch_size,
                max_gene_express_len
            ),
            dtype=torch.int64,
        )
        express_bin_input_ids.fill_(self.padding_idx)
        express_bin_attention_mask = torch.empty(
            (
                batch_size,
                max_gene_express_len
            ),
            dtype=torch.int64,
        )
        express_bin_attention_mask.fill_(0)
        return gene_express_list_encoding, express_bin_input_ids, express_bin_attention_mask, max_gene_express_len

    def __cell_vector_encode__(self, batch_size, vectors):
        embedding_vector_dim = vectors[0].shape[0]
        filled_vectors = torch.empty(
            (
                batch_size,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_vectors.fill_(0.0)
        return filled_vectors, 1

    def __cell_matrix_encode__(self, batch_size, matrices):
        '''
        filled_matrices根据设置是否加上两个特殊符号token的占位行
        :param batch_size:
        :param matrices:
        :return:
        '''
        gene_seqs_max_num = max(matrix.shape[0] for matrix in matrices)
        if self.cell_matrix_add_special_token:
            gene_seqs_max_num -= 2
        if self.cell_truncation_length:
            gene_seqs_max_num = min(gene_seqs_max_num, self.cell_truncation_length)
        gene_seqs_max_num = gene_seqs_max_num + self.cell_append_len
        embedding_vector_dim = matrices[0].shape[1]
        # for input
        filled_matrices = torch.empty(
            (
                batch_size,
                gene_seqs_max_num,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_matrices.fill_(0.0)
        attention_masks = torch.empty(
            (
                batch_size,
                gene_seqs_max_num,
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)
        return filled_matrices, attention_masks, gene_seqs_max_num

    def __call_single__(
            self,
            batch_size,
            cell_types,
            cell_gene_ids,
            cell_gene_seq_types,
            cell_gene_seqs,
            cell_gene_express_bins,
            cell_gene_seq_vectors,
            cell_gene_seq_matrices,
            cell_vectors,
            cell_matrices,
            labels
    ):
        max_length = sys.maxsize
        gene_seqs_seq_part_of_input = False
        gene_seqs_seq_input_ids, gene_seqs_seq_position_ids, gene_seqs_seq_token_type_ids, gene_seqs_seq_attention_masks = None, None, None, None
        if cell_gene_seqs is not None and len(cell_gene_seqs) > 0:
            sample_gene_seqs = []
            for sample_idx, sample_gene_seqs in enumerate(cell_gene_seqs):
                cur_sample_gene_seqs = []
                for gene_idx, gene_seq in enumerate(sample_gene_seqs):
                    cur_sample_gene_seqs.append(self.seq_tokenizer.gene_seq_replace(
                        cell_gene_seq_types[sample_idx][gene_idx] if isinstance(cell_gene_seq_types[sample_idx], list) else cell_gene_seq_types[sample_idx],
                        gene_seq.upper()
                    ))
                sample_gene_seqs.append(cur_sample_gene_seqs)
            gene_seqs_seq_encoded, gene_seqs_seq_input_ids, gene_seqs_seq_position_ids, \
            gene_seqs_seq_token_type_ids, gene_seqs_seq_attention_masks, \
            gene_seqs_seq_max_num, gene_seqs_seq_max_len = self.__gene_seqs_seq_encode__(
                batch_size,
                cell_gene_seq_types,
                sample_gene_seqs
            )
            gene_seqs_seq_part_of_input = True

        gene_seqs_vector_part_of_input = False
        gene_seqs_vector_input, gene_seqs_vector_attention_masks = None, None
        if cell_gene_seq_vectors is not None and len(cell_gene_seq_vectors) > 0:
            gene_seqs_vector_encoded, gene_seqs_vector_input, gene_seqs_vector_attention_masks, gene_seqs_vector_max_num = self.__gene_seqs_vector_encode__(
                batch_size=batch_size,
                vectors=cell_gene_seq_vectors
            )
            gene_seqs_vector_part_of_input = True

        gene_seqs_matrix_part_of_input = False
        gene_seqs_matrix_input, gene_seqs_matrix_attention_masks = None, None
        if cell_gene_seq_matrices is not None and len(cell_gene_seq_matrices) > 0:
            gene_seqs_matrix_encoded, gene_seqs_matrix_input, gene_seqs_matrix_attention_masks, gene_seqs_matrix_max_sum, gene_seqs_matrix_max_len = self.__gene_seqs_matrix_encode__(
                batch_size=batch_size,
                matrices=cell_gene_seq_matrices
            )
            gene_seqs_matrix_part_of_input = True

        cell_vector_part_of_input = False
        cell_encoded_vectors = None
        if cell_vectors is not None and len(cell_vectors) > 0:
            cell_encoded_vectors, _ = self.__cell_vector_encode__(
                batch_size=batch_size,
                vectors=cell_vectors
            )
            cell_vector_part_of_input = True

        cell_matrix_part_of_input = False
        cell_encoded_matrices, cell_matrix_attention_masks = None, None
        if cell_matrices is not None and len(cell_matrices) > 0:
            cell_encoded_matrices, cell_matrix_attention_masks, cell_matrix_max_len = self.__cell_matrix_encode__(
                batch_size=batch_size,
                matrices=cell_matrices
            )
            cell_matrix_part_of_input = True

        has_label = False
        if labels:
            has_label = True

        new_labels = []
        for sample_idx in range(batch_size):
            # gene seqs  as input
            if gene_seqs_seq_part_of_input:
                seq_types = cell_gene_seq_types[sample_idx]
                cls_idx = self.cls_idx
                eos_idx = self.eos_idx
                cur_gene_seqs_seq_encoded = gene_seqs_seq_encoded[sample_idx]
                for gene_idx, gene_seq_encoded in enumerate(cur_gene_seqs_seq_encoded):
                    real_seq_len = len(gene_seq_encoded)
                    if self.seq_prepend_bos:
                        gene_seqs_seq_input_ids[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0] = cls_idx
                    gene_seqs_seq_input_ids[
                        sample_idx,
                        int(self.cell_prepend_bos) + gene_idx,
                        int(self.seq_prepend_bos): real_seq_len + int(self.seq_prepend_bos)
                    ] = torch.tensor(gene_seq_encoded, dtype=torch.int64)
                    if self.seq_append_eos:
                        gene_seqs_seq_input_ids[
                            sample_idx,
                            int(self.cell_prepend_bos) + gene_idx,
                            real_seq_len + int(self.seq_prepend_bos)
                        ] = eos_idx
                    cur_len = real_seq_len + self.seq_append_len
                    if not self.seq_no_position_embeddings:
                        for pos_idx in range(0, cur_len):
                            gene_seqs_seq_position_ids[sample_idx, int(self.cell_prepend_bos) + gene_idx, pos_idx] = pos_idx + 1

                    if not self.seq_no_token_type_embeddings:
                        seq_type = seq_types[sample_idx][gene_idx]
                        if seq_type in ["gene", "dna"]:
                            type_value = 1
                        elif seq_type == "rna":
                            type_value = 2
                        else:
                            type_value = 3

                        for pos_idx in range(0, cur_len):
                            gene_seqs_seq_token_type_ids[sample_idx, int(self.cell_prepend_bos) + gene_idx, pos_idx] = type_value

                    gene_seqs_seq_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: cur_len] = 1

            # gene seqs embedding vector as input
            if gene_seqs_vector_part_of_input:
                cur_gene_seqs_vector_encoded = gene_seqs_vector_encoded[sample_idx]
                real_seq_len = len(cur_gene_seqs_vector_encoded)
                if isinstance(cur_gene_seqs_vector_encoded, Tuple) or isinstance(cur_gene_seqs_vector_encoded, List):
                    gene_seqs_vector_input[sample_idx, int(self.cell_prepend_bos):real_seq_len + int(self.cell_prepend_bos), :] = torch.stack(
                        cur_gene_seqs_vector_encoded,
                        dim=0
                    )
                else:
                    gene_seqs_vector_input[sample_idx, int(self.cell_prepend_bos):real_seq_len + int(self.cell_prepend_bos), :] = torch.tensor(
                        cur_gene_seqs_vector_encoded,
                        dtype=torch.float32
                    ) if isinstance(cur_gene_seqs_vector_encoded, numpy.ndarray) else cur_gene_seqs_vector_encoded.clone()
                cur_len = real_seq_len + self.seq_append_len
                gene_seqs_vector_attention_masks[sample_idx, 0: cur_len] = 1

            # gene seqs embedding matrix as input
            if gene_seqs_matrix_part_of_input:
                cur_gene_seqs_matrix_encoded = gene_seqs_matrix_encoded[sample_idx]
                for gene_idx, gene_seq_matrix in enumerate(cur_gene_seqs_matrix_encoded):
                    real_seq_len = gene_seq_matrix.shape[0]
                    if self.seq_matrix_add_special_token:
                        real_seq_len -= 2
                    try:
                        matrix = gene_seq_matrix.clone().detach()
                    except Exception as e:
                        matrix = torch.tensor(gene_seq_matrix, dtype=torch.float32)
                    if self.seq_matrix_add_special_token and self.seq_prepend_bos and self.seq_append_eos:
                        # 矩阵中有特殊字符的token embedding，且输入也设置了需要
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len + 1] = matrix[0: real_seq_len + 1]
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, real_seq_len + 1] = matrix[-1]
                        gene_seqs_matrix_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len + 2] = 1
                    elif self.seq_matrix_add_special_token:
                        # 矩阵中有特殊字符的token embedding，但输入设置了不需要
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len] = matrix[1: real_seq_len + 1]
                        gene_seqs_matrix_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len] = 1
                    elif self.seq_prepend_bos and self.seq_append_eos:
                        # 矩阵中没有特殊字符的token embedding，但输入设置了需要
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, 1: real_seq_len + 1] = matrix[0: real_seq_len]
                        gene_seqs_matrix_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len + 2] = 1
                    else:
                        # 矩阵中没有特殊字符的token embedding，输入也设置了不需要
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len] = matrix[0: real_seq_len]
                        gene_seqs_matrix_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len] = 1

            # cell embedding vector as input
            if cell_vector_part_of_input:
                cell_encoded_vectors[sample_idx, :] = torch.tensor(
                    cell_vectors[sample_idx],
                    dtype=torch.float32
                )

            # cell embedding matrix as input
            if cell_matrix_part_of_input:
                matrix_encoded = cell_matrices[sample_idx]
                if self.cell_matrix_add_special_token:
                    real_seq_len = matrix_encoded.shape[0] - 2
                else:
                    real_seq_len = matrix_encoded.shape[0]
                real_seq_len = min(real_seq_len, self.cell_truncation_length)
                try:
                    matrix = matrix_encoded.clone().detach()
                except Exception as e:
                    matrix = torch.tensor(matrix_encoded, dtype=torch.float32)
                if self.cell_matrix_add_special_token and self.cell_prepend_bos and self.cell_append_eos:
                    # 矩阵中有特殊字符的token embedding，且输入也设置了需要
                    cell_encoded_matrices[sample_idx, 0: real_seq_len + 1] = matrix[0: real_seq_len + 1]
                    cell_encoded_matrices[sample_idx, real_seq_len + 1] = matrix[-1]
                    cell_matrix_attention_masks[sample_idx, 0: real_seq_len + 2] = 1
                elif self.cell_matrix_add_special_token:
                    # 矩阵中有特殊字符的token embedding，但输入设置了不需要
                    cell_encoded_matrices[sample_idx, 0: real_seq_len] = matrix[1: real_seq_len + 1]
                    cell_matrix_attention_masks[sample_idx, 0: real_seq_len] = 1
                elif self.cell_prepend_bos and self.cell_append_eos:
                    # 矩阵中没有特殊字符的token embedding，但输入设置了需要
                    cell_encoded_matrices[sample_idx, 1: real_seq_len + 1] = matrix[0: real_seq_len]
                    cell_matrix_attention_masks[sample_idx, 0: real_seq_len + 2] = 1
                else:
                    # 矩阵中没有特殊字符的token embedding，输入也设置了不需要
                    cell_encoded_matrices[sample_idx, 0: real_seq_len] = matrix[0: real_seq_len]
                    cell_matrix_attention_masks[sample_idx, 0: real_seq_len] = 1

            if has_label:
                new_labels.append(
                    self.__parse_label__(
                        max_length,
                        self.task_level_type,
                        self.label_size,
                        self.output_mode,
                        labels[sample_idx]
                    ))
        if new_labels is not None and new_labels:
            if self.output_mode in ["regression"]:
                labels = torch.tensor(new_labels, dtype=torch.float32)
            else:
                labels = torch.tensor(new_labels, dtype=torch.int64)
        else:
            labels = None
        return gene_seqs_seq_input_ids, \
               gene_seqs_seq_position_ids, \
               gene_seqs_seq_token_type_ids, \
               gene_seqs_seq_attention_masks, \
               gene_seqs_vector_input, \
               gene_seqs_vector_attention_masks, \
               gene_seqs_matrix_input, \
               gene_seqs_matrix_attention_masks, \
               cell_encoded_vectors, \
               cell_encoded_matrices, \
               cell_matrix_attention_masks, \
               labels

    def __call_single_for_finetune__(
            self,
            batch_size,
            cell_types,
            cell_gene_ids,
            cell_gene_seq_types,
            cell_gene_seqs,
            cell_gene_express_bins,
            cell_gene_seq_vectors,
            cell_gene_seq_matrices,
            cell_vectors,
            cell_matrices,
            labels
    ):
        max_length = sys.maxsize
        gene_seqs_seq_part_of_input = False
        gene_seqs_seq_input_ids, gene_seqs_seq_position_ids, gene_seqs_seq_token_type_ids, gene_seqs_seq_attention_masks = None, None, None, None
        if cell_gene_seqs is not None and len(cell_gene_seqs) > 0:
            sample_gene_seqs = []
            for sample_idx, sample_gene_seqs in enumerate(cell_gene_seqs):
                cur_sample_gene_seqs = []
                for gene_idx, gene_seq in enumerate(sample_gene_seqs):
                    cur_sample_gene_seqs.append(self.seq_tokenizer.gene_seq_replace(
                        cell_gene_seq_types[sample_idx][gene_idx] if isinstance(cell_gene_seq_types[sample_idx], list) else cell_gene_seq_types[sample_idx],
                        gene_seq.upper()
                    ))
                sample_gene_seqs.append(cur_sample_gene_seqs)
            gene_seqs_seq_encoded, gene_seqs_seq_input_ids, gene_seqs_seq_position_ids, \
            gene_seqs_seq_token_type_ids, gene_seqs_seq_attention_masks, \
            gene_seqs_seq_max_num, gene_seqs_seq_max_len = self.__gene_seqs_seq_encode__(
                batch_size,
                cell_gene_seq_types,
                sample_gene_seqs
            )
            gene_seqs_seq_part_of_input = True

        gene_seqs_vector_part_of_input = False
        gene_seqs_vector_input, gene_seqs_vector_attention_masks = None, None
        if cell_gene_seq_vectors is not None and len(cell_gene_seq_vectors) > 0:
            gene_seqs_vector_encoded, gene_seqs_vector_input, gene_seqs_vector_attention_masks, gene_seqs_vector_max_num = self.__gene_seqs_vector_encode__(
                batch_size=batch_size,
                vectors=cell_gene_seq_vectors
            )
            gene_seqs_vector_part_of_input = True

        gene_seqs_matrix_part_of_input = False
        gene_seqs_matrix_input, gene_seqs_matrix_attention_masks = None, None
        if cell_gene_seq_matrices is not None and len(cell_gene_seq_matrices) > 0:
            gene_seqs_matrix_encoded, gene_seqs_matrix_input, gene_seqs_matrix_attention_masks, gene_seqs_matrix_max_sum, gene_seqs_matrix_max_len = self.__gene_seqs_matrix_encode__(
                batch_size=batch_size,
                matrices=cell_gene_seq_matrices
            )
            gene_seqs_matrix_part_of_input = True

        gene_express_list_encoding, gene_express_bin_input_ids, gene_express_bin_attention_mask, max_gene_express_len = self.__gene_express_bins_encode__(batch_size, cell_gene_express_bins)
        has_label = False
        if labels:
            has_label = True

        new_labels = []
        for sample_idx in range(batch_size):
            # gene seqs  as input
            if gene_seqs_seq_part_of_input:
                seq_types = cell_gene_seq_types[sample_idx]
                cls_idx = self.cls_idx
                eos_idx = self.eos_idx
                cur_gene_seqs_seq_encoded = gene_seqs_seq_encoded[sample_idx]
                for gene_idx, gene_seq_encoded in enumerate(cur_gene_seqs_seq_encoded):
                    real_seq_len = len(gene_seq_encoded)
                    if self.seq_prepend_bos:
                        gene_seqs_seq_input_ids[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0] = cls_idx
                    gene_seqs_seq_input_ids[
                        sample_idx,
                        int(self.cell_prepend_bos) + gene_idx,
                        int(self.seq_prepend_bos): real_seq_len + int(self.seq_prepend_bos)
                    ] = torch.tensor(gene_seq_encoded, dtype=torch.int64)
                    if self.seq_append_eos:
                        gene_seqs_seq_input_ids[
                            sample_idx,
                            int(self.cell_prepend_bos) + gene_idx,
                            real_seq_len + int(self.seq_prepend_bos)
                        ] = eos_idx
                    cur_len = real_seq_len + self.seq_append_len
                    if not self.seq_no_position_embeddings:
                        for pos_idx in range(0, cur_len):
                            gene_seqs_seq_position_ids[sample_idx, int(self.cell_prepend_bos) + gene_idx, pos_idx] = pos_idx + 1

                    if not self.seq_no_token_type_embeddings:
                        seq_type = seq_types[sample_idx][gene_idx]
                        if seq_type in ["gene", "dna"]:
                            type_value = 1
                        elif seq_type == "rna":
                            type_value = 2
                        else:
                            type_value = 3

                        for pos_idx in range(0, cur_len):
                            gene_seqs_seq_token_type_ids[sample_idx, int(self.cell_prepend_bos) + gene_idx, pos_idx] = type_value

                    gene_seqs_seq_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: cur_len] = 1

            # gene seqs embedding vector as input
            if gene_seqs_vector_part_of_input:
                cur_gene_seqs_vector_encoded = gene_seqs_vector_encoded[sample_idx]
                real_seq_len = len(cur_gene_seqs_vector_encoded)
                if isinstance(cur_gene_seqs_vector_encoded, Tuple) or isinstance(cur_gene_seqs_vector_encoded, List):
                    if self.lucacell_finetune:
                        gene_seqs_vector_input[sample_idx, 0:real_seq_len, :] = torch.stack(
                            cur_gene_seqs_vector_encoded,
                            dim=0
                        )
                    else:
                        gene_seqs_vector_input[sample_idx, int(self.cell_prepend_bos):real_seq_len + int(self.cell_prepend_bos), :] = torch.stack(
                            cur_gene_seqs_vector_encoded,
                            dim=0
                    )
                else:
                    if self.lucacell_finetune:
                        gene_seqs_vector_input[sample_idx, 0:real_seq_len, :] = torch.tensor(
                            cur_gene_seqs_vector_encoded,
                            dtype=torch.float32
                        ) if isinstance(cur_gene_seqs_vector_encoded, numpy.ndarray) else cur_gene_seqs_vector_encoded.clone()
                    else:
                        gene_seqs_vector_input[sample_idx, int(self.cell_prepend_bos):real_seq_len + int(self.cell_prepend_bos), :] = torch.tensor(
                            cur_gene_seqs_vector_encoded,
                            dtype=torch.float32
                        ) if isinstance(cur_gene_seqs_vector_encoded, numpy.ndarray) else cur_gene_seqs_vector_encoded.clone()

                cur_len = real_seq_len + self.seq_append_len
                gene_seqs_vector_attention_masks[sample_idx, 0: cur_len] = 1

            # gene seqs embedding matrix as input
            if gene_seqs_matrix_part_of_input:
                cur_gene_seqs_matrix_encoded = gene_seqs_matrix_encoded[sample_idx]
                for gene_idx, gene_seq_matrix in enumerate(cur_gene_seqs_matrix_encoded):
                    real_seq_len = gene_seq_matrix.shape[0]
                    if self.seq_matrix_add_special_token:
                        real_seq_len -= 2
                    try:
                        matrix = gene_seq_matrix.clone().detach()
                    except Exception as e:
                        matrix = torch.tensor(gene_seq_matrix, dtype=torch.float32)
                    if self.seq_matrix_add_special_token and self.seq_prepend_bos and self.seq_append_eos:
                        # 矩阵中有特殊字符的token embedding，且输入也设置了需要
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len + 1] = matrix[0: real_seq_len + 1]
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, real_seq_len + 1] = matrix[-1]
                        gene_seqs_matrix_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len + 2] = 1
                    elif self.seq_matrix_add_special_token:
                        # 矩阵中有特殊字符的token embedding，但输入设置了不需要
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len] = matrix[1: real_seq_len + 1]
                        gene_seqs_matrix_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len] = 1
                    elif self.seq_prepend_bos and self.seq_append_eos:
                        # 矩阵中没有特殊字符的token embedding，但输入设置了需要
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, 1: real_seq_len + 1] = matrix[0: real_seq_len]
                        gene_seqs_matrix_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len + 2] = 1
                    else:
                        # 矩阵中没有特殊字符的token embedding，输入也设置了不需要
                        gene_seqs_matrix_input[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len] = matrix[0: real_seq_len]
                        gene_seqs_matrix_attention_masks[sample_idx, int(self.cell_prepend_bos) + gene_idx, 0: real_seq_len] = 1

            # for express list
            if self.cell_prepend_bos:
                gene_express_bin_input_ids[sample_idx, 0] = self.cell_tokenizer.express_cls_idx
            cur_express = gene_express_list_encoding[sample_idx]
            cur_express_list_len = len(cur_express)
            cur_express = torch.tensor(cur_express, dtype=torch.int64)
            gene_express_bin_input_ids[sample_idx, int(self.cell_prepend_bos): cur_express_list_len + int(self.cell_prepend_bos)] = cur_express
            if self.cell_append_eos:
                gene_express_bin_input_ids[sample_idx, cur_express_list_len + int(self.cell_append_eos)] = self.cell_tokenizer.express_eos_idx
            gene_express_bin_attention_mask[sample_idx, 0: cur_express_list_len + self.cell_append_len] = 1

            if has_label:
                new_labels.append(
                    self.__parse_label__(
                        max_length,
                        self.task_level_type,
                        self.label_size,
                        self.output_mode,
                        labels[sample_idx]
                    ))
        if new_labels is not None and new_labels:
            if self.output_mode in ["regression"]:
                labels = torch.tensor(new_labels, dtype=torch.float32)
            else:
                labels = torch.tensor(new_labels, dtype=torch.int64)
        else:
            labels = None
        return gene_seqs_seq_input_ids, \
                   gene_seqs_seq_position_ids, \
                   gene_seqs_seq_token_type_ids, \
                   gene_seqs_seq_attention_masks, \
                   gene_seqs_vector_input, \
                   gene_seqs_vector_attention_masks, \
                   gene_seqs_matrix_input, \
                   gene_seqs_matrix_attention_masks, \
                   gene_express_bin_input_ids, \
                   gene_express_bin_attention_mask, \
                   labels

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        res = {}
        is_pair = False
        if "sample_id_a" in raw_batch[0] and "sample_id_b" in raw_batch[0]:
            is_pair = True
            sample_ids_a = []
            sample_types_a = []
            sample_gene_id_list_a = []
            sample_gene_seq_type_list_a = []
            sample_gene_seq_list_a = []
            sample_gene_express_bin_list_a = []
            sample_gene_seq_vector_list_a = []
            sample_gene_seq_matrix_list_a = []
            sample_cell_vectors_a = []
            sample_cell_matrices_a = []

            sample_ids_b = []
            sample_types_b = []
            sample_gene_id_list_b = []
            sample_gene_seq_type_list_b = []
            sample_gene_seq_list_b = []
            sample_gene_express_bin_list_b = []
            sample_gene_seq_vector_list_b = []
            sample_gene_seq_matrix_list_b = []
            sample_cell_vectors_b = []
            sample_cell_matrices_b = []

            labels = []
            for item in raw_batch:
                sample_ids_a.append(item["sample_id_a"])
                sample_types_a.append(item["sample_type_a"])
                if item["sample_gene_id_list_a"] is not None:
                    sample_gene_id_list_a.append(item["sample_gene_id_list_a"])
                if item["sample_gene_seq_type_list_a"] is not None:
                    sample_gene_seq_type_list_a.append(item["sample_gene_seq_type_list_a"])
                if item["sample_gene_seq_list_a"] is not None:
                    sample_gene_seq_list_a.append(item["sample_gene_seq_list_a"])
                if item["sample_gene_express_bin_list_a"] is not None:
                    sample_gene_express_bin_list_a.append(item["sample_gene_express_bin_list_a"])
                if item["sample_gene_seq_vector_list_a"] is not None:
                    sample_gene_seq_vector_list_a.append(item["sample_gene_seq_vector_list_a"])
                if item["sample_gene_seq_matrix_list_a"] is not None:
                    sample_gene_seq_matrix_list_a.append(item["sample_gene_seq_matrix_list_a"])
                if item["sample_cell_vector_a"] is not None:
                    sample_cell_vectors_a.append(item["sample_cell_vector_a"])
                if item["sample_cell_matrix_a"] is not None:
                    sample_cell_matrices_a.append(item["sample_cell_matrix_a"])

                sample_ids_b.append(item["sample_id_b"])
                sample_types_b.append(item["sample_type_b"])
                if item["sample_gene_id_list_b"] is not None:
                    sample_gene_id_list_b.append(item["sample_gene_id_list_b"])
                if item["sample_gene_seq_type_list_b"] is not None:
                    sample_gene_seq_type_list_b.append(item["sample_gene_seq_type_list_b"])
                if item["sample_gene_seq_list_b"] is not None:
                    sample_gene_seq_list_b.append(item["sample_gene_seq_list_b"])
                if item["sample_gene_express_bin_list_b"] is not None:
                    sample_gene_express_bin_list_b.append(item["sample_gene_express_bin_list_b"])
                if item["sample_gene_seq_vector_list_b"] is not None:
                    sample_gene_seq_vector_list_b.append(item["sample_gene_seq_vector_list_b"])
                if item["sample_gene_seq_matrix_list_b"] is not None:
                    sample_gene_seq_matrix_list_b.append(item["sample_gene_seq_matrix_list_b"])
                if item["sample_cell_vector_b"] is not None:
                    sample_cell_vectors_b.append(item["sample_cell_vector_b"])
                if item["sample_cell_matrix_b"] is not None:
                    sample_cell_matrices_b.append(item["sample_cell_matrix_b"])

                if "label" in item and item["label"] is not None:
                    labels.append(item["label"])
        else:
            sample_ids = []
            sample_types = []
            sample_gene_id_list = []
            sample_gene_seq_type_list = []
            sample_gene_seq_list = []
            sample_gene_express_bin_list = []
            sample_gene_seq_vector_list = []
            sample_gene_seq_matrix_list = []
            sample_cell_vectors = []
            sample_cell_matrices = []
            labels = []
            for item in raw_batch:
                sample_ids.append(item["sample_id"])
                sample_types.append(item["sample_type"])
                if item["sample_gene_id_list"] is not None:
                    sample_gene_id_list.append(item["sample_gene_id_list"])
                if item["sample_gene_seq_type_list"] is not None:
                    sample_gene_seq_type_list.append(item["sample_gene_seq_type_list"])
                if item["sample_gene_seq_list"] is not None:
                    sample_gene_seq_list.append(item["sample_gene_seq_list"])
                if item["sample_gene_express_bin_list"] is not None:
                    sample_gene_express_bin_list.append(item["sample_gene_express_bin_list"])
                if item["sample_gene_seq_vector_list"] is not None:
                    sample_gene_seq_vector_list.append(item["sample_gene_seq_vector_list"])
                if item["sample_gene_seq_matrix_list"] is not None:
                    sample_gene_seq_matrix_list.append(item["sample_gene_seq_matrix_list"])
                if item["sample_cell_vector"] is not None:
                    sample_cell_vectors.append(item["sample_cell_vector"])
                if item["sample_cell_matrix"] is not None:
                    sample_cell_matrices.append(item["sample_cell_matrix"])

                if "label" in item and item["label"] is not None:
                    labels.append(item["label"])
        if self.lucacell_finetune or self.not_frozen_gene_express_bin_embedding:
            if is_pair:
                # todo
                pass
            else:
                gene_seqs_seq_input_ids, \
                    gene_seqs_seq_position_ids, \
                    gene_seqs_seq_token_type_ids, \
                    gene_seqs_seq_attention_masks, \
                    gene_seqs_vector_input, \
                    gene_seqs_vector_attention_masks, \
                    gene_seqs_matrix_input, \
                    gene_seqs_matrix_attention_masks, \
                    gene_express_input_ids, \
                    gene_express_attention_masks, \
                    labels = self.__call_single_for_finetune__(
                        batch_size,
                        sample_types,
                        sample_gene_id_list,
                        sample_gene_seq_type_list,
                        sample_gene_seq_list,
                        sample_gene_express_bin_list,
                        sample_gene_seq_vector_list,
                        sample_gene_seq_matrix_list,
                        sample_cell_vectors,
                        sample_cell_matrices,
                        labels
                    )
                if self.lucacell_finetune:
                    res.update({
                        "nucleotide_input_ids": gene_seqs_seq_input_ids,
                        "nucleotide_position_ids": gene_seqs_seq_position_ids,
                        "nucleotide_attention_mask": gene_seqs_vector_attention_masks if gene_seqs_vector_input is not None else gene_seqs_seq_attention_masks,
                        "nucleotide_input_embeds": gene_seqs_vector_input,
                        "express_input_ids": gene_express_input_ids,
                        "express_attention_mask": gene_express_attention_masks,
                        "labels": labels if labels is not None and len(labels) > 0 else None
                    })
                else:
                    res.update({
                        "cell_gene_seq_inputs_embeds": gene_seqs_vector_input,
                        "cell_gene_express_input_ids": gene_express_input_ids,
                        "cell_gene_attention_masks": gene_express_attention_masks if gene_express_attention_masks is not None else gene_seqs_seq_attention_masks,
                        "labels": labels if labels is not None and len(labels) > 0 else None
                    })
            return res
        else:
            # pair
            if is_pair:
                gene_seqs_seq_input_ids_a, \
                    gene_seqs_seq_position_ids_a, \
                    gene_seqs_seq_token_type_ids_a, \
                    gene_seqs_seq_attention_masks_a, \
                    gene_seqs_vector_input_a, \
                    gene_seqs_vector_attention_masks_a, \
                    gene_seqs_matrix_input_a, \
                    gene_seqs_matrix_attention_masks_a, \
                    cell_encoded_vectors_a, \
                    cell_encoded_matrices_a, \
                    cell_matrix_attention_masks_a, \
                    labels = self.__call_single__(
                        batch_size,
                        sample_types_a,
                        sample_gene_id_list_a,
                        sample_gene_seq_type_list_a,
                        sample_gene_seq_list_a,
                        sample_gene_express_bin_list_a,
                        sample_gene_seq_vector_list_a,
                        sample_gene_seq_matrix_list_a,
                        sample_cell_vectors_a,
                        sample_cell_matrices_a,
                        labels
                    )
                res.update({
                    "gene_seqs_seq_input_ids_a": gene_seqs_seq_input_ids_a,
                    "gene_seqs_seq_position_ids_a": gene_seqs_seq_position_ids_a,
                    "gene_seqs_seq_token_type_ids_a": gene_seqs_seq_token_type_ids_a,
                    "gene_seqs_seq_attention_masks_a": gene_seqs_seq_attention_masks_a,
                    "gene_seqs_vector_input_a": gene_seqs_vector_input_a,
                    "gene_seqs_vector_attention_masks_a": gene_seqs_vector_attention_masks_a,
                    "gene_seqs_matrix_input_a": gene_seqs_matrix_input_a,
                    "gene_seqs_matrix_attention_masks_a": gene_seqs_matrix_attention_masks_a,
                    "cell_encoded_vectors_a": cell_encoded_vectors_a,
                    "cell_encoded_matrices_a": cell_encoded_matrices_a,
                    "cell_matrix_attention_masks_a": cell_matrix_attention_masks_a,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })

                gene_seqs_seq_input_ids_b, \
                    gene_seqs_seq_position_ids_b, \
                    gene_seqs_seq_token_type_ids_b, \
                    gene_seqs_seq_attention_masks_b, \
                    gene_seqs_vector_input_b, \
                    gene_seqs_vector_attention_masks_b, \
                    gene_seqs_matrix_input_b, \
                    gene_seqs_matrix_attention_masks_b, \
                    cell_encoded_vectors_b, \
                    cell_encoded_matrices_b, \
                    cell_matrix_attention_masks_b, \
                    _ = self.__call_single__(
                        batch_size,
                        sample_types_b,
                        sample_gene_id_list_b,
                        sample_gene_seq_type_list_b,
                        sample_gene_seq_list_b,
                        sample_gene_express_bin_list_b,
                        sample_gene_seq_vector_list_b,
                        sample_gene_seq_matrix_list_b,
                        sample_cell_vectors_b,
                        sample_cell_matrices_b,
                        None
                    )
                res.update({
                    "gene_seqs_seq_input_ids_b": gene_seqs_seq_input_ids_b,
                    "gene_seqs_seq_position_ids_b": gene_seqs_seq_position_ids_b,
                    "gene_seqs_seq_token_type_ids_b": gene_seqs_seq_token_type_ids_b,
                    "gene_seqs_seq_attention_masks_b": gene_seqs_seq_attention_masks_b,
                    "gene_seqs_vector_input_b": gene_seqs_vector_input_b,
                    "gene_seqs_vector_attention_masks_b": gene_seqs_vector_attention_masks_b,
                    "gene_seqs_matrix_input_b": gene_seqs_matrix_input_b,
                    "gene_seqs_matrix_attention_masks_b": gene_seqs_matrix_attention_masks_b,
                    "cell_encoded_vectors_b": cell_encoded_vectors_b,
                    "cell_encoded_matrices_b": cell_encoded_matrices_b,
                    "cell_matrix_attention_masks_b": cell_matrix_attention_masks_b
                })
                return res
            else:
                gene_seqs_seq_input_ids, \
                    gene_seqs_seq_position_ids, \
                    gene_seqs_seq_token_type_ids, \
                    gene_seqs_seq_attention_masks, \
                    gene_seqs_vector_input, \
                    gene_seqs_vector_attention_masks, \
                    gene_seqs_matrix_input, \
                    gene_seqs_matrix_attention_masks, \
                    cell_encoded_vectors, \
                    cell_encoded_matrices, \
                    cell_matrix_attention_masks, \
                    labels = self.__call_single__(
                        batch_size,
                        sample_types,
                        sample_gene_id_list,
                        sample_gene_seq_type_list,
                        sample_gene_seq_list,
                        sample_gene_express_bin_list,
                        sample_gene_seq_vector_list,
                        sample_gene_seq_matrix_list,
                        sample_cell_vectors,
                        sample_cell_matrices,
                        labels
                    )
                res.update({
                    "gene_seqs_seq_input_ids": gene_seqs_seq_input_ids,
                    "gene_seqs_seq_position_ids": gene_seqs_seq_position_ids,
                    "gene_seqs_seq_token_type_ids": gene_seqs_seq_token_type_ids,
                    "gene_seqs_seq_attention_masks": gene_seqs_seq_attention_masks,
                    "gene_seqs_vector_input": gene_seqs_vector_input,
                    "gene_seqs_vector_attention_masks": gene_seqs_vector_attention_masks,
                    "gene_seqs_matrix_input": gene_seqs_matrix_input,
                    "gene_seqs_matrix_attention_masks": gene_seqs_matrix_attention_masks,
                    "cell_encoded_vectors": cell_encoded_vectors,
                    "cell_encoded_matrices": cell_encoded_matrices,
                    "cell_matrix_attention_masks": cell_matrix_attention_masks,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })
                return res

