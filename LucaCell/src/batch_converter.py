#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 15:14
@project: LucaOne
@file: batch_converter
@desc: batch converter
'''
import random
import torch
import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../src")
try:
    from models.alphabet import Alphabet
except ImportError:
    from src.models.alphabet import Alphabet


class BatchConverter(object):
    def __init__(
            self,
            alphabet: Alphabet,
            nucleotide_input_type,
            no_nucleotide_position_embeddings,
            no_gene_positions_embeddings,
            no_express_sorted_embeddings,
            no_gene_type_embeddings,
            truncation_nucleotide_seq_length: int = None,
            truncation_gene_seq_length: int = None,
            ignore_index: int = -100,
            mlm_probability=0.5
    ):
        self.alphabet = alphabet
        self.nucleotide_input_type = nucleotide_input_type
        self.ignore_index = ignore_index
        self.mlm_probability = mlm_probability
        self.no_nucleotide_position_embeddings = no_nucleotide_position_embeddings
        self.no_express_sorted_embeddings = no_express_sorted_embeddings
        self.no_gene_positions_embeddings = no_gene_positions_embeddings
        self.no_gene_type_embeddings = no_gene_type_embeddings
        self.truncation_nucleotide_seq_length = truncation_nucleotide_seq_length - int(alphabet.nucleotide_prepend_bos) - int(alphabet.nucleotide_append_eos)
        self.truncation_gene_seq_length = truncation_gene_seq_length - int(alphabet.gene_prepend_bos) - int(alphabet.gene_append_eos)

    def __mask_tokens__(
            self,
            gene_input_ids,
            express_input_ids,
            express_sorted_position_ids,
            seq_len
    ):
        labels = {}
        if gene_input_ids is not None:
            labels = {
                "gene_mask": gene_input_ids.clone(),
            }
        if express_sorted_position_ids is not None:
            labels.update({
                "express_sorted_mask": express_sorted_position_ids.clone()
            })
        labels.update({
            "express_value_mask": express_input_ids.clone()
        })
        probability_matrix = torch.full(express_input_ids.shape, self.mlm_probability)

        # 特殊字符处为1
        special_tokens_mask = [
            1 if v in self.alphabet.all_express_special_token_ids and v != self.alphabet.get_express_idx('[E_NON]') else 0 for v in express_input_ids.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # 将特殊字符处填充为0.0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 非特殊字符的位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 特殊字符处为-100
        # We only compute loss on masked tokens
        if gene_input_ids is not None:
            labels["gene_mask"][~masked_indices] = self.ignore_index
        if express_sorted_position_ids is not None:
            labels["express_sorted_mask"][~masked_indices] = self.ignore_index
        labels["express_value_mask"][~masked_indices] = self.ignore_index

        # 80% of the time, we replace masked input tokens with alphabet.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(express_input_ids.shape, 0.8)).bool() & masked_indices
        if gene_input_ids is not None:
            gene_input_ids[indices_replaced] = self.alphabet.gene_mask_idx
        if express_sorted_position_ids is not None:
            express_sorted_position_ids[indices_replaced] = self.alphabet.express_sorted_mask_idx
        express_input_ids[indices_replaced] = self.alphabet.express_mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(express_input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        if gene_input_ids is not None:
            random_gene_words = torch.randint(1, self.alphabet.gene_vocab_size, express_input_ids.shape, dtype=torch.long)
            gene_input_ids[indices_random] = random_gene_words[indices_random]
        if express_sorted_position_ids is not None:
            random_express_sorted_words = torch.randint(1, self.alphabet.express_sorted_vocab_size, express_input_ids.shape, dtype=torch.long)
            express_sorted_position_ids[indices_random] = random_express_sorted_words[indices_random]
        random_express_words = torch.randint(1, self.alphabet.express_vocab_size, express_input_ids.shape, dtype=torch.long)
        express_input_ids[indices_random] = random_express_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        if torch.any(labels["express_value_mask"] != self.ignore_index):
            return gene_input_ids, express_input_ids, express_sorted_position_ids, labels
        else:
            # non [MASK]， random one position, convect to [MASK]
            rand_idx = random.randint(int(self.alphabet.express_prepend_bos), seq_len + int(self.alphabet.express_prepend_bos) - 1)
            labels["express_value_mask"][rand_idx] = express_input_ids[rand_idx]
            express_input_ids[rand_idx] = self.alphabet.express_mask_idx
            if gene_input_ids is not None:
                labels["gene_mask"][rand_idx] = gene_input_ids[rand_idx]
                gene_input_ids[rand_idx] = self.alphabet.gene_mask_idx
            if express_sorted_position_ids is not None:
                labels["express_sorted_mask"][rand_idx] = express_sorted_position_ids[rand_idx]
                express_sorted_position_ids[rand_idx] = self.alphabet.express_sorted_mask_idx
            return gene_input_ids, express_input_ids, express_sorted_position_ids, labels

    def __call_single__(
            self,
            batch_size,
            sample_types,
            sample_gene_id_list,
            sample_gene_seq_list,
            sample_gene_embedding_list,
            sample_gene_express_list,
            sample_gene_express_list_raw,
    ):
        if self.truncation_gene_seq_length:
            if sample_gene_seq_list:
                sample_gene_seq_list = [gene_seq_list[:self.truncation_gene_seq_length] for gene_seq_list in sample_gene_seq_list]
            if sample_gene_id_list:
                sample_gene_id_list = [gene_id_list[:self.truncation_gene_seq_length] for gene_id_list in sample_gene_id_list]
            if sample_gene_embedding_list:
                sample_gene_embedding_list = [gene_embedding_list[:self.truncation_gene_seq_length] for gene_embedding_list in sample_gene_embedding_list]
            sample_gene_express_list = [gene_express_list[:self.truncation_gene_seq_length] for gene_express_list in sample_gene_express_list]

        max_gene_seq_len = max(len(express) for express in sample_gene_express_list)
        max_gene_seq_len = max_gene_seq_len + int(self.alphabet.express_prepend_bos) + int(self.alphabet.express_append_eos)

        if sample_gene_seq_list:
            nucleotide_attention_mask = None
            max_nucleotide_seq_len_list = []
            if self.truncation_nucleotide_seq_length:
                new_sample_gene_seq_list = []
                for sample_idx, gene_seqs in enumerate(sample_gene_seq_list):
                    new_gene_seqs = [gene_seq[:self.truncation_nucleotide_seq_length] for gene_seq in gene_seqs]
                    new_sample_gene_seq_list.append(new_gene_seqs)
                    for idj in range(len(new_gene_seqs)):
                        while len(max_nucleotide_seq_len_list) <= idj:
                            max_nucleotide_seq_len_list.append([])
                        max_nucleotide_seq_len_list[idj].append(len(new_gene_seqs[idj]))
                sample_gene_seq_list = new_sample_gene_seq_list
            else:
                for sample_idx, gene_seqs in enumerate(sample_gene_seq_list):
                    for idj in range(len(gene_seqs)):
                        while len(max_nucleotide_seq_len_list) <= idj:
                            max_nucleotide_seq_len_list.append([])
                        max_nucleotide_seq_len_list[idj].append(len(gene_seqs[idj]))
            nucleotide_input_ids = []
            nucleotide_position_ids = []
            max_seq_len = max_gene_seq_len - int(self.alphabet.express_prepend_bos) - int(self.alphabet.express_append_eos)
            for idx in range(max_seq_len):
                cur_max_nucleotide_seq_len = max(max_nucleotide_seq_len_list[idx]) + int(self.alphabet.nucleotide_prepend_bos) + int(self.alphabet.nucleotide_append_eos)
                cur_nucleotide_input_ids = torch.empty(
                    (
                        batch_size,
                        cur_max_nucleotide_seq_len
                    ),
                    dtype=torch.int64,
                )
                cur_nucleotide_input_ids.fill_(self.alphabet.nucleotide_padding_idx)
                nucleotide_input_ids.append(cur_nucleotide_input_ids)
                cur_nucleotide_position_ids = torch.zeros(
                    (
                        batch_size,
                        cur_max_nucleotide_seq_len
                    ),
                    dtype=torch.int64,
                )
                nucleotide_position_ids.append(cur_nucleotide_position_ids)
            nucleotide_input_embeds = None
            gene_input_ids = None
        elif sample_gene_embedding_list:
            embed_dim = None
            max_nucleotide_seq_len_list = []
            if batch_size > 1:
                nucleotide_attention_mask = []
            else:
                nucleotide_attention_mask = None
            if self.truncation_nucleotide_seq_length:
                if self.nucleotide_input_type == "embedding_matrix":
                    new_sample_gene_embedding_list = []
                    for sample_idx, gene_embeddings in enumerate(sample_gene_embedding_list):
                        embed_dim = gene_embeddings[0].shape[1]
                        new_gene_embeddings = [torch.cat([gene_embedding[:self.truncation_nucleotide_seq_length + 1, :], gene_embedding[-1:, :]], dim=0) for gene_embedding in gene_embeddings]
                        new_sample_gene_embedding_list.append(new_gene_embeddings)
                        for idj in range(len(new_gene_embeddings)):
                            while len(max_nucleotide_seq_len_list) <= idj:
                                max_nucleotide_seq_len_list.append([])
                            max_nucleotide_seq_len_list[idj].append(new_gene_embeddings[idj].shape[0] - 2)
                else:
                    embed_dim = sample_gene_embedding_list[0][0].shape[0]
                    new_sample_gene_embedding_list = sample_gene_embedding_list
            else:
                new_sample_gene_embedding_list = sample_gene_embedding_list
                if self.nucleotide_input_type == "embedding_matrix":
                    for sample_idx, gene_embeddings in enumerate(sample_gene_embedding_list):
                        embed_dim = gene_embeddings[0].shape[1]
                        for idj in range(len(gene_embeddings)):
                            while len(max_nucleotide_seq_len_list) <= idj:
                                max_nucleotide_seq_len_list.append([])
                            max_nucleotide_seq_len_list[idj].append(gene_embeddings[idj].shape[0] - 2)
                else:
                    embed_dim = sample_gene_embedding_list[0][0].shape[0]

            nucleotide_input_ids = None
            nucleotide_position_ids = None
            nucleotide_input_embeds = []
            max_embedding_len = max_gene_seq_len - int(self.alphabet.express_prepend_bos) - int(self.alphabet.express_append_eos)
            for idx in range(max_embedding_len):
                if self.nucleotide_input_type == "embedding_matrix":
                    cur_max_nucleotide_seq_len = max(max_nucleotide_seq_len_list[idx]) + int(self.alphabet.nucleotide_prepend_bos) + int(self.alphabet.nucleotide_append_eos)
                    if self.alphabet.nucleotide_prepend_bos and self.alphabet.nucleotide_append_eos:
                        cur_nucleotide_embeddings = torch.zeros(
                            (
                                batch_size,
                                cur_max_nucleotide_seq_len,
                                embed_dim
                            ),
                            dtype=torch.float32,
                        )
                        if batch_size > 1:
                            cur_nucleotide_attention_mask = torch.zeros(
                                (
                                    batch_size,
                                    cur_max_nucleotide_seq_len
                                ),
                                dtype=torch.long,
                            )
                        for sample_idx, item in enumerate(new_sample_gene_embedding_list):
                            m = item[idx]
                            cur_nucleotide_embeddings[sample_idx, :m.shape[0], :] = m
                            if batch_size > 1:
                                cur_nucleotide_attention_mask[sample_idx, :m.shape[0]] = 1
                    else:
                        cur_nucleotide_embeddings = torch.zeros(
                            (
                                batch_size,
                                cur_max_nucleotide_seq_len,
                                embed_dim
                            ),
                            dtype=torch.float32,
                        )
                        if batch_size > 1:
                            cur_nucleotide_attention_mask = torch.zeros(
                                (
                                    batch_size,
                                    cur_max_nucleotide_seq_len
                                ),
                                dtype=torch.long,
                            )
                        for sample_idx, item in enumerate(new_sample_gene_embedding_list):
                            m = item[idx]
                            cur_nucleotide_embeddings[sample_idx, :m.shape[0] - 2, :] = m[1:-1, :]
                            if batch_size > 1:
                                cur_nucleotide_attention_mask[sample_idx, :m.shape[0] - 2] = 1
                    nucleotide_input_embeds.append(cur_nucleotide_embeddings)
                    if batch_size > 1:
                        nucleotide_attention_mask.append(cur_nucleotide_attention_mask)
                else:
                    cur_nucleotide_embeddings = torch.zeros(
                        (
                            batch_size,
                            embed_dim
                        ),
                        dtype=torch.float32,
                    )
                    for sample_idx, item in enumerate(new_sample_gene_embedding_list):
                        if idx >= len(item):
                            raise IndexError(f"Index {idx} out of range for item at index {sample_idx}")
                        m = item[idx]
                        cur_nucleotide_embeddings[sample_idx, :] = m
                    # seq_len list: (batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
                    nucleotide_input_embeds.append(cur_nucleotide_embeddings)
            if self.nucleotide_input_type != "embedding_matrix":
                nucleotide_input_embeds = torch.stack(nucleotide_input_embeds, dim=1)
                nucleotide_attention_mask = None
            gene_input_ids = None
        else:
            nucleotide_input_ids = None
            nucleotide_input_embeds = None
            gene_input_ids = torch.empty(
                (
                    batch_size,
                    max_gene_seq_len
                ),
                dtype=torch.int64,
            )
            gene_input_ids.fill_(self.alphabet.gene_padding_idx)
            if not self.no_gene_positions_embeddings:
                gene_position_ids = torch.zeros(
                    (
                        batch_size,
                        max_gene_seq_len
                    ),
                    dtype=torch.int64,
                )
        gene_type_ids = None
        if not self.no_gene_type_embeddings:
            gene_type_ids = torch.zeros(
                (
                    batch_size,
                    max_gene_seq_len
                ),
                dtype=torch.int64,
            )

        express_input_ids = torch.empty(
            (
                batch_size,
                max_gene_seq_len
            ),
            dtype=torch.int64,
        )
        express_input_ids.fill_(self.alphabet.express_padding_idx)

        express_sorted_position_ids = None
        if not self.no_express_sorted_embeddings:
            express_sorted_position_ids = torch.zeros(
                (
                    batch_size,
                    max_gene_seq_len
                ),
                dtype=torch.int64,
            )
        # strs = []
        labels = []
        for sample_idx, cur_express in enumerate(sample_gene_express_list):
            # strs.append(";".join([str(v) for v in cur_express]))
            if nucleotide_input_ids is not None:
                cur_gene_seq_list = sample_gene_seq_list[sample_idx]
                for cur_gene_seq_idx, cur_gene_seq in enumerate(cur_gene_seq_list):
                    if self.alphabet.nucleotide_prepend_bos:
                        nucleotide_input_ids[cur_gene_seq_idx][sample_idx, 0] = self.alphabet.nucleotide_cls_idx
                    cur_gene_seq_len = len(cur_gene_seq)
                    cur_gene_seq = torch.tensor(cur_gene_seq, dtype=torch.int64)
                    nucleotide_input_ids[cur_gene_seq_idx][sample_idx, int(self.alphabet.nucleotide_prepend_bos): cur_gene_seq_len + int(self.alphabet.nucleotide_prepend_bos)] = cur_gene_seq
                    if self.alphabet.nucleotide_append_eos:
                        nucleotide_input_ids[cur_gene_seq_idx][sample_idx, cur_gene_seq_len + int(self.alphabet.nucleotide_prepend_bos)] = self.alphabet.nucleotide_eos_idx
                    if not self.no_nucleotide_position_embeddings:
                        for idx in range(0, cur_gene_seq_len + int(self.alphabet.nucleotide_prepend_bos) + int(self.alphabet.nucleotide_append_eos)):
                            nucleotide_position_ids[cur_gene_seq_idx][sample_idx, idx] = idx + 1
            elif nucleotide_input_embeds is not None:
                pass
            else:
                cur_gene_id_list = sample_gene_id_list[sample_idx]
                if self.alphabet.gene_prepend_bos:
                    gene_input_ids[sample_idx, 0] = self.alphabet.gene_cls_idx
                cur_gene_id_size = len(cur_gene_id_list)
                gene_ids = torch.tensor(cur_gene_id_list, dtype=torch.int64)
                gene_input_ids[sample_idx, int(self.alphabet.gene_prepend_bos): cur_gene_id_size + int(self.alphabet.gene_prepend_bos)] = gene_ids
                if self.alphabet.gene_append_eos:
                    gene_input_ids[sample_idx, cur_gene_id_size + int(self.alphabet.gene_prepend_bos)] = self.alphabet.gene_eos_idx

                if not self.no_gene_positions_embeddings:
                    for idx in range(0, cur_gene_id_size + int(self.alphabet.gene_prepend_bos) + int(self.alphabet.gene_append_eos)):
                        gene_position_ids[sample_idx, idx] = idx + 1

            if not self.no_gene_type_embeddings:
                cur_sample_type = sample_types[sample_idx].lower()
                if "rna" in cur_sample_type:
                    type_value = 2
                else:
                    type_value = 1
                cur_len = int(self.alphabet.express_prepend_bos) + len(cur_express) + int(self.alphabet.express_append_eos)
                for idx in range(0, cur_len):
                    gene_type_ids[sample_idx, idx] = type_value

            # for express list
            if self.alphabet.express_prepend_bos:
                express_input_ids[sample_idx, 0] = self.alphabet.express_cls_idx
            cur_express_list_len = len(cur_express)
            cur_express = torch.tensor(cur_express, dtype=torch.int64)
            express_input_ids[sample_idx, int(self.alphabet.express_prepend_bos): cur_express_list_len + int(self.alphabet.express_prepend_bos)] = cur_express
            if self.alphabet.express_append_eos:
                express_input_ids[sample_idx, cur_express_list_len + int(self.alphabet.express_prepend_bos)] = self.alphabet.express_eos_idx

            if not self.no_express_sorted_embeddings:
                cur_sample_gene_express_list_raw = sample_gene_express_list_raw[sample_idx]
                for idx in range(int(self.alphabet.express_prepend_bos), cur_express_list_len + int(self.alphabet.express_prepend_bos)):
                    express_sorted_position_ids[sample_idx, idx] = int(cur_sample_gene_express_list_raw[idx - int(self.alphabet.express_prepend_bos)]) + 4
                if self.alphabet.express_prepend_bos:
                    express_sorted_position_ids[sample_idx, 0] = self.alphabet.express_sorted_cls_idx
                if self.alphabet.express_append_eos:
                    express_sorted_position_ids[sample_idx, cur_express_list_len + int(self.alphabet.express_prepend_bos)] = self.alphabet.express_sorted_eos_idx

            if self.alphabet.alphabet_type == "nucleotide":
                if not self.no_express_sorted_embeddings:
                    _, express_input_ids[sample_idx, :], express_sorted_position_ids[sample_idx, :], label = self.__mask_tokens__(
                        None,
                        express_input_ids[sample_idx, :],
                        express_sorted_position_ids[sample_idx, :],
                        seq_len=cur_express_list_len
                    )
                else:
                    _, express_input_ids[sample_idx, :], _, label = self.__mask_tokens__(
                        None,
                        express_input_ids[sample_idx, :],
                        None,
                        seq_len=cur_express_list_len
                    )
            else:
                if not self.no_express_sorted_embeddings:
                    gene_input_ids[sample_idx, :], express_input_ids[sample_idx, :], express_sorted_position_ids[sample_idx, :], label = self.__mask_tokens__(
                        gene_input_ids[sample_idx, :],
                        express_input_ids[sample_idx, :],
                        express_sorted_position_ids[sample_idx, :],
                        seq_len=cur_express_list_len
                    )
                else:
                    gene_input_ids[sample_idx, :], express_input_ids[sample_idx, :], _, label = self.__mask_tokens__(
                        gene_input_ids[sample_idx, :],
                        express_input_ids[sample_idx, :],
                        None,
                        seq_len=cur_express_list_len
                    )
            labels.append(label)
        new_labels = {}
        for label in labels:
            if "express_value_mask" in label:
                if "express_value_mask" not in new_labels:
                    new_labels["express_value_mask"] = []
                new_labels["express_value_mask"].append(label["express_value_mask"])
            if "express_sorted_mask" in label:
                if "express_sorted_mask" not in new_labels:
                    new_labels["express_sorted_mask"] = []
                new_labels["express_sorted_mask"].append(label["express_sorted_mask"])
            if "gene_mask" in label:
                if "gene_mask" not in new_labels:
                    new_labels["gene_mask"] = []
                new_labels["gene_mask"].append(label["gene_mask"])
        if "express_value_mask" in new_labels:
            new_labels["express_value_mask"] = torch.stack(new_labels["express_value_mask"], dim=0)
        if "express_sorted_mask" in new_labels:
            new_labels["express_sorted_mask"] = torch.stack(new_labels["express_sorted_mask"], dim=0)
        if "gene_mask" in new_labels:
            new_labels["gene_mask"] = torch.stack(new_labels["gene_mask"], dim=0)
        return nucleotide_input_ids, nucleotide_input_embeds, \
            nucleotide_attention_mask, \
            gene_input_ids, gene_type_ids, express_input_ids, \
            express_sorted_position_ids, new_labels

    def __call__(self, raw_batch: list[dict]):
        batch_size = len(raw_batch)
        sample_ids = []
        sample_types = []
        sample_gene_id_list = []
        sample_gene_seq_list = []
        sample_gene_embedding_list = []
        sample_gene_express_list = []
        sample_gene_express_list_raw = []
        for item in raw_batch:
            sample_ids.append(item["obj_id"])
            sample_types.append(item["obj_type"])
            if self.alphabet.alphabet_type == "nucleotide":
                if self.nucleotide_input_type == "seq":
                    sample_gene_seq_list.append(item["obj_gene_seq_list"])
                    sample_gene_embedding_list = None
                else:
                    sample_gene_seq_list = None
                    sample_gene_embedding_list.append(item["obj_gene_embedding_list"])
                sample_gene_id_list = None
            else:
                sample_gene_seq_list = None
                sample_gene_embedding_list = None
                sample_gene_id_list.append(item["obj_gene_id_list"])
            sample_gene_express_list.append(item["obj_gene_express_list"])
            sample_gene_express_list_raw.append(item["obj_gene_express_list_raw"])

        nucleotide_input_ids, nucleotide_input_embeds, nucleotide_attention_mask, gene_input_ids, gene_type_ids, express_input_ids, express_sorted_position_ids, labels = self.__call_single__(
            batch_size,
            sample_types,
            sample_gene_id_list,
            sample_gene_seq_list,
            sample_gene_embedding_list,
            sample_gene_express_list,
            sample_gene_express_list_raw
        )
        return {
            "sample_ids": sample_ids,
            "nucleotide_input_ids": nucleotide_input_ids,
            "nucleotide_input_embeds": nucleotide_input_embeds,
            "gene_input_ids": gene_input_ids,
            "gene_type_ids": gene_type_ids,
            "express_input_ids": express_input_ids,
            "express_sorted_position_ids": express_sorted_position_ids,
            "labels": labels
        }