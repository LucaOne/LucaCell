#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/21 16:10
@project: LucaOne
@file: encoder
@desc: encoder
'''
import os.path
import random
import sys
import numpy
import torch
sys.path.append(".")
sys.path.append("../")
sys.path.append("../src")
try:
    from models.alphabet import Alphabet
    from file_operator import fasta_reader, csv_reader
    from utils import calc_emb_filename_by_seq_id
    from topk_min_heap import TopKMinHeap
except ImportError:
    from src.models.alphabet import Alphabet
    from src.file_operator import fasta_reader, csv_reader
    from src.utils import calc_emb_filename_by_seq_id
    from src.topk_min_heap import TopKMinHeap


class Encoder(object):
    def __init__(
            self,
            config,
            nucleotide_input_type,
            tokenizer: Alphabet,
            max_nucleotide_seq_length,
            max_gene_seq_length,
            ignore_index,
            global_nucleotide_seqs_filepath,
            global_nucleotide_seq_embeddings_dirpath,
            global_gene_positions_filepath,
            embedding_buffer_size=None
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_nucleotide_seq_length = max_nucleotide_seq_length
        self.max_gene_seq_length = max_gene_seq_length
        self.ignore_index = ignore_index
        self.nucleotide_input_type = nucleotide_input_type
        assert "add_special_tokens" in self.config
        self.add_special_tokens = self.config["add_special_tokens"]
        assert "truncation" in self.config
        self.truncation = self.config["truncation"]
        self.global_nucleotide_seqs_filepath = global_nucleotide_seqs_filepath.split(";")
        self.global_nucleotide_seq_embeddings_dirpath = global_nucleotide_seq_embeddings_dirpath.split(";")
        self.global_gene_positions_filepath = global_gene_positions_filepath.split(";")
        self.global_gene_positions = {}
        if self.nucleotide_input_type == "seq":
            self.global_nucleotide_seqs = {}
            self.load_global_nucleotide_seqs()
        else:
            self.embedding_buffer_size = embedding_buffer_size
            assert self.embedding_buffer_size is not None and self.embedding_buffer_size > 0
            print("embedding_buffer_size: %d" % self.embedding_buffer_size)
            self.global_nucleotide_seq_embeddings = {}
            self.access_times = TopKMinHeap(self.embedding_buffer_size)
            self.load_global_nucleotide_seq_embeddings()
        self.load_global_gene_positions()

    def load_global_gene_positions(self):
        for filepath in self.global_gene_positions_filepath:
            for row in csv_reader(filepath):
                gene_id, position = row[0], int(row[1])
                if gene_id[0] == ">":
                    gene_id = gene_id[1:]
                self.global_gene_positions[gene_id] = position

    def load_global_nucleotide_seqs(self):
        for filepath in self.global_nucleotide_seqs_filepath:
            for row in fasta_reader(filepath):
                seq_id, seq = row[0], row[1].strip().upper()
                if seq_id[0] == ">":
                    seq_id = seq_id[1:]
                self.global_nucleotide_seqs[seq_id] = seq

    def load_global_nucleotide_seq_embeddings(self):
        if self.global_nucleotide_seq_embeddings_dirpath:
            loaded_over = False
            loaded_num = 0
            max_nucleotide_seq_length = self.max_nucleotide_seq_length
            if self.add_special_tokens:
                max_nucleotide_seq_length = max_nucleotide_seq_length - 2
            for seq_filepath in self.global_nucleotide_seqs_filepath:
                for row in fasta_reader(seq_filepath):
                    seq_id = row[0]
                    if seq_id[0] == ">":
                        seq_id = seq_id[1:]
                    embedding_filename = calc_emb_filename_by_seq_id(
                        seq_id,
                        "matrix" if self.nucleotide_input_type == "embedding_matrix" else "vector"
                    )
                    for emb_dir in self.global_nucleotide_seq_embeddings_dirpath:
                        embedding_filepath = os.path.join(
                            emb_dir,
                            embedding_filename
                        )
                        if os.path.exists(embedding_filepath):
                            emb = torch.load(embedding_filepath, weights_only=True)
                            if isinstance(emb, numpy.ndarray):
                                emb = torch.from_numpy(emb)
                            if self.nucleotide_input_type == "embedding_matrix":
                                if emb.shape[0] > max_nucleotide_seq_length + 2:
                                    if self.truncation == "left":
                                        emb = torch.cat([emb[0:1, :], emb[- max_nucleotide_seq_length - 1:, :]], dim=0)
                                    else:
                                        emb = torch.cat([emb[0:max_nucleotide_seq_length + 1, :], emb[-1:, :]], dim=0)
                            removed_element = self.access_times.insert(seq_id, 0)
                            if removed_element:
                                del self.global_nucleotide_seq_embeddings[removed_element.seq_id]
                                print("deleted %s" % removed_element.seq_id)
                            self.global_nucleotide_seq_embeddings[seq_id] = emb
                            loaded_num += 1
                            break
                    if self.embedding_buffer_size is not None and loaded_num >= self.embedding_buffer_size:
                        loaded_over = True
                        break
                if loaded_over:
                    break
            print("access_times: %d" % self.access_times.size())

    def get_nucleotide_seq(self, seq_id):
        return self.global_nucleotide_seqs[seq_id]

    def get_nucleotide_seq_embedding(self, seq_id):
        if seq_id in self.global_nucleotide_seq_embeddings:
            self.access_times.update(seq_id, self.access_times.get_access_cnt(seq_id) + 1)
            return self.global_nucleotide_seq_embeddings[seq_id]
        max_nucleotide_seq_length = self.max_nucleotide_seq_length
        if self.add_special_tokens:
            max_nucleotide_seq_length = max_nucleotide_seq_length - 2
        emb_filename = calc_emb_filename_by_seq_id(
            seq_id,
            "matrix" if self.nucleotide_input_type == "embedding_matrix" else "vector"
        )
        emb = None
        for emb_dir in self.global_nucleotide_seq_embeddings_dirpath:
            embedding_filepath = os.path.join(
                emb_dir,
                emb_filename
            )
            if os.path.exists(embedding_filepath):
                emb = torch.load(embedding_filepath, weights_only=True)
                if isinstance(emb, numpy.ndarray):
                    emb = torch.from_numpy(emb)
                if self.nucleotide_input_type == "embedding_matrix":
                    if emb.shape[0] > max_nucleotide_seq_length + 2:
                        if self.truncation == "left":
                            emb = torch.cat([emb[0:1, :], emb[- max_nucleotide_seq_length - 1:, :]], dim=0)
                        else:
                            emb = torch.cat([emb[0:max_nucleotide_seq_length + 1, :], emb[-1:, :]], dim=0)
                removed_element = self.access_times.insert(seq_id, 1)
                if removed_element:
                    del self.global_nucleotide_seq_embeddings[removed_element.seq_id]
                self.global_nucleotide_seq_embeddings[seq_id] = emb
                if len(self.global_nucleotide_seq_embeddings) >= 5000:
                    print("size: %d" % len(self.global_nucleotide_seq_embeddings))
                break
        return emb

    def encode(
            self,
            obj_id,
            obj_type,
            obj_gene_id_list,
            obj_gene_express_list,
            obj_non_express_gene_id_list,
    ):
        record = {
            "obj_id": obj_id,
            "obj_type": obj_type
        }
        tmp_gene_idx_list = list(range(len(obj_gene_id_list)))
        tmp_non_express_gene_idx_list = list(range(len(obj_non_express_gene_id_list)))
        if len(obj_gene_id_list) <= 1000:
            obj_gene_idx_list_1 = tmp_gene_idx_list
            for _ in range(5):
                random.shuffle(tmp_non_express_gene_idx_list)
            size = min(4 * len(obj_gene_idx_list_1), len(obj_non_express_gene_id_list), 1200 - len(obj_gene_idx_list_1))
            obj_gene_idx_list_2 = tmp_non_express_gene_idx_list[:size]
        else:
            for _ in range(5):
                random.shuffle(tmp_gene_idx_list)
                random.shuffle(tmp_non_express_gene_idx_list)
            obj_gene_idx_list_1 = tmp_gene_idx_list[:1000]
            obj_gene_idx_list_2 = tmp_non_express_gene_idx_list[:(1200 - len(obj_gene_idx_list_1))]
        new_obj_gene_id_list = [
            (obj_gene_id_list[gene_idx], self.global_gene_positions[obj_gene_id_list[gene_idx]]) for gene_idx in obj_gene_idx_list_1
        ] + [
            (obj_non_express_gene_id_list[gene_idx], self.global_gene_positions[obj_non_express_gene_id_list[gene_idx]]) for gene_idx in obj_gene_idx_list_2
        ]
        new_obj_gene_express_list = [(obj_gene_express_list[gene_idx], self.global_gene_positions[obj_gene_id_list[gene_idx]]) for gene_idx in obj_gene_idx_list_1] + [
            ('[E_NON]', self.global_gene_positions[obj_non_express_gene_id_list[gene_idx]]) for gene_idx in obj_gene_idx_list_2
        ]
        obj_gene_id_list = new_obj_gene_id_list
        obj_gene_express_list = new_obj_gene_express_list
        obj_gene_id_list = sorted(obj_gene_id_list, key=lambda x: x[1])
        obj_gene_id_list = [x[0] for x in obj_gene_id_list]
        obj_gene_express_list = sorted(obj_gene_express_list, key=lambda x: x[1])
        obj_gene_express_list = [x[0] for x in obj_gene_express_list]
        if self.nucleotide_input_type == "seq":
            obj_gene_seq_list = [self.get_nucleotide_seq(gene_id) for gene_id in obj_gene_id_list]
            assert len(obj_gene_id_list) == len(obj_gene_express_list) == len(obj_gene_seq_list)
        else:
            obj_gene_embedding_list = [self.get_nucleotide_seq_embedding(gene_id) for gene_id in obj_gene_id_list]
            assert len(obj_gene_id_list) == len(obj_gene_express_list) == len(obj_gene_embedding_list)
        cur_max_gene_seq_length = self.max_gene_seq_length
        if self.add_special_tokens:
            cur_max_gene_seq_length = cur_max_gene_seq_length - 2
        if len(obj_gene_id_list) > cur_max_gene_seq_length:
            if self.truncation == "left":
                obj_gene_id_list = obj_gene_id_list[-cur_max_gene_seq_length:]
                if self.nucleotide_input_type == "seq":
                    obj_gene_seq_list = obj_gene_seq_list[-cur_max_gene_seq_length:]
                else:
                    obj_gene_embedding_list = obj_gene_embedding_list[-cur_max_gene_seq_length:]
                obj_gene_express_list = obj_gene_express_list[-cur_max_gene_seq_length:]
            else:
                obj_gene_id_list = obj_gene_id_list[:cur_max_gene_seq_length]
                if self.nucleotide_input_type == "seq":
                    obj_gene_seq_list = obj_gene_seq_list[:cur_max_gene_seq_length]
                else:
                    obj_gene_embedding_list = obj_gene_embedding_list[:cur_max_gene_seq_length]
                obj_gene_express_list = obj_gene_express_list[:cur_max_gene_seq_length]
        gene_express_list_encoding = self.tokenizer.express_encode(obj_gene_express_list)
        record["obj_gene_express_list"] = gene_express_list_encoding
        record["obj_gene_express_list_raw"] = obj_gene_express_list
        if self.tokenizer.alphabet_type == "nucleotide":
            cur_max_nucleotide_seq_length = self.max_nucleotide_seq_length
            if self.add_special_tokens:
                cur_max_nucleotide_seq_length = cur_max_nucleotide_seq_length - 2
            max_gene_seq_len = 0
            if self.nucleotide_input_type == "seq":
                for gene_seq in obj_gene_seq_list:
                    gene_seq_len = len(gene_seq)
                    if max_gene_seq_len < gene_seq_len:
                        max_gene_seq_len = gene_seq_len
            elif self.nucleotide_input_type == "embedding_matrix":
                for gene_seq_embedding in obj_gene_embedding_list:
                    gene_seq_len = gene_seq_embedding.shape[0] - 2
                    if max_gene_seq_len < gene_seq_len:
                        max_gene_seq_len = gene_seq_len
            else:
                # embedding_vector
                max_gene_seq_len = cur_max_nucleotide_seq_length
            if cur_max_nucleotide_seq_length < max_gene_seq_len:
                if self.nucleotide_input_type == "seq":
                    new_obj_gene_seq_list = []
                    for gene_seq in obj_gene_seq_list:
                        if self.truncation == "left":
                            new_obj_gene_seq_list.append(gene_seq[-cur_max_nucleotide_seq_length:])
                        else:
                            new_obj_gene_seq_list.append(gene_seq[:cur_max_nucleotide_seq_length])
                    obj_gene_seq_list = new_obj_gene_seq_list
                    gene_seq_list_encoding = self.tokenizer.nucleotide_encode(obj_gene_seq_list)
                    record["obj_gene_seq_list"] = gene_seq_list_encoding
                elif self.nucleotide_input_type == "embedding_matrix":
                    new_obj_gene_embedding_list = []
                    for gene_embedding in obj_gene_embedding_list:
                        if self.truncation == "left":
                            new_obj_gene_embedding_list.append(torch.cat([gene_embedding[0:1, :], gene_embedding[- cur_max_nucleotide_seq_length - 1:, :]], dim=0))
                        else:
                            new_obj_gene_embedding_list.append(torch.cat([gene_embedding[0:cur_max_nucleotide_seq_length + 1, :], gene_embedding[-1:, :]], dim=0))
                    obj_gene_embedding_list = new_obj_gene_embedding_list
                    record["obj_gene_embedding_list"] = obj_gene_embedding_list
                else:
                    record["obj_gene_embedding_list"] = obj_gene_embedding_list
            else:
                if self.nucleotide_input_type == "seq":
                    gene_seq_list_encoding = self.tokenizer.nucleotide_encode(obj_gene_seq_list)
                    record["obj_gene_seq_list"] = gene_seq_list_encoding
                else:
                    record["obj_gene_embedding_list"] = obj_gene_embedding_list
        else:
            gene_id_list_encoding = self.tokenizer.gene_tokenize(obj_gene_id_list)
            record["obj_gene_id_list"] = gene_id_list_encoding
        return record
