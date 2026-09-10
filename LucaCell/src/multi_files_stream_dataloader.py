#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/21 10:54
@project: LucaOne
@file: multi_files_stream_dataloader
@desc: file stream for LucaOne
'''
import os
import random
import numpy as np
from file_operator import *
csv.field_size_limit(sys.maxsize)


class MultiFilesStreamLoader(object):
    def __init__(
            self,
            filepaths,
            batch_size,
            buffer_size,
            parse_row_func,
            batch_data_func,
            dataset_type="train",
            header=True,
            shuffle=False,
            seed=1221
    ):
        if buffer_size % batch_size != 0:
            raise Exception("buffer_size must be evenly div by batch_size")
        if isinstance(filepaths, str):
            if os.path.isdir(filepaths):
                new_filepaths = []
                for filename in os.listdir(filepaths):
                    if not filename.startswith("."):
                        new_filepaths.append(os.path.join(filepaths, filename))
                filepaths = new_filepaths
            else:
                filepaths = [filepaths]
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.parse_row_func = parse_row_func
        self.batch_data_func = batch_data_func
        self.global_nucleotide_seqs = {}
        self.shuffle = shuffle
        self.dataset_type = dataset_type
        if self.dataset_type == "train":
            self.shuffle = True
        if self.shuffle:
            for _ in range(10):
                random.shuffle(self.filepaths)
        self.header = header
        self.rnd = np.random.RandomState(seed)
        self.ptr = 0  # cur index of buffer
        self.total_filename_num = len(self.filepaths)
        print("Total Filename Num: %d" % self.total_filename_num )
        self.cur_file_idx = 0
        print("FilePath: %s" % self.filepaths[self.cur_file_idx % self.total_filename_num])
        self.cur_fin = file_reader(
            self.filepaths[self.cur_file_idx % self.total_filename_num],
            header_filter=True,
            header=self.header
        )
        # memory buffer
        self.buffer = []
        self.enough_flag = False
        self.epoch_over = False
        self.reload_buffer()

    def next_file_reader(self):
        # self.cur_fin.close()
        self.cur_file_idx += 1
        self.cur_fin = file_reader(
            self.filepaths[self.cur_file_idx % self.total_filename_num],
            header_filter=True,
            header=self.header
        )

    def reset_file_reader(self):
        # self.cur_fin.close()
        self.cur_file_idx = 0
        self.cur_fin = file_reader(
            self.filepaths[self.cur_file_idx % self.total_filename_num],
            header_filter=True,
            header=self.header
        )

    def read_one_line(self):
        try:
            row = self.cur_fin.__next__()
            if len(row) == 5:
                sample_id, sample_type, gene_id_list, gene_express_list, non_express_gene_id_list = row[0], row[1], row[2], row[3], row[4]
            else:
                sample_id, sample_type, gene_id_list, gene_express_list, non_express_gene_id_list = row[0], row[1], row[4], row[5], row[6]
            gene_id_list = eval(gene_id_list)
            gene_express_list = eval(gene_express_list)
            non_express_gene_id_list = eval(non_express_gene_id_list)
            return {
                "obj_id": sample_id,
                "obj_type": sample_type,
                "obj_gene_id_list": gene_id_list,
                "obj_gene_express_list": gene_express_list,
                "obj_non_express_gene_id_list": non_express_gene_id_list
            }
        except Exception as e:
            print(e)
            return None

    def encode_line(self, line):
        return self.parse_row_func(
            line["obj_id"],
            line["obj_type"],
            line["obj_gene_id_list"],
            line["obj_gene_express_list"],
            line["obj_non_express_gene_id_list"]
        )

    def reload_buffer(self):
        self.buffer = []
        self.ptr = 0
        ct = 0  # number of lines read
        while ct < self.buffer_size:
            line = self.read_one_line()
            # print("self.cur_file_idx :%d" % self.cur_file_idx )
            # cur file over
            if line is None or len(line) == 0:
                # read next file
                if self.cur_file_idx < self.total_filename_num - 1:
                    self.next_file_reader()
                    # line = self.read_one_line()
                    # self.buffer.append(self.encode_line(line))
                    # ct += 1
                else: # reset
                    # print("file index %d" % (self.cur_file_idx % self.total_filename_num), end="", flush=True)
                    # one epoch over(all files readed)
                    self.epoch_over = True
                    # next epoch
                    self.reset_file_reader()
                    break
            else:
                # done one line
                self.buffer.append(self.encode_line(line))
                ct += 1
        if not self.enough_flag and self.buffer_size == len(self.buffer):
            self.enough_flag = True
        if self.shuffle:
            for _ in range(5):
                self.rnd.shuffle(self.buffer)  # in-place

    def get_batch(self, start, end):
        '''
        :param start:
        :param end:
        :return:
        '''
        cur_batch = self.buffer[start:end]
        batch_input = self.batch_data_func(cur_batch)
        return batch_input

    def __iter__(self):
        return self

    def __next__(self):
        '''
        next batch
        :return:
        '''
        if self.enough_flag:
            if self.epoch_over and self.ptr < len(self.buffer):
                start = self.ptr
                end = min(len(self.buffer), self.ptr + self.batch_size)
                self.ptr = end
                return self.get_batch(start, end)
            elif self.epoch_over:
                # init for next epoch
                self.reload_buffer()
                self.epoch_over = False
                raise StopIteration
            elif self.ptr + self.batch_size > len(self.buffer):  # less than a batch
                start = self.ptr
                end = len(self.buffer)
                batch_input = self.get_batch(start, end)
                self.reload_buffer()
                return batch_input
            else:
                # more than batch
                start = self.ptr
                end = self.ptr + self.batch_size
                batch_input = self.get_batch(start, end)
                self.ptr += self.batch_size
                if self.ptr == len(self.buffer):
                    self.reload_buffer()
                return batch_input
        else:
            if self.ptr < len(self.buffer):
                start = self.ptr
                end = min(len(self.buffer), self.ptr + self.batch_size)
                self.ptr = end
                # print("ok1:", self.epoch_over, self.ptr, len(self.buffer))
                return self.get_batch(start, end)
            else:
                # init for next epoch only for train dataset
                if self.dataset_type == "train":
                    self.reload_buffer()
                raise StopIteration



