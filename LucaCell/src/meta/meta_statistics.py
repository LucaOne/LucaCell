#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/6/12 13:59
@project: LucaCell
@file: meta_statistics
@desc: xxxx
"""
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from file_operator import fasta_reader
except ImportError:
    from src.file_operator import fasta_reader


filepath = "../../meta/hg38.ensGene_genes.fa"

seq_lens = []
for row in fasta_reader(filepath):
    seq_id, seq = row[0], row[1].strip().upper()
    if seq_id[0] == ">":
        seq_id = seq_id[1:]
    seq_len = len(seq)
    seq_lens.append(seq_len)
seq_lens = [v for v in seq_lens if v <= 20000]
print("seq_len stats:")
print("min: %d, max: %d, mean: %f, median: %d, 25: %d, 45: %d, 60: %d, 75: %d, 80: %d, 85: %d, 90: %d, 95: %d, 99: %d" %(
    np.min(seq_lens),
    np.max(seq_lens),
    np.mean(seq_lens),
    np.median(seq_lens),
    np.percentile(seq_lens, 25),
    np.percentile(seq_lens, 45),
    np.percentile(seq_lens, 60),
    np.percentile(seq_lens, 75),
    np.percentile(seq_lens, 80),
    np.percentile(seq_lens, 85),
    np.percentile(seq_lens, 90),
    np.percentile(seq_lens, 95),
    np.percentile(seq_lens, 99)
))

plt.figure(figsize=(20, 10))

# 绘制箱线图
sns.boxplot(x=seq_lens, color='lightgreen')

# 添加标题和标签
plt.title('Seq Len Stats')
plt.xlabel('Seq Len')
plt.savefig("../../pics/hg38.ensGene_genes_seq_len_stats.png", dpi=600)

filepath = "../../meta/mm10.ensGene_genes.fa"

seq_lens = []
for row in fasta_reader(filepath):
    seq_id, seq = row[0], row[1].strip().upper()
    if seq_id[0] == ">":
        seq_id = seq_id[1:]
    seq_len = len(seq)
    seq_lens.append(seq_len)
seq_lens = [v for v in seq_lens if v <= 20000]
print("seq_len stats:")
print("min: %d, max: %d, mean: %f, median: %d, 25: %d, 45: %d, 60: %d, 75: %d, 80: %d, 85: %d, 90: %d, 95: %d, 99: %d" %(
    np.min(seq_lens),
    np.max(seq_lens),
    np.mean(seq_lens),
    np.median(seq_lens),
    np.percentile(seq_lens, 25),
    np.percentile(seq_lens, 45),
    np.percentile(seq_lens, 60),
    np.percentile(seq_lens, 75),
    np.percentile(seq_lens, 80),
    np.percentile(seq_lens, 85),
    np.percentile(seq_lens, 90),
    np.percentile(seq_lens, 95),
    np.percentile(seq_lens, 99)
))

plt.figure(figsize=(20, 10))

# 绘制箱线图
sns.boxplot(x=seq_lens, color='lightgreen')

# 添加标题和标签
plt.title('Seq Len Stats')
plt.xlabel('Seq Len')
plt.savefig("../../pics/mm10.ensGene_genes_seq_len_stats.png", dpi=600)
'''
seq_len stats:
min: 8, max: 267372, mean: 1751.443473, median: 847, 25: 358, 45: 703, 60: 1266, 75: 2379, 80: 2870, 85: 3551, 90: 4485, 95: 6004, 99: 10275
seq_len stats:
min: 10, max: 106824, mean: 1935.474051, median: 1196, 25: 470, 45: 994, 60: 1754, 75: 2799, 80: 3233, 85: 3775, 90: 4505, 95: 5807, 99: 9408

cut <= 200000
seq_len stats:
min: 8, max: 19984, mean: 1724.937775, median: 846, 25: 358, 45: 701, 60: 1264, 75: 2372, 80: 2862, 85: 3545, 90: 4469, 95: 5972, 99: 10029
seq_len stats:
min: 10, max: 19823, mean: 1922.572518, median: 1195, 25: 470, 45: 993, 60: 1753, 75: 2797, 80: 3229, 85: 3771, 90: 4498, 95: 5790, 99: 9292
'''