#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/6/18 10:37
@project: LucaCell
@file: data_statistics
@desc: xxxx
"""
import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from file_operator import csv_reader
except ImportError:
    from src.file_operator import csv_reader

dirpath = "/mnt/luca1/sunyan4/data/cellxgene/binning/"

gene_id_seq_lens = []
for filename in os.listdir(dirpath):
    if not filename.endswith(".csv"):
        continue
    for row in csv_reader(os.path.join(dirpath, filename)):
        gene_id_list = eval(row[4])
        gene_id_seq_lens.append(len(gene_id_list))

print("seq_len stats:")
print("min: %d, max: %d, mean: %f, median: %d, 25: %d, 45: %d, 60: %d, 75: %d, 80: %d, 85: %d, 90: %d, 95: %d, 99: %d" %(
    np.min(gene_id_seq_lens),
    np.max(gene_id_seq_lens),
    np.mean(gene_id_seq_lens),
    np.median(gene_id_seq_lens),
    np.percentile(gene_id_seq_lens, 25),
    np.percentile(gene_id_seq_lens, 45),
    np.percentile(gene_id_seq_lens, 60),
    np.percentile(gene_id_seq_lens, 75),
    np.percentile(gene_id_seq_lens, 80),
    np.percentile(gene_id_seq_lens, 85),
    np.percentile(gene_id_seq_lens, 90),
    np.percentile(gene_id_seq_lens, 95),
    np.percentile(gene_id_seq_lens, 99)
))

plt.figure(figsize=(20, 10))

# 绘制箱线图
sns.boxplot(x=gene_id_seq_lens)

# 添加标题和标签
plt.title('Gene Seq Len Stats')
plt.xlabel('Gene Seq Len')
plt.savefig("../../pics/gene_id_seq_len_stats.png", dpi=600)
