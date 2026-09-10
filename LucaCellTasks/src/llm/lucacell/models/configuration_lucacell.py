#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/6/3 16:04
@project: LucaCell
@file: configuration_lucacell
@desc: LucaCell Configuration
"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LucaCellConfig(PretrainedConfig):

    model_type = "LucaCell"

    def __init__(
            self,
            pretrain_task_name="express_token_mask,express_sorted_mask",
            pretrain_task_weight={"express_token_mask": 1.0, "express_sorted_mask": 1.0},
            alphabet_type="nucleotide",
            embed_dim=2560,
            nucleotide_token_encoder_num_attention_heads=8,
            nucleotide_token_encoder_num_layers=2,
            nucleotide_token_pooling_type="value_attention",
            num_attention_heads=20,
            num_layers=10,
            no_nucleotide_token_embeddings=False,
            no_nucleotide_token_encoder=False,
            no_nucleotide_position_embeddings=True,
            max_nucleotide_position_embeddings=None,
            no_gene_positions_embeddings=True,
            max_gene_position_embeddings=None,
            no_express_sorted_embeddings=False,
            max_express_sorted_position_embeddings=102,
            no_gene_type_embeddings=False,
            ignore_index=-100,
            use_rotary_embeddings=True,
            use_embed_layer_norm=False,
            use_last_layer_norm=True,
            embed_scale=1.0,
            pretrained_model_name="lucaone-gene",
            **kwargs,
    ):
        super().__init__(pad_token_id=None, **kwargs)
        self.pretrain_task_name = pretrain_task_name
        self.pretrain_task_weight = pretrain_task_weight
        self.alphabet_type = alphabet_type
        self.embed_dim = embed_dim
        self.nucleotide_token_encoder_num_attention_heads = nucleotide_token_encoder_num_attention_heads
        self.nucleotide_token_encoder_num_layers = nucleotide_token_encoder_num_layers
        self.nucleotide_token_pooling_type = nucleotide_token_pooling_type
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.no_nucleotide_token_embeddings = no_nucleotide_token_embeddings
        self.no_nucleotide_token_encoder = no_nucleotide_token_encoder
        self.no_nucleotide_position_embeddings = no_nucleotide_position_embeddings
        self.max_nucleotide_position_embeddings = max_nucleotide_position_embeddings
        self.no_gene_positions_embeddings = no_gene_positions_embeddings
        self.max_gene_position_embeddings = max_gene_position_embeddings
        self.no_express_sorted_embeddings = no_express_sorted_embeddings
        self.max_express_sorted_position_embeddings = max_express_sorted_position_embeddings
        self.no_gene_type_embeddings = no_gene_type_embeddings
        self.ignore_index = ignore_index
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_embed_layer_norm = use_embed_layer_norm
        self.use_last_layer_norm = use_last_layer_norm
        self.embed_scale = embed_scale
        self.pretrained_model_name = pretrained_model_name
