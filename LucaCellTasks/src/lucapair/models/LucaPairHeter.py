#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/21 17:32
@project: LucaOnePlusTasks
@file: LucaPairHeter
@desc: LucaPairHeter for heterogeneous double sequence
'''

import sys
import logging
from abc import ABC

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../src")
try:
    from utils import *
    from common.pooling import *
    from common.loss import *
    from common.multi_label_metrics import *
    from common.modeling_bert import BertPreTrainedModel
    from common.modeling_transformer import LucaTransformer
    from common.metrics import *
except ImportError:
    from src.utils import *
    from src.common.pooling import *
    from src.common.loss import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *
    from src.common.modeling_bert import BertPreTrainedModel
    from src.common.modeling_transformer import LucaTransformer
logger = logging.getLogger(__name__)


class LucaPairHeter(BertPreTrainedModel, ABC):
    def __init__(self, config, args):
        super(LucaPairHeter, self).__init__(config)
        # gene_vs_matrix, matrix_vs_gene, matrix_vs_matrix, vector_vs_matrix, matrix_vs_vector, vector_vs_vector
        self.input_type = args.input_type
        self.loss_type = args.loss_type
        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        if self.task_level_type not in ["cell_level"]:
            assert self.fusion_type == "add"

        self.matrix_encoder_a, self.matrix_pooler_a = None, None
        self.matrix_encoder_b, self.matrix_pooler_b = None, None
        self.encoder_type_list_a = [False, False]
        self.input_size_list_a = [0, 0]
        self.linear_idx_a = [-1, -1]
        self.encoder_type_list_b = [False, False]
        self.input_size_list_b = [0, 0]
        self.linear_idx_b = [-1, -1]

        if self.input_type in ["matrix_vs_matrix", "gene_vs_matrix", "matrix_vs_gene"]:
            # emb matrix -> (encoder) - > (pooler) -> fc * -> classifier
            if args.matrix_encoder:
                matrix_encoder_config_a = copy.deepcopy(config)
                matrix_encoder_config_a.no_position_embeddings = True
                matrix_encoder_config_a.no_token_type_embeddings = True
                matrix_encoder_config_a.embedding_input_size = config.embedding_input_size_a
                matrix_encoder_config_a.max_position_embeddings = config.cell_max_length_a
                matrix_encoder_config_b = copy.deepcopy(config)
                matrix_encoder_config_b.no_position_embeddings = True
                matrix_encoder_config_b.no_token_type_embeddings = True
                matrix_encoder_config_b.embedding_input_size = config.embedding_input_size_b
                matrix_encoder_config_b.max_position_embeddings = config.cell_max_length_b
                if args.matrix_encoder_act:
                    self.matrix_encoder_a = nn.ModuleList([
                        nn.Linear(config.embedding_input_size_a, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config_a, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                    self.matrix_encoder_b = nn.ModuleList([
                        nn.Linear(config.embedding_input_size_b, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config_b, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])

                else:
                    self.matrix_encoder_a = nn.ModuleList([
                        nn.LayerNorm(config.embedding_input_size_a, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config_a, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                    self.matrix_encoder_b = nn.ModuleList([
                        nn.LayerNorm(config.embedding_input_size_b, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config_b, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["cell_level"]:
                    self.matrix_pooler_a = create_pooler(pooler_type="matrix", config=config, args=args)
                    self.matrix_pooler_b = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list_a[1] = config.embedding_input_size
                self.input_size_list_b[1] = config.embedding_input_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list_a[1] = config.embedding_input_size_a
                self.input_size_list_b[1] = config.embedding_input_size_b
                if self.task_level_type in ["cell_level"]:
                    ori_embedding_input_size = config.embedding_input_size
                    config.embedding_input_size = config.embedding_input_size_a
                    self.matrix_pooler_a = create_pooler(pooler_type="matrix", config=config, args=args)
                    config.embedding_input_size = config.embedding_input_size_b
                    self.matrix_pooler_b = create_pooler(pooler_type="matrix", config=config, args=args)
                    config.embedding_input_size = ori_embedding_input_size
            self.encoder_type_list_a[1] = True
            self.encoder_type_list_b[1] = True
            self.linear_idx_a[1] = 0
            self.linear_idx_b[1] = 0

            self.input_size_list_b[0] = config.embedding_input_size_b
            self.encoder_type_list_b[0] = True
            self.linear_idx_b[0] = 0
        elif self.input_type == "vector_vs_matrix":
            # emb vector -> fc * -> classifier
            self.input_size_list_a[0] = config.embedding_input_size_a
            self.encoder_type_list_a[0] = True
            self.linear_idx_a[0] = 0

            # matrix
            if args.matrix_encoder:
                matrix_encoder_config_b = copy.deepcopy(config)
                matrix_encoder_config_b.no_position_embeddings = True
                matrix_encoder_config_b.no_token_type_embeddings = True
                matrix_encoder_config_b.max_position_embeddings = config.cell_max_length_b
                if args.matrix_encoder_act:
                    self.matrix_encoder_b = nn.ModuleList([
                        nn.Linear(config.embedding_input_size_b, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config_b, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                else:
                    self.matrix_encoder_b = nn.ModuleList([
                        nn.LayerNorm(config.embedding_input_size_b, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config_b, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                ori_embedding_input_size = config.embedding_input_size_b
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["cell_level"]:
                    self.matrix_pooler_b = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list_b[1] = config.hidden_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list_b[1] = config.embedding_input_size_b
                if self.task_level_type in ["cell_level"]:
                    ori_embedding_input_size = config.embedding_input_size
                    config.embedding_input_size = config.embedding_input_size_b
                    self.matrix_pooler_b = create_pooler(pooler_type="matrix", config=config, args=args)
                    config.embedding_input_size = ori_embedding_input_size
            self.encoder_type_list_b[1] = True
            self.linear_idx_b[1] = 1
        elif self.input_type == "matrix_vs_vector":
            # matrix
            if args.matrix_encoder:
                matrix_encoder_config_a = copy.deepcopy(config)
                matrix_encoder_config_a.no_position_embeddings = True
                matrix_encoder_config_a.no_token_type_embeddings = True
                matrix_encoder_config_a.max_position_embeddings = config.cell_max_length_a
                if args.matrix_encoder_act:
                    self.matrix_encoder_a = nn.ModuleList([
                        nn.Linear(config.embedding_input_size_a, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config_a, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                else:
                    self.matrix_encoder_a = nn.ModuleList([
                        nn.LayerNorm(config.embedding_input_size_a, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config_a, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                ori_embedding_input_size = config.embedding_input_size_a
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["cell_level"]:
                    self.matrix_pooler_a = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list_a[1] = config.hidden_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list_a[1] = config.embedding_input_size_a
                if self.task_level_type in ["cell_level"]:
                    ori_embedding_input_size = config.embedding_input_size
                    config.embedding_input_size = config.embedding_input_size_a
                    self.matrix_pooler_a = create_pooler(pooler_type="matrix", config=config, args=args)
                    config.embedding_input_size = ori_embedding_input_size
            self.encoder_type_list_a[1] = True
            self.linear_idx_a[1] = 1

            # emb vector -> fc * -> classifier
            self.input_size_list_b[0] = config.embedding_input_size_a
            self.encoder_type_list_b[0] = True
            self.linear_idx_b[0] = 0
        elif self.input_type == "vector_vs_vector":
            # emb vector -> fc * -> classifier
            self.input_size_list_a[0] = config.embedding_input_size_a
            self.encoder_type_list_a[0] = True
            self.linear_idx_a[0] = 0
            # emb vector -> fc * -> classifier
            self.input_size_list_b[0] = config.embedding_input_size_b
            self.encoder_type_list_b[0] = True
            self.linear_idx_b[0] = 0
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        fc_size_list = [config.vector_fc_size, config.matrix_fc_size]
        all_linear_list_a = [None, None]
        all_linear_list_b = [None, None]
        self.output_size_a = [0, 0]
        self.output_size_b = [0, 0]
        print("self.encoder_type_list:", self.encoder_type_list_a)
        print("self.encoder_type_list:", self.encoder_type_list_b)
        for encoder_idx, encoder_flag in enumerate(self.encoder_type_list_a):
            if not encoder_flag:
                continue
            fc_size = fc_size_list[encoder_idx]
            input_size_a = self.input_size_list_a[encoder_idx]
            input_size_b = self.input_size_list_b[encoder_idx]
            print("encoder_idx:", encoder_idx, ", input_size_a:", input_size_a, ", input_size_b:", input_size_b)
            if fc_size is not None and len(fc_size) > 0:
                if isinstance(fc_size, list):
                    fc_size = [int(v) for v in fc_size]
                else:
                    fc_size = [int(fc_size)]
                linear_list_a = []
                linear_list_b = []
                for idx in range(len(fc_size)):
                    linear_a = nn.Linear(input_size_a, fc_size[idx])
                    linear_list_a.append(linear_a)
                    linear_list_a.append(create_activate(config.fc_activate_func))
                    input_size_a = fc_size[idx]
                    linear_b = nn.Linear(input_size_b, fc_size[idx])
                    linear_list_b.append(linear_b)
                    linear_list_b.append(create_activate(config.fc_activate_func))
                    input_size_b = fc_size[idx]
                all_linear_list_a[encoder_idx] = nn.ModuleList(linear_list_a)
                all_linear_list_b[encoder_idx] = nn.ModuleList(linear_list_b)
                self.output_size_a[encoder_idx] = fc_size[-1]
                self.output_size_b[encoder_idx] = fc_size[-1]
            else:
                # 没有全连接层
                self.linear_idx_a[encoder_idx] = -1
                self.linear_idx_b[encoder_idx] = -1
                self.output_size_a[encoder_idx] = input_size_a
                self.output_size_b[encoder_idx] = input_size_b
        all_linear_list_a = [linear for linear in all_linear_list_a if linear is not None]
        all_linear_list_b = [linear for linear in all_linear_list_b if linear is not None]
        if all_linear_list_a is not None and len(all_linear_list_a) > 0:
            self.linear_a = nn.ModuleList(all_linear_list_a)
        if all_linear_list_b is not None and len(all_linear_list_b) > 0:
            self.linear_b = nn.ModuleList(all_linear_list_b)
        if self.fusion_type == "add":
            output_size_a = [v for v in self.output_size_a if v > 0]
            assert len(set(output_size_a)) == 1
            last_hidden_size_a = output_size_a[0]
            output_size_b = [v for v in self.output_size_b if v > 0]
            assert len(set(output_size_b)) == 1
            last_hidden_size_b = output_size_b[0]
        else:
            last_hidden_size_a = sum(self.output_size_a)
            last_hidden_size_b = sum(self.output_size_b)
        last_hidden_size = last_hidden_size_a + last_hidden_size_b
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=last_hidden_size,
                classifier_size=args.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )
        self.post_init()

    def __forward_a__(
            self,
            vectors,
            matrices,
            matrix_attention_masks,
            **kwargs
    ):
        sample_ids = kwargs["sample_ids"] if "sample_ids" in kwargs else None
        prefix = kwargs["prefix"] if "prefix" in kwargs else None
        attention_pooling_scores_savepath = kwargs["attention_pooling_scores_savepath"] if "attention_pooling_scores_savepath" in kwargs else None

        if self.input_type in ["vector_vs_matrix", "vector_vs_vector"]:
            vector_vector = vectors
            vector_linear_idx = self.linear_idx_a[0]
            if vector_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_a[vector_linear_idx]):
                    vector_vector = layer_module(vector_vector)
        else:
            if self.matrix_encoder_a is not None:
                for module in self.matrix_encoder_a[:-1]:
                    matrices = module(matrices)
                matrices_output = self.matrix_encoder_a[-1](
                    input_ids=None,
                    attention_mask=matrices,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=matrix_attention_masks,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False
                )
                matrices = matrices_output[0]
            if self.matrix_pooler_a is not None:
                matrix_vector = self.matrix_pooler_a(
                    matrices,
                    mask=matrix_attention_masks,
                    sample_ids=sample_ids,
                    prefix=prefix,
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                )
            elif self.task_level_type in ["cell_level"]:
                tmp_mask = torch.unsqueeze(matrix_attention_masks, dim=-1)
                matrices = matrices.masked_fill(tmp_mask == 0, 0.0)
                # 均值pooling
                matrix_vector = torch.sum(matrices, dim=1)/(torch.sum(tmp_mask, dim=1) + 1e-12)
            else:
                matrix_vector = matrices
            matrix_linear_idx = self.linear_idx_a[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_a[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)

        if self.input_type in ["vector_vs_matrix", "vector_vs_vector"]:
            repr_vector = vector_vector
        elif self.input_type in ["matrix_vs_matrix", "matrix_vs_vector", "gene_vs_matrix", "matrix_vs_gene"]:
            repr_vector = matrix_vector
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        return repr_vector

    def __forward_b__(
            self,
            vectors,
            matrices,
            matrix_attention_masks,
            **kwargs
    ):
        sample_ids = kwargs["sample_ids"] if "sample_ids" in kwargs else None
        prefix = kwargs["prefix"] if "prefix" in kwargs else None
        attention_pooling_scores_savepath = kwargs["attention_pooling_scores_savepath"] if "attention_pooling_scores_savepath" in kwargs else None

        if self.input_type in ["matrix_vs_vector", "vector_vs_vector"]:
            vector_vector = vectors
            vector_linear_idx = self.linear_idx_b[0]
            if vector_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_b[vector_linear_idx]):
                    vector_vector = layer_module(vector_vector)
        else:
            if self.matrix_encoder_b is not None:
                for module in self.matrix_encoder_b[:-1]:
                    matrices = module(matrices)
                matrices_output = self.matrix_encoder_b[-1](
                    input_ids=None,
                    attention_mask=matrix_attention_masks,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=matrices,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False
                )
                matrices = matrices_output[0]
            if self.matrix_pooler_b is not None:
                matrix_vector = self.matrix_pooler_b(
                    matrices,
                    mask=matrix_attention_masks,
                    sample_ids=sample_ids,
                    prefix=prefix,
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                )
            elif self.task_level_type in ["cell_level"]:
                tmp_mask = torch.unsqueeze(matrix_attention_masks, dim=-1)
                matrices = matrices.masked_fill(tmp_mask == 0, 0.0)
                # 均值pooling
                matrix_vector = torch.sum(matrices, dim=1)/(torch.sum(tmp_mask, dim=1) + 1e-12)
            else:
                matrix_vector = matrices
            matrix_linear_idx = self.linear_idx_b[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_b[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)

        if self.input_type in ["matrix_vs_vector", "vector_vs_vector"]:
            repr_vector = vector_vector
        elif self.input_type in ["matrix_vs_matrix", "matrix_vs_gene", "gene_vs_matrix", "vector_vs_matrix"]:
            repr_vector = matrix_vector
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        return repr_vector

    def forward(
            self,
            gene_seqs_vector_input_a=None,
            gene_seqs_vector_input_b=None,
            gene_seqs_vector_attention_masks_a=None,
            gene_seqs_vector_attention_masks_b=None,
            cell_encoded_vectors_a=None,
            cell_encoded_vectors_b=None,
            cell_encoded_matrices_a=None,
            cell_encoded_matrices_b=None,
            cell_matrix_attention_masks_a=None,
            cell_matrix_attention_masks_b=None,
            labels=None,
            **kwargs
    ):
        sample_ids = kwargs["sample_ids"] if "sample_ids" in kwargs else None
        attention_pooling_scores_savepath = kwargs["attention_pooling_scores_savepath"] if "attention_pooling_scores_savepath" in kwargs else None
        output_classification_vector_dirpath = kwargs["output_classification_vector_dirpath"] if "output_classification_vector_dirpath" in kwargs else None
        if self.input_type == "gene_vs_matrix":
            representation_vector_a = self.__forward_a__(
                None,
                gene_seqs_vector_input_a,
                gene_seqs_vector_attention_masks_a,
                sample_ids=sample_ids,
                prefix="gene_a",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
            representation_vector_b = self.__forward_b__(
                None,
                cell_encoded_matrices_b,
                cell_matrix_attention_masks_b,
                sample_ids=sample_ids,
                prefix="cell_b",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
        elif self.input_type == "vector_vs_matrix":
            representation_vector_a = self.__forward_a__(
                cell_encoded_vectors_a,
                None,
                None,
                sample_ids=sample_ids,
                prefix="cell_a",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
            representation_vector_b = self.__forward_b__(
                None,
                cell_encoded_matrices_b,
                cell_matrix_attention_masks_b,
                sample_ids=sample_ids,
                prefix="cell_b",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
        elif self.input_type == "matrix_vs_gene":
            representation_vector_a = self.__forward_a__(
                None,
                cell_encoded_matrices_a,
                cell_matrix_attention_masks_a,
                sample_ids=sample_ids,
                prefix="cell_a",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
            representation_vector_b = self.__forward_b__(
                None,
                gene_seqs_vector_input_b,
                gene_seqs_vector_attention_masks_b,
                sample_ids=sample_ids,
                prefix="gene_b",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
        elif self.input_type == "matrix_vs_vector":
            representation_vector_a = self.__forward_a__(
                None,
                cell_encoded_matrices_a,
                cell_matrix_attention_masks_a,
                sample_ids=sample_ids,
                prefix="cell_a",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
            representation_vector_b = self.__forward_b__(
                cell_encoded_vectors_b,
                None,
                None,
                sample_ids=sample_ids,
                prefix="cell_b",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
        elif self.input_type == "vector_vs_vector":
            representation_vector_a = self.__forward_a__(
                cell_encoded_vectors_a,
                None,
                None,
                sample_ids=sample_ids,
                prefix="cell_a",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
            representation_vector_b = self.__forward_b__(
                cell_encoded_vectors_b,
                None,
                None,
                sample_ids=sample_ids,
                prefix="cell_b",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
        elif self.input_type == "matrix_vs_matrix":
            representation_vector_a = self.__forward_a__(
                None,
                cell_encoded_matrices_a,
                cell_matrix_attention_masks_a,
                sample_ids=sample_ids,
                prefix="cell_a",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
            representation_vector_b = self.__forward_b__(
                None,
                cell_encoded_matrices_b,
                cell_matrix_attention_masks_b,
                sample_ids=sample_ids,
                prefix="cell_b",
                attention_pooling_scores_savepath=attention_pooling_scores_savepath
            )
        else:
            raise Exception("Not support input_type=%s" % self.input_type)

        concat_vector = torch.cat([representation_vector_a, representation_vector_b], dim=-1)
        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)
        if self.hidden_layer is not None:
            concat_vector = self.hidden_layer(concat_vector)
        if self.hidden_act is not None:
            concat_vector = self.hidden_act(concat_vector)
        logits = self.classifier(concat_vector)
        if self.output:
            output = self.output(logits)
        else:
            output = logits
        outputs = [logits, output]

        if output_classification_vector_dirpath and sample_ids:
            for sample_idx, sample_id in enumerate(sample_ids):
                output_classification_vector = {
                    "representation_vector_a": representation_vector_a[sample_idx].detach().cpu(),
                    "representation_vector_b": representation_vector_b[sample_idx].detach().cpu(),
                    "classification_vector": concat_vector[sample_idx].detach().cpu(),
                }
                filepath = os.path.join(output_classification_vector_dirpath, "%s_classification_vector.pt" % sample_id)
                torch.save(output_classification_vector, filepath)

        if labels is not None:
            if self.output_mode in ["regression"]:
                if self.task_level_type not in ["cell_level"] and self.loss_reduction == "meanmean":
                    # logits: N, seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels)
                elif self.loss_type == "dynamic_weighted_mse":
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.output_mode in ["multi_label", "multi-label"]:
                if self.loss_reduction == "meanmean":
                    # logits: N , label_size
                    # labels: N , label_size
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N , label_size
                    # labels: N , label_size
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["cell_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N * seq_len * 1
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                if self.task_level_type not in ["cell_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, label_size
                    # labels: N , seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len, label_size
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, *outputs]
        return outputs



