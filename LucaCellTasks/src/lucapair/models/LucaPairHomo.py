#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/21 17:32
@project: LucaOnePlusTasks
@file: LucaPairHomo
@desc: LucaPairHomo for homogeneous double sequence
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
    from common.metrics import *
    from common.modeling_bert import BertPreTrainedModel
    from common.modeling_transformer import LucaTransformer
except ImportError:
    from src.utils import *
    from src.common.pooling import *
    from src.common.loss import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *
    from src.common.modeling_bert import BertPreTrainedModel
    from src.common.modeling_transformer import LucaTransformer
logger = logging.getLogger(__name__)


class LucaPairHomo(BertPreTrainedModel, ABC):
    def __init__(self, config, args):
        super(LucaPairHomo, self).__init__(config)
        # vector_vs_vector, matrix_vs_matrix
        if config.cell_max_length is None and config.cell_max_length_a == config.cell_max_length_b:
            config.cell_max_length = config.cell_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b:
            config.embedding_input_size = config.embedding_input_size_a

        self.input_type = args.input_type
        self.loss_type = args.loss_type
        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        if self.task_level_type not in ["cell_level"]:
            assert self.input_type != "vector_vs_vector"
            assert self.fusion_type == "add"

        self.matrix_encoder, self.matrix_pooler = None, None
        self.encoder_type_list = [False, False]
        self.input_size_list = [0, 0]
        self.linear_idx = [-1, -1]
        if self.input_type == "vector_vs_vector":
            # emb vector -> fc * -> classifier
            self.input_size_list[0] = config.embedding_input_size
            self.encoder_type_list[0] = True
            self.linear_idx[0] = 0
        elif self.input_type == "matrix_vs_matrix":
            # emb matrix -> (encoder) - > (pooler) -> fc * -> classifier
            if args.matrix_encoder:
                matrix_encoder_config = copy.deepcopy(config)
                matrix_encoder_config.no_position_embeddings = True
                matrix_encoder_config.no_token_type_embeddings = True
                matrix_encoder_config.max_position_embeddings = config.cell_max_length
                if args.matrix_encoder_act:
                    self.matrix_encoder = nn.ModuleList([
                        nn.Linear(config.embedding_input_size, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])

                else:
                    self.matrix_encoder = nn.ModuleList([
                        nn.LayerNorm(config.embedding_input_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["cell_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list[1] = config.embedding_input_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list[1] = config.embedding_input_size
                if self.task_level_type in ["cell_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
            self.encoder_type_list[1] = True
            self.linear_idx[1] = 0
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        fc_size_list = [config.vector_fc_size, config.matrix_fc_size]
        all_linear_list = [None, None]
        self.output_size = [0, 0]
        print("self.encoder_type_list:", self.encoder_type_list)
        for encoder_idx, encoder_flag in enumerate(self.encoder_type_list):
            if not encoder_flag:
                continue
            fc_size = fc_size_list[encoder_idx]
            input_size = self.input_size_list[encoder_idx]
            print("encoder_idx", encoder_idx, "input_size:", input_size)
            if fc_size is not None and len(fc_size) > 0:
                if isinstance(fc_size, list):
                    fc_size = [int(v) for v in fc_size]
                else:
                    fc_size = [int(fc_size)]
                linear_list = []
                for idx in range(len(fc_size)):
                    linear = nn.Linear(input_size, fc_size[idx])
                    linear_list.append(linear)
                    linear_list.append(create_activate(config.fc_activate_func))
                    input_size = fc_size[idx]
                all_linear_list[encoder_idx] = nn.ModuleList(linear_list)
                self.output_size[encoder_idx] = fc_size[-1]
            else:
                # 没有全连接层
                self.linear_idx[encoder_idx] = -1
                self.output_size[encoder_idx] = input_size
        all_linear_list = [linear for linear in all_linear_list if linear is not None]
        if all_linear_list is not None and len(all_linear_list) > 0:
            self.linear = nn.ModuleList(all_linear_list)
        if self.fusion_type == "add":
            output_size = [v for v in self.output_size if v > 0]
            assert len(set(output_size)) == 1
            last_hidden_size = output_size[0]
        else:
            last_hidden_size = sum(self.output_size)
        last_hidden_size = last_hidden_size * 2
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

    def __forward__(
            self,
            cell_encoded_vectors,
            cell_encoded_matrices,
            cell_matrix_attention_masks,
            **kwargs
    ):
        sample_ids = kwargs["sample_ids"] if "sample_ids" in kwargs else None
        prefix = kwargs["prefix"] if "prefix" in kwargs else None
        attention_pooling_scores_savepath = kwargs["attention_pooling_scores_savepath"] if "attention_pooling_scores_savepath" in kwargs else None

        if "vector" in self.input_type:
            vector_vector = cell_encoded_vectors
            vector_linear_idx = self.linear_idx[0]
            if vector_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[vector_linear_idx]):
                    vector_vector = layer_module(vector_vector)
        if "matrix" in self.input_type:
            matrices = cell_encoded_matrices
            if self.matrix_encoder is not None:
                for module in self.matrix_encoder[:-1]:
                    matrices = module(matrices)
                matrices_output = self.matrix_encoder[-1](
                    input_ids=None,
                    attention_mask=cell_matrix_attention_masks,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=matrices,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False
                )
                matrices = matrices_output[0]
            if self.matrix_pooler is not None:
                matrix_vector = self.matrix_pooler(
                    matrices,
                    mask=cell_matrix_attention_masks,
                    sample_ids=sample_ids,
                    prefix=prefix,
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                )
            elif self.task_level_type in ["cell_level"]:
                tmp_mask = torch.unsqueeze(cell_matrix_attention_masks, dim=-1)
                matrices = matrices.masked_fill(tmp_mask == 0, 0.0)
                # 均值pooling
                matrix_vector = torch.sum(matrices, dim=1)/(torch.sum(tmp_mask, dim=1) + 1e-12)
            else:
                matrix_vector = matrices
            matrix_linear_idx = self.linear_idx[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)
        if "vector" in self.input_type:
            concat_vector = vector_vector
        elif "matrix" in self.input_type:
            concat_vector = matrix_vector
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        return concat_vector

    def forward(
            self,
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

        representation_vector_a = self.__forward__(
            cell_encoded_vectors_a,
            cell_encoded_matrices_a,
            cell_matrix_attention_masks_a,
            sample_ids=sample_ids,
            prefix="cell_a",
            attention_pooling_scores_savepath=attention_pooling_scores_savepath
        )
        representation_vector_b = self.__forward__(
            cell_encoded_vectors_b,
            cell_encoded_matrices_b,
            cell_matrix_attention_masks_b,
            sample_ids=sample_ids,
            prefix="cell_b",
            attention_pooling_scores_savepath=attention_pooling_scores_savepath
        )
        if self.fusion_type == "add":
            merged_vector = torch.add(representation_vector_a, representation_vector_b)
        else:
            merged_vector = torch.cat([representation_vector_a, representation_vector_b], dim=-1)
        if self.dropout is not None:
            merged_vector = self.dropout(merged_vector)
        if self.hidden_layer is not None:
            merged_vector = self.hidden_layer(merged_vector)
        if self.hidden_act is not None:
            merged_vector = self.hidden_act(merged_vector)
        logits = self.classifier(merged_vector)
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
                    "classification_vector": merged_vector[sample_idx].detach().cpu(),
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



