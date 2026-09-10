#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/8/2 19:48
@project: LucaOnePlusTasks
@file: luca_base.py
@desc: luca base
'''
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from common.pooling import *
    from common.loss import *
    from utils import *
    from common.multi_label_metrics import *
    from common.modeling_bert import BertPreTrainedModel
    from common.metrics import *
    from common.modeling_transformer import LucaTransformer, LucaEmbeddings
except ImportError:
    from src.common.pooling import *
    from src.common.loss import *
    from src.utils import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *
    from src.common.modeling_bert import BertPreTrainedModel
    from src.common.modeling_transformer import LucaTransformer, LucaEmbeddings


class LucaSingle(BertPreTrainedModel):
    def __init__(self, config, args):
        super(LucaSingle, self).__init__(config)
        if hasattr(config, "padding_idx"):
            self.padding_idx = config.padding_idx
        elif hasattr(config, "pad_token_id"):
            self.padding_idx = config.pad_token_id
        else:
            self.padding_idx = 0
        # cell vector or cell matrix
        self.loss_type = args.loss_type
        self.input_type = args.input_type
        self.output_mode = args.output_mode
        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type
        self.not_frozen_gene_express_bin_embedding = args.not_frozen_gene_express_bin_embedding
        self.task_level_type = args.task_level_type
        self.loss_reduction = args.loss_reduction
        self.embedding_frozen = args.embedding_frozen if hasattr(args, "embedding_frozen") else True
        if self.task_level_type not in ["cell_level"]:
            assert self.input_type != "vector"
            assert self.fusion_type == "add"
        self.matrix_encoder, self.matrix_pooler = None, None
        self.encoder_type_list = [False, False]
        self.input_size_list = [0, 0]
        self.linear_idx = [-1, -1]

        if self.not_frozen_gene_express_bin_embedding:
            # embedding 不固定，只使用LucaCell的embedding作为初始化
            matrix_encoder_config = copy.deepcopy(config)
            matrix_encoder_config.no_position_embeddings = True
            matrix_encoder_config.no_token_type_embeddings = True
            matrix_encoder_config.max_position_embeddings = config.cell_max_length
            old_vocab_size = matrix_encoder_config.vocab_size
            matrix_encoder_config.vocab_size = matrix_encoder_config.gene_express_vocab_size
            self.gene_express_embedding = LucaEmbeddings(
                matrix_encoder_config,
                init_weight_filepath=args.init_weight_filepath_for_gene_express_bin
            )
            matrix_encoder_config.vocab_size = old_vocab_size
            self.matrix_encoder = LucaTransformer(
                matrix_encoder_config,
                emb_layer=self.gene_express_embedding,
                use_pretrained_embedding=False,
                add_pooling_layer=False
            )
            ori_embedding_input_size = config.embedding_input_size
            config.embedding_input_size = config.hidden_size
            if self.task_level_type in ["cell_level"]:
                self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
            self.input_size_list[1] = config.embedding_input_size
            config.embedding_input_size = ori_embedding_input_size
            self.encoder_type_list[1] = True
            self.linear_idx[1] = 0
        else:
            if self.input_type == "vector":
                # emb vector -> fc * -> classifier
                self.input_size_list[0] = config.embedding_input_size
                self.encoder_type_list[0] = True
                self.linear_idx[0] = 0
            elif self.input_type == "matrix":
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
            elif self.input_type == "vector_matrix":
                # vector + matrix

                # for vector
                self.input_size_list[0] = config.embedding_input_size
                self.encoder_type_list[0] = True
                self.linear_idx[0] = 0

                # for matrix
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
                        if self.layer_norm_type == "pre":
                            self.matrix_encoder = nn.ModuleList([
                                nn.LayerNorm(config.embedding_input_size, eps=config.layer_norm_eps),
                                LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                            ])
                        else:
                            self.matrix_encoder = nn.ModuleList([
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
                self.linear_idx[1] = 1
            else:
                raise Exception("Not support input_type=%s" % self.input_type)

        fc_size_list = [config.vector_fc_size, config.matrix_fc_size]
        all_linear_list = [None, None]
        self.output_size = [0, 0]
        for encoder_idx, encoder_flag in enumerate(self.encoder_type_list):
            if not encoder_flag:
                continue
            fc_size = fc_size_list[encoder_idx]
            input_size = self.input_size_list[encoder_idx]
            if fc_size is not None:
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
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=last_hidden_size,
                classifier_size=config.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )
        self.post_init()

    def forward(
            self,
            cell_gene_seq_inputs_embeds=None,
            cell_gene_express_input_ids=None,
            cell_gene_attention_masks=None,
            cell_encoded_vectors=None,
            cell_encoded_matrices=None,
            cell_matrix_attention_masks=None,
            labels=None,
            **kwargs
    ):
        sample_ids = kwargs["sample_ids"] if "sample_ids" in kwargs else None
        attention_scores_savepath = kwargs["attention_scores_savepath"] if "attention_scores_savepath" in kwargs else None
        attention_pooling_scores_savepath = kwargs["attention_pooling_scores_savepath"] if "attention_pooling_scores_savepath" in kwargs else None
        output_classification_vector_dirpath = kwargs["output_classification_vector_dirpath"] if "output_classification_vector_dirpath" in kwargs else None
        return_attentions = sample_ids is not None and attention_scores_savepath is not None
        if self.not_frozen_gene_express_bin_embedding:
            outputs = self.matrix_encoder(
                gene_seq_inputs_embeds=cell_gene_seq_inputs_embeds,
                gene_express_input_ids=cell_gene_express_input_ids,
                attention_mask=cell_gene_attention_masks,
                position_ids=None,
                token_type_ids=None,
                return_attentions=return_attentions
            )
            if return_attentions:
                output_seq_attentions = outputs[2]
                for sample_idx, sample_id in enumerate(sample_ids):
                    cur_sample_output_seq_attentions = {}
                    for layer_idx, attn in output_seq_attentions.items():
                        cur_sample_output_seq_attentions[layer_idx] = attn[sample_idx].detach().cpu()
                filepath = os.path.join(attention_scores_savepath, "%s_seq_attention_scores.pt" % sample_id)
                torch.save(cur_sample_output_seq_attentions, filepath)
            if self.matrix_pooler is not None:
                matrix_vector = self.matrix_pooler(
                    outputs[0],
                    mask=cell_gene_attention_masks,
                    sample_ids=sample_ids,
                    prefix="cell",
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                )
            elif self.task_level_type in ["cell_level"]:
                tmp_mask = torch.unsqueeze(cell_matrix_attention_masks, dim=-1)
                matrices = outputs[0].masked_fill(tmp_mask == 0, 0.0)
                # 均值pooling
                matrix_vector = torch.sum(matrices, dim=1)/(torch.sum(tmp_mask, dim=1) + 1e-12)
            else:
                matrix_vector = outputs[0]
            matrix_linear_idx = self.linear_idx[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)
        else:
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
                        inputs_embeds=matrices
                    )
                    matrices = matrices_output[0]
                    if return_attentions:
                        output_matrix_attentions = matrices_output[2]
                        for sample_idx, sample_id in enumerate(sample_ids):
                            cur_sample_output_matrix_attentions = {}
                            for layer_idx, attn in output_matrix_attentions.items():
                                cur_sample_output_matrix_attentions[layer_idx] = attn[sample_idx].detach().cpu()
                            filepath = os.path.join(attention_scores_savepath, "%s_matrix_attention_scores.pt" % sample_id)
                            torch.save(cur_sample_output_matrix_attentions, filepath)
                if self.matrix_pooler is not None:
                    matrix_vector = self.matrix_pooler(
                        matrices,
                        mask=cell_matrix_attention_masks,
                        sample_ids=sample_ids,
                        prefix="matrix",
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
        if output_classification_vector_dirpath and sample_ids:
            for sample_idx, sample_id in enumerate(sample_ids):
                if self.not_frozen_gene_express_bin_embedding:
                    output_classification_vector = {
                        "representation_vector": matrix_vector[sample_idx].detach().cpu(),
                    }
                elif "vector" in self.input_type:
                    output_classification_vector = {
                        "representation_vector": vector_vector[sample_idx].detach().cpu(),
                    }
                elif "matrix" in self.input_type:
                    output_classification_vector = {
                        "representation_vector": matrix_vector[sample_idx].detach().cpu(),
                    }
                else:
                    raise Exception("Not support input_type=%s" % self.input_type)
                filepath = os.path.join(output_classification_vector_dirpath, "%s_classification_vector.pt" % sample_id)
                torch.save(output_classification_vector, filepath)
        if self.not_frozen_gene_express_bin_embedding:
            merged_vector = matrix_vector
        elif self.input_type == "vector":
            merged_vector = vector_vector
        elif self.input_type == "matrix":
            merged_vector = matrix_vector
        elif self.input_type == "vector_matrix":
            if self.fusion_type == "add":
                merged_vector = torch.add(vector_vector, matrix_vector)
            else:
                merged_vector = torch.cat([vector_vector, matrix_vector], dim=-1)
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        if self.dropout is not None:
            merged_vector = self.dropout(merged_vector)
        if self.hidden_layer is not None:
            merged_vector = self.hidden_layer(merged_vector)
        if self.hidden_act is not None:
            merged_vector = self.hidden_act(merged_vector)
        logits = self.classifier(merged_vector)
        if self.output is not None:
            output = self.output(logits)
        else:
            output = logits
        outputs = [logits, output]
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
