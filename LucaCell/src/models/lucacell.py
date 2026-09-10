#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 10:01
@project: LucaOne
@file: lucaone_gplm
@desc: LucaOne Model
'''
import sys
from typing import Union, List

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../src")
try:
    from common.loss import *
    from model_utils import AllOutput, create_output_loss_lucacell
    from alphabet import Alphabet
    from modeling_lucacell import *
    from models.pooling import *
    from models.configuration_lucacell import *
except ImportError:
    from src.common.loss import *
    from src.models.model_utils import AllOutput, create_output_loss_lucacell
    from src.models.alphabet import Alphabet
    from src.models.modeling_lucacell import *
    from src.models.pooling import *
    from src.models.configuration_lucacell import *
from torch.nn import CrossEntropyLoss


class LucaCell(nn.Module):
    def __init__(
            self,
            config: LucaCellConfig,
            args=None
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.ffn_size = config.ffn_size
        self.nucleotide_input_type = config.nucleotide_input_type
        if self.nucleotide_input_type in ["embedding_vector", "embedding_matrix"]:
            self.nucleotide_token_encoder_num_attention_heads = 0
            self.nucleotide_token_encoder_num_layers = 0
        else:
            self.nucleotide_token_encoder_num_attention_heads = config.nucleotide_token_encoder_num_attention_heads
            self.nucleotide_token_encoder_num_layers = config.nucleotide_token_encoder_num_layers
        self.attention_heads = config.num_attention_heads
        self.num_layers = config.num_layers
        self.use_rotary_embeddings = config.use_rotary_embeddings
        if self.use_rotary_embeddings:
            config.no_nucleotide_position_embeddings = True
            config.no_gene_positions_embeddings = True

        self.no_nucleotide_token_embeddings = config.no_nucleotide_token_embeddings
        self.no_nucleotide_token_encoder = config.no_nucleotide_token_encoder
        if self.no_nucleotide_token_encoder:
            self.nucleotide_token_encoder_num_attention_heads = 0
            self.nucleotide_token_encoder_num_layers = 0
        self.nucleotide_token_pooling_type = config.nucleotide_token_pooling_type
        self.no_nucleotide_position_embeddings = config.no_nucleotide_position_embeddings
        self.max_nucleotide_position_embeddings = config.max_nucleotide_position_embeddings
        self.no_gene_positions_embeddings = config.no_gene_positions_embeddings
        self.max_gene_position_embeddings = config.max_gene_position_embeddings
        self.no_express_sorted_embeddings = config.no_express_sorted_embeddings
        self.max_express_sorted_position_embeddings = config.max_express_sorted_position_embeddings
        self.no_gene_type_embeddings = config.no_gene_type_embeddings

        self.alphabet = Alphabet.from_predefined(
            config.alphabet_type,
            gene_id_list_filepath=args.gene_id_list_filepath if args and hasattr(args, "gene_id_list_filepath") else None,
            express_bin_list_filepath=args.express_bin_list_filepath if args and hasattr(args, "express_bin_list_filepath") else None,
            express_sorted_list_filepath=args.express_sorted_list_filepath if args and hasattr(args, "express_sorted_list_filepath") else None,
            tokenizer_dir=None,
            use_express_sorted=args and hasattr(args, "express_sorted_list_filepath") and args.express_sorted_list_filepath is not None,
        )

        self.nucleotide_padding_idx = self.alphabet.nucleotide_padding_idx
        self.nucleotide_mask_idx = self.alphabet.nucleotide_mask_idx
        self.nucleotide_cls_idx = self.alphabet.nucleotide_cls_idx
        self.nucleotide_eos_idx = self.alphabet.nucleotide_eos_idx
        self.nucleotide_prepend_bos = self.alphabet.nucleotide_prepend_bos
        self.nucleotide_append_eos = self.alphabet.nucleotide_append_eos

        self.gene_padding_idx = self.alphabet.gene_padding_idx
        self.gene_mask_idx = self.alphabet.gene_mask_idx
        self.gene_cls_idx = self.alphabet.gene_cls_idx
        self.gene_eos_idx = self.alphabet.gene_eos_idx
        self.gene_prepend_bos = self.alphabet.gene_prepend_bos
        self.gene_append_eos = self.alphabet.gene_append_eos

        self.express_padding_idx = self.alphabet.express_padding_idx
        self.express_mask_idx = self.alphabet.express_mask_idx
        self.express_cls_idx = self.alphabet.express_cls_idx
        self.express_eos_idx = self.alphabet.express_eos_idx
        self.express_prepend_bos = self.alphabet.express_prepend_bos
        self.express_append_eos = self.alphabet.express_append_eos

        self.ignore_index = config.ignore_index
        self.use_embed_layer_norm = config.use_embed_layer_norm
        self.use_last_layer_norm = config.use_last_layer_norm
        self.embed_scale = config.embed_scale
        self.pretrained_model_name = args.pretrained_model_name
        # 如果只用于embedding 推理则有些层不加载与构建
        if hasattr(args, "embedding_inference"):
            self.embedding_inference = args.embedding_inference
        else:
            self.embedding_inference = False
        self._init_submodules()
        if self.pretrained_model_name is not None:
            print("Load Pretrained Model Name=%s" % self.pretrained_model_name)
            self._init_submodules_new(self.pretrained_model_name)
        if not self.embedding_inference:
            if self.alphabet.alphabet_type == "gene":
                self.mask_gene_token_loss_fct = CrossEntropyLoss(
                    ignore_index=-100,
                    reduction="mean",
                    label_smoothing=0.0
                )
            self.mask_express_token_loss_fct = CrossEntropyLoss(
                ignore_index=-100,
                reduction="mean",
                label_smoothing=0.0
            )
            if not self.no_express_sorted_embeddings:
                self.mask_express_sorted_loss_fct = CrossEntropyLoss(
                    ignore_index=-100,
                    reduction="mean",
                    label_smoothing=0.0
                )

    def _init_submodules(self):
        if self.alphabet.alphabet_type == "nucleotide":
            self.nucleotide_tokens_embed = None
            self.nucleotide_positions_embed = None
            self.nucleotide_tokens_encoder = None
            self.nucleotide_last_layer_norm = None
            if not self.no_nucleotide_token_embeddings:
                self.nucleotide_tokens_embed = nn.Embedding(
                    self.alphabet.nucleotide_vocab_size,
                    self.embed_dim,
                    padding_idx=self.alphabet.nucleotide_padding_idx,
                )
                if not self.no_nucleotide_position_embeddings:
                    self.nucleotide_positions_embed = nn.Embedding(
                        self.max_nucleotide_position_embeddings,
                        self.embed_dim,
                        padding_idx=self.alphabet.nucleotide_padding_idx
                    )

                if not self.no_nucleotide_token_encoder:
                    if self.use_embed_layer_norm:
                        self.nucleotide_embed_layer_norm = Luca1bLayerNorm(self.embed_dim)
                    else:
                        self.nucleotide_embed_layer_norm = None
                    self.nucleotide_tokens_encoder = nn.ModuleList(
                        [
                            LucaTransformerLayer(
                                embed_dim=self.embed_dim,
                                ffn_embed_dim=self.ffn_size,
                                attention_heads=self.nucleotide_token_encoder_num_attention_heads,
                                add_bias_kv=False,
                                use_luca1b_layer_norm=True,
                                use_rotary_embeddings=self.use_rotary_embeddings
                            )
                            for _ in range(self.nucleotide_token_encoder_num_layers)
                        ]
                    )
                    if self.use_last_layer_norm:
                        self.nucleotide_last_layer_norm = Luca1bLayerNorm(self.embed_dim, eps=1e-8)
                    else:
                        self.nucleotide_last_layer_norm = None
            if self.nucleotide_token_pooling_type == "context_attention":
                self.nucleotide_tokens_pooler = GlobalMaskContextAttentionPooling1D(
                    self.embed_dim
                )
            elif self.nucleotide_token_pooling_type == "weighted_attention":
                self.nucleotide_tokens_pooler = GlobalMaskWeightedAttentionPooling1D(
                    self.embed_dim
                )
            elif self.nucleotide_token_pooling_type == "value_attention":
                self.nucleotide_tokens_pooler = GlobalMaskValueAttentionPooling1D(
                    self.embed_dim
                )
            else:
                self.nucleotide_tokens_pooler = None
            if self.alphabet.express_prepend_bos and self.alphabet.express_prepend_bos:
                self.gene_tokens_embed = nn.Embedding(
                    self.alphabet.gene_vocab_size,
                    self.embed_dim,
                    padding_idx=self.alphabet.gene_padding_idx
                )
        else:
            self.gene_tokens_embed = nn.Embedding(
                self.alphabet.gene_vocab_size,
                self.embed_dim,
                padding_idx=self.alphabet.gene_padding_idx
            )
            if not self.no_gene_positions_embeddings:
                self.gene_positions_embed = nn.Embedding(
                    self.max_gene_position_embeddings,
                    self.embed_dim,
                    padding_idx=self.alphabet.gene_padding_idx
                )

        self.express_tokens_embed = nn.Embedding(
            self.alphabet.express_vocab_size,
            self.embed_dim,
            padding_idx=self.alphabet.express_padding_idx,
        )
        self.express_sorted_positions_embed = None
        if not self.no_express_sorted_embeddings:
            self.express_sorted_positions_embed = nn.Embedding(
                self.max_express_sorted_position_embeddings,
                self.embed_dim,
                padding_idx=self.alphabet.express_padding_idx
            )
        self.gene_types_embed = None
        if not self.no_gene_type_embeddings:
            self.gene_types_embed = nn.Embedding(
                self.alphabet.gene_type_size + 1,
                self.embed_dim,
                padding_idx=self.gene_padding_idx,
            )
        if self.use_embed_layer_norm:
            self.embed_layer_norm = Luca1bLayerNorm(self.embed_dim)
        else:
            self.embed_layer_norm = None

        self.layers = nn.ModuleList(
            [
                LucaTransformerLayer(
                    self.embed_dim,
                    self.ffn_size,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_luca1b_layer_norm=True,
                    use_rotary_embeddings=self.use_rotary_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.layer_size = len(self.layers)

        if self.use_last_layer_norm:
            self.last_layer_norm = Luca1bLayerNorm(self.embed_dim, eps=1e-8)
        else:
            self.last_layer_norm = None

        if not self.embedding_inference:
            if self.alphabet.alphabet_type == "gene":
                self.gene_lm_head = LucaRobertaLMHead(
                    embed_dim=self.embed_dim,
                    output_dim=self.alphabet.gene_vocab_size,
                    weight=self.gene_tokens_embed.weight,
                )
            self.express_lm_head = LucaRobertaLMHead(
                embed_dim=self.embed_dim,
                output_dim=self.alphabet.express_vocab_size,
                weight=self.express_tokens_embed.weight,
            )
            if not self.no_express_sorted_embeddings:
                self.express_sorted_lm_head = LucaRobertaLMHead(
                    embed_dim=self.embed_dim,
                    output_dim=self.max_express_sorted_position_embeddings,
                    weight=self.express_sorted_positions_embed.weight,
                )

    def _init_embedding(
            self,
            pretrained_token_matrix,
            token_matrix
    ):
        for idx in range(0, 10):
            dim = token_matrix[idx, :].shape[0]
            token_matrix[idx, :] = pretrained_token_matrix[idx, :dim]
        return token_matrix

    def _init_submodules_new(self, pretrained_model_name):
        import os, sys
        from collections import OrderedDict
        sys.path.append(".")
        sys.path.append("..")
        sys.path.append("../../")
        sys.path.append("../../src")
        try:
            from utils import download_trained_checkpoint_lucaone_v2
            from lucaone.v2_0.alphabet import Alphabet
            from lucaone.v2_0.lucaone_gplm_config import LucaGPLMConfig
            from lucaone.v2_0.lucaone_gplm import LucaGPLM
        except ImportError:
            from src.utils import download_trained_checkpoint_lucaone_v2
            from src.lucaone.v2_0.alphabet import Alphabet
            from src.lucaone.v2_0.lucaone_gplm_config import LucaGPLMConfig
            from src.lucaone.v2_0.lucaone_gplm import LucaGPLM

        if pretrained_model_name == "lucaone":
            print("using LucaOne-Gene for weights init.")
            llm_type = "lucaone"
            llm_version = "lucaone"
            llm_step = "60000000"
            llm_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            download_trained_checkpoint_lucaone_v2(
                llm_dir=llm_dir,
                llm_type=llm_type,
                llm_version=llm_version,
                llm_step=llm_step
            )
            model_dirpath = "%s/models/%s/%s/checkpoint-step%s" % (
                llm_dir,
                llm_type,
                llm_version,
                llm_step
            )

            # load exists checkpoint
            pretrained_state_dict = torch.load(
                os.path.join(model_dirpath, "pytorch.pth"),
                map_location=torch.device("cpu"),
                weights_only=True
            )
            print("Load LucaOne Pretrained Model Success: %s" % model_dirpath)
            lucaone_pretrained_state_dict = OrderedDict()
            lucaone_num_layers = 0
            for k, v in pretrained_state_dict.items():
                if k.startswith("module."):
                    # remove `module.`
                    name = k[7:]
                else:
                    name = k
                lucaone_pretrained_state_dict[name] = v
                if "layers." in name:
                    cur_layer_id = int(name.split(".")[1])
                    if cur_layer_id > lucaone_num_layers:
                        lucaone_num_layers = cur_layer_id
            lucaone_num_layers += 1
            print("Init LucaOne layers: %d" % lucaone_num_layers)
            '''
            print("lucaone_pretrained_state_dict:")
            print(lucaone_pretrained_state_dict.keys())
            input("continue:")
            '''
            our_model_state_dict = {}
            our_model_state_dict_keys = set()
            for key, value in self.state_dict().items():
                our_model_state_dict[key] = value
                our_model_state_dict_keys.add(key)
            '''
            print("our_model_state_dict:")
            print(our_model_state_dict.keys())
            input("continue:")
            '''
            new_state_dict = OrderedDict()
            lucaone_last_layer_info = {}
            for name, weight in lucaone_pretrained_state_dict.items():
                layer_id = None
                if name.startswith("layers."):
                    layer_id = int(name.split(".")[1])
                    if not self.no_nucleotide_token_encoder and layer_id >= self.num_layers + self.nucleotide_token_encoder_num_layers \
                            or self.no_nucleotide_token_encoder and layer_id >= self.num_layers:
                        continue
                if name.startswith("embed_tokens.weight") and not self.no_nucleotide_token_embeddings:
                    our_name = "nucleotide_tokens_embed.weight"
                    new_state_dict[our_name] = self._init_embedding(weight, our_model_state_dict[our_name])
                    '''
                    print("our_model_state_dict:")
                    print(our_model_state_dict.keys())
                    '''
                    del our_model_state_dict[our_name]
                else:
                    if layer_id is not None:
                        if layer_id == lucaone_num_layers - 1:
                            lucaone_last_layer_info[name] = weight
                        if not self.no_nucleotide_token_encoder:
                            if layer_id < self.nucleotide_token_encoder_num_layers:
                                    our_name = name.replace("layers.", "nucleotide_tokens_encoder.")
                                    if weight.shape == our_model_state_dict[our_name].shape:
                                        new_state_dict[our_name] = weight
                                        del our_model_state_dict[our_name]
                            else:
                                our_name = name.replace(".%d." % layer_id, ".%d." % (layer_id - self.nucleotide_token_encoder_num_layers))
                                if weight.shape == our_model_state_dict[our_name].shape:
                                    new_state_dict[our_name] = weight
                                    del our_model_state_dict[our_name]
                                elif weight.ndim == 2 and weight.shape[0] == our_model_state_dict[our_name].shape[0] and weight.shape[1] > our_model_state_dict[our_name].shape[1]:
                                    new_state_dict[our_name] = weight[:, :our_model_state_dict[our_name].shape[1]]
                                    del our_model_state_dict[our_name]
                                elif weight.ndim == 2 and weight.shape[1] == our_model_state_dict[our_name].shape[1] and weight.shape[0] > our_model_state_dict[our_name].shape[0]:
                                    new_state_dict[our_name] = weight[:our_model_state_dict[our_name].shape[0], :]
                                    del our_model_state_dict[our_name]
                        else:
                            if weight.shape == our_model_state_dict[name].shape:
                                new_state_dict[name] = weight
                                del our_model_state_dict[name]
                            elif weight.ndim == 2 and weight.shape[0] == our_model_state_dict[name].shape[0] and weight.shape[1] > our_model_state_dict[name].shape[1]:
                                new_state_dict[name] = weight[:, :our_model_state_dict[name].shape[1]]
                                del our_model_state_dict[name]
                            elif weight.ndim == 2 and weight.shape[1] == our_model_state_dict[name].shape[1] and weight.shape[0] > our_model_state_dict[name].shape[0]:
                                new_state_dict[name] = weight[:our_model_state_dict[name].shape[0], :]
                                del our_model_state_dict[name]
                    elif name in our_model_state_dict_keys:
                        if weight.shape == our_model_state_dict[name].shape:
                            new_state_dict[name] = weight
                            del our_model_state_dict[name]
            if self.num_layers > lucaone_num_layers:
                for layer_id in range(lucaone_num_layers, self.num_layers):
                    for item in lucaone_last_layer_info.items():
                        our_name = item[0].replace(".%d." % (lucaone_num_layers - 1), ".%d." %  layer_id)
                        if item[1].shape == our_model_state_dict[our_name].shape:
                            new_state_dict[our_name] = item[1]
                            del our_model_state_dict[our_name]
                        elif item[1].ndim == 2 and item[1].shape[0] == our_model_state_dict[our_name].shape[0] and item[1].shape[1] > our_model_state_dict[our_name].shape[1]:
                            new_state_dict[our_name] = item[1][:, :our_model_state_dict[our_name].shape[1]]
                            del our_model_state_dict[our_name]
                        elif item[1].ndim == 2 and item[1].shape[1] == our_model_state_dict[our_name].shape[1] and item[1].shape[0] > our_model_state_dict[our_name].shape[0]:
                            new_state_dict[our_name] = item[1][:our_model_state_dict[our_name].shape[0], :]
                            del our_model_state_dict[our_name]
            print("Used state_dict:")
            print(new_state_dict.keys())
            diff = set(our_model_state_dict.keys()).difference(set(new_state_dict.keys()))
            print("Diff:")
            print(diff)
            new_state_dict.update(our_model_state_dict)
            self.load_state_dict(new_state_dict)
        else:
            print("using ESM2-3B for weights init.")
            from esm import pretrained
            from collections import OrderedDict
            our_model_state_dict = {}
            our_model_state_dict_keys = set()
            for key, value in self.state_dict().items():
                our_model_state_dict[key] = value
                our_model_state_dict_keys.add(key)

            pretrained, _ = pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
            esm_pretrained_state_dict = pretrained.state_dict()

            new_state_dict = OrderedDict()
            our_model_state_dict = {}
            for key, value in self.state_dict().items():
                our_model_state_dict[key] = value

            for name, weight in esm_pretrained_state_dict.items():
                layer_id = None
                if name.startswith("layers."):
                    layer_id = int(name.split(".")[1])
                    if not self.no_nucleotide_token_encoder and layer_id >= self.num_layers + self.nucleotide_token_encoder_num_layers \
                            or self.no_nucleotide_token_encoder and layer_id >= self.num_layers:
                        continue
                if "final_layer_norm" in name:
                    name = name.replace("final_layer_norm", "post_layer_norm")
                elif "self_attn_layer_norm" in name:
                    name = name.replace("self_attn_layer_norm", "pre_layer_norm")
                elif "emb_layer_norm_after" in name:
                    name = name.replace("emb_layer_norm_after", "last_layer_norm")

                if name.startswith("embed_tokens.weight") and not self.no_nucleotide_token_embeddings:
                    our_name = "nucleotide_tokens_embed.weight"
                    new_state_dict[our_name] = self._init_embedding(weight, our_model_state_dict[our_name])
                    del our_model_state_dict[our_name]
                else:
                    if layer_id is not None:
                        if not self.no_nucleotide_token_encoder:
                            if layer_id < self.nucleotide_token_encoder_num_layers:
                                our_name = name.replace("layers.", "nucleotide_tokens_encoder.")
                                if weight.shape == our_model_state_dict[our_name].shape:
                                    new_state_dict[our_name] = weight
                                    del our_model_state_dict[our_name]
                            else:
                                our_name = name.replace(".%d." % layer_id, ".%d." % (layer_id - self.nucleotide_token_encoder_num_layers))
                                if weight.shape == our_model_state_dict[our_name].shape:
                                    new_state_dict[our_name] = weight
                                    del our_model_state_dict[our_name]
                        else:
                            if weight.shape == our_model_state_dict[name].shape:
                                new_state_dict[name] = weight
                                del our_model_state_dict[name]
                    elif name in our_model_state_dict_keys:
                        if weight.shape == our_model_state_dict[name].shape:
                            new_state_dict[name] = weight
                            del our_model_state_dict[name]
            print("Used state_dict:")
            print(new_state_dict.keys())
            diff = set(our_model_state_dict.keys()).difference(set(new_state_dict.keys()))
            print("Diff:")
            print(diff)
            new_state_dict.update(our_model_state_dict)
            self.load_state_dict(new_state_dict)

    def last_nonzero_mask(self, mask):
        # 找到每一行最后一个非零元素的索引
        last_nonzero = torch.max((mask != 0).long().cumsum(dim=1), dim=1).indices

        # 创建一个与输入相同形状的全零矩阵
        new_mask = torch.zeros_like(mask)

        # 使用高级索引，将每一行的最后一个非零位置设为1
        new_mask[torch.arange(mask.size(0)), last_nonzero] = 1

        return new_mask

    def __forword__(
            self,
            nucleotide_input_ids: Optional[List[torch.Tensor]] = None,
            nucleotide_position_ids: Optional[List[torch.Tensor]] = None,
            nucleotide_input_embeds: Optional[List[torch.Tensor]] = None,
            nucleotide_attention_mask: Optional[List[torch.Tensor]]= None,
            gene_input_ids: Optional[torch.Tensor] = None,
            gene_position_ids: Optional[torch.Tensor] = None,
            gene_attention_mask: Optional[torch.Tensor] = None,
            gene_type_ids: Optional[torch.Tensor] = None,
            express_input_ids: Optional[torch.Tensor] = None,
            express_attention_mask: Optional[torch.Tensor] = None,
            express_sorted_position_ids: Optional[torch.Tensor] = None,
            labels: Optional[dict[str, torch.Tensor]] = None,
            repr_layers=[-1],
            need_weights=False,
            need_head_weights=False,
            use_last_layer_norm=True,
            return_simple: Optional[bool] = False,
            device=None,
    ):
        if device is None:
            device = express_input_ids.device
        batch_size = express_input_ids.shape[0]
        assert all(-(self.layer_size + 1) <= i <= self.layer_size for i in repr_layers)
        repr_layers = [(i + self.layer_size + 1) % (self.layer_size + 1) for i in repr_layers]
        assert nucleotide_input_ids is None or nucleotide_input_ids.ndim == 3
        # 动态求mask，(B * Seq_len) 被mask掉位置的值为True
        if nucleotide_input_ids is not None and nucleotide_attention_mask is None:
            nucleotide_attention_mask = []
            for cur_nucleotide_input_ids in nucleotide_input_ids:
                nucleotide_attention_mask.append(~cur_nucleotide_input_ids.eq(self.alphabet.nucleotide_padding_idx).long())

        if express_attention_mask is None:
            express_attention_mask = (~express_input_ids.eq(self.express_padding_idx)).long()

        # nucleotide_input_ids: (B, nucleotide_seq_num, nucleotide_seq_len)
        if nucleotide_input_ids is not None or nucleotide_input_embeds is not None:
            if nucleotide_input_ids is not None:
                seq_num = len(nucleotide_input_ids)
                x_list = []
                for seq_idx in range(seq_num):
                    x = self.nucleotide_tokens_embed(nucleotide_input_ids[seq_idx].to(device))
                    if self.nucleotide_positions_embed is not None:
                        x += self.nucleotide_positions_embed(nucleotide_position_ids[seq_idx].to(device))
                    if self.nucleotide_tokens_encoder is not None:
                        if self.nucleotide_embed_layer_norm is not None:
                            x = self.nucleotide_embed_layer_norm(x)
                        # (L, B, E)
                        x = x.transpose(0, 1)
                        for layer_idx, layer in enumerate(self.nucleotide_tokens_encoder):
                            x, _ = layer(
                                x,
                                self_attn_padding_mask=1 - nucleotide_attention_mask[seq_idx].to(device),
                                need_head_weights=need_head_weights,
                                need_weights=need_weights,
                            )
                        # (L, B, E)
                        if self.nucleotide_last_layer_norm is not None and use_last_layer_norm:
                            # 最后一层隐含层 加一层layerNorm
                            x = self.nucleotide_last_layer_norm(x)
                        # (B, L, E)
                        x = x.transpose(0, 1)
                        x = self.nucleotide_tokens_pooler(
                            x,
                            mask=nucleotide_attention_mask[seq_idx].to(device)
                        ).cpu()
                    torch.cuda.empty_cache()
                    x_list.append(x)
                x = torch.stack(x_list, dim=1)
                '''
                显卡不够 会报错
                batch_size, seq_num, seq_len = nucleotide_input_ids.shape
                nucleotide_input_ids_reshaped = nucleotide_input_ids.view(-1, seq_len)
                x = self.nucleotide_tokens_embed(nucleotide_input_ids_reshaped)
                if self.nucleotide_positions_embed is not None:
                    nucleotide_position_ids_reshaped = nucleotide_position_ids.view(-1, seq_len)
                    x += self.nucleotide_positions_embed(nucleotide_position_ids_reshaped)
                if self.nucleotide_tokens_encoder is not None:
                    if self.nucleotide_embed_layer_norm is not None:
                        x = self.nucleotide_embed_layer_norm(x)

                    nucleotide_attention_mask_reshaped = nucleotide_attention_mask.view(-1, seq_len)
                    # (L, B, E)
                    x = x.transpose(0, 1)
                    for layer_idx, layer in enumerate(self.nucleotide_tokens_encoder):
                        x, _ = layer(
                            x,
                            self_attn_padding_mask=1 - nucleotide_attention_mask_reshaped,
                            need_head_weights=need_head_weights,
                            need_weights=need_weights,
                        )
                    # (L, B, E)
                    if self.nucleotide_last_layer_norm is not None and use_last_layer_norm:
                        # 最后一层隐含层 加一层layerNorm
                        x = self.nucleotide_last_layer_norm(x)
                    x = x.transpose(0, 1)
                    x = x.view(batch_size, seq_num, seq_len, self.embed_dim)
                    x = self.nucleotide_tokens_pooler(
                        x,
                        mask=nucleotide_attention_mask
                    )
                '''
            else:
                if self.nucleotide_input_type == "embedding_matrix":
                    seq_num = len(nucleotide_input_embeds)
                    x_list = []
                    cpu_flag_list = []
                    for seq_idx in range(seq_num):
                        emb = nucleotide_input_embeds[seq_idx].to(device)
                        if nucleotide_attention_mask is not None:
                            mask = nucleotide_attention_mask[seq_idx].to(device)
                        else:
                            mask = None
                        x = self.nucleotide_tokens_pooler(
                            emb,
                            mask=mask
                        )
                        cur_seq_len = emb.shape[1]
                        if cur_seq_len > 2048:
                            cpu_flag_list.append(True)
                            x = x.cpu()
                        else:
                            cpu_flag_list.append(False)
                        x_list.append(x)
                    if all(cpu_flag_list):
                        x_list = torch.stack(x_list, dim=1).to(device)
                    elif any(cpu_flag_list):
                        x_list = torch.stack([v.to(device) if cpu_flag_list[v_idx] else v for v_idx, v in enumerate(x_list)], dim=1)
                    else:
                        x_list = torch.stack(x_list, dim=1)
                    x = x_list
                    # print("x:", x.shape)
                    # del x_list
                else:
                    '''
                    x = torch.stack(nucleotide_input_embeds, dim=1).to(device)
                    del nucleotide_input_embeds
                    '''
                    x = nucleotide_input_embeds
            if self.alphabet.express_prepend_bos and self.alphabet.express_prepend_bos:
                if batch_size > 1:
                    x = torch.cat(
                        (
                            self.gene_tokens_embed(
                                torch.full(
                                    (x.shape[0], 1),
                                    self.alphabet.gene_cls_idx,
                                    dtype=torch.long,
                                    device=x.device
                                )
                            ),
                            x,
                            torch.zeros(
                                (x.shape[0], 1, self.embed_dim),
                                dtype=torch.float32,
                                device=x.device
                            )
                        ),
                        dim=1
                    )
                    eos = self.gene_tokens_embed(
                        torch.full(
                            (x.shape[0], 1),
                            self.alphabet.gene_eos_idx,
                            dtype=torch.long,
                            device=x.device
                        )
                    ).expand(x.shape[0], x.shape[1], self.embed_dim)
                    # print("x:", x.shape)
                    x += eos * self.last_nonzero_mask(express_attention_mask.unsqueeze(-1))
                else:
                    x = torch.cat(
                        (
                            self.gene_tokens_embed(
                                torch.full(
                                    (batch_size, 1),
                                    self.alphabet.gene_cls_idx,
                                    dtype=torch.long,
                                    device=x.device
                                )
                            ),
                            x,
                            self.gene_tokens_embed(
                                torch.full(
                                    (batch_size, 1),
                                    self.alphabet.gene_eos_idx,
                                    dtype=torch.long,
                                    device=x.device
                                )
                            )
                        ),
                        dim=1
                    )
        else:
            x = self.embed_scale * self.gene_tokens_embed(gene_input_ids)
            if not self.no_gene_positions_embeddings:
                x += self.gene_positions_embed(gene_position_ids)
        if self.gene_types_embed:
            x += self.gene_types_embed(gene_type_ids)
        x += self.embed_scale * self.express_tokens_embed(express_input_ids)
        if self.express_sorted_positions_embed is not None and express_sorted_position_ids is not None:
            x += self.express_sorted_positions_embed(express_sorted_position_ids)
        if self.embed_layer_norm is not None:
            x = self.embed_layer_norm(x)

        # Mask 操作
        if express_attention_mask is not None:
            x = x * express_attention_mask.unsqueeze(-1).type_as(x)

        # 返回值包括哪些
        repr_layers = set(repr_layers)
        if not return_simple:
            hidden_representations = {}
            # 0:embedding
            if 0 in repr_layers:
                hidden_representations[0] = x

            # 是否需要返回head weights
            if need_head_weights:
                attn_weights = []

        # (B, L, E) => (L, B, E)
        x = x.transpose(0, 1)

        for layer_idx, layer in enumerate(self.layers):
            if return_simple:
                x, _ = layer(
                    x,
                    self_attn_padding_mask=1 - express_attention_mask,
                    need_head_weights=need_head_weights,
                    need_weights=need_weights,
                    return_simple=True
                )
            else:
                x, attn = layer(
                    x,
                    self_attn_padding_mask=1 - express_attention_mask,
                    need_head_weights=need_head_weights,
                    need_weights=need_weights,
                    return_simple=False
                )

            if not return_simple:
                if (layer_idx + 1) in repr_layers:
                    hidden_representations[layer_idx + 1] = x.transpose(0, 1)
                if need_head_weights:
                    # (H, B, L, L) => (B, H, L, L)
                    attn_weights.append(attn.transpose(1, 0))

        # (L, B, E)
        if self.last_layer_norm is not None and use_last_layer_norm:
            # 最后一层隐含层 加一层layerNorm
            x = self.last_layer_norm(x)
        x = x.transpose(0, 1)  # (L, B, E) => (B, L, E)

        # last hidden representation should have layer norm applied
        if not return_simple:
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x
            # 最后一层作为表征矩阵
            # (B, L, E)
            representation_matrix = hidden_representations[self.layer_size]
        # mask 任务
        # B * Seq_len * vocab_size
        mask_gene_token_lm_logits = None
        mask_express_token_lm_logits = None
        mask_express_sorted_lm_logits = None
        if not self.embedding_inference:
            if self.alphabet.alphabet_type == "gene":
                mask_gene_token_lm_logits = self.gene_lm_head(x)
            mask_express_token_lm_logits = self.express_lm_head(x)
            if not self.no_express_sorted_embeddings:
                mask_express_sorted_lm_logits = self.express_sorted_lm_head(x)
        # lm head的输出向量作为表征向量
        # (B, E)
        if not return_simple:
            representation_vector = representation_matrix[:, 0, :]

            representations = {
                "representation_matrix": representation_matrix,
                "representation_vector": representation_vector
            }
            # 每一层的attention值
            if need_head_weights:
                # attentions: B x Layers x H x L x L
                attentions = torch.stack(attn_weights, 1)
                if express_attention_mask is not None:
                    attention_mask = express_attention_mask.type_as(attentions)
                    attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                    attentions = attentions * attention_mask[:, None, None, :, :]
                if not return_simple:
                    representations["attentions"] = attentions

        mask_gene_token_loss = None
        mask_express_token_loss = None
        mask_express_sorted_loss = None
        if not self.embedding_inference and labels is not None:
            if self.alphabet.alphabet_type == "gene":
                mask_gene_token_labels = labels["gene_mask"].to(mask_gene_token_lm_logits.device)
                mask_gene_token_loss = self.mask_gene_token_loss_fct(
                    mask_gene_token_lm_logits.view(-1, self.alphabet.gene_vocab_size),
                    mask_gene_token_labels.view(-1)
                )

            mask_express_token_labels = labels["express_value_mask"].to(mask_express_token_lm_logits.device)
            mask_express_token_loss = self.mask_express_token_loss_fct(
                mask_express_token_lm_logits.view(-1, self.alphabet.express_vocab_size),
                mask_express_token_labels.view(-1)
            )
            if not self.no_express_sorted_embeddings:
                mask_express_sorted_labels = labels["express_sorted_mask"].to(mask_express_sorted_lm_logits.device)
                mask_express_sorted_loss = self.mask_express_sorted_loss_fct(
                    mask_express_sorted_lm_logits.view(-1, self.max_express_sorted_position_embeddings),
                    mask_express_sorted_labels.view(-1)
                )
        return None if return_simple else representations, \
               None if return_simple else mask_gene_token_lm_logits, \
               None if return_simple else mask_express_token_lm_logits, \
               None if return_simple else mask_express_sorted_lm_logits, \
               mask_gene_token_loss, \
               mask_express_token_loss,\
               mask_express_sorted_loss

    def forward(
            self,
            nucleotide_input_ids: Optional[torch.Tensor] = None,
            nucleotide_input_embeds: Optional[torch.Tensor] = None,
            nucleotide_attention_mask: Optional[torch.Tensor] = None,
            gene_input_ids: Optional[torch.Tensor] = None,
            gene_attention_mask: Optional[torch.Tensor] = None,
            gene_type_ids: Optional[torch.Tensor] = None,
            express_input_ids: Optional[torch.Tensor] = None,
            express_attention_mask: Optional[torch.Tensor] = None,
            express_sorted_position_ids: Optional[torch.Tensor] = None,
            labels: Optional[dict[str, torch.Tensor]] = None,
            need_head_weights: Optional[bool] = None,
            repr_layers: Optional[list[int]] = None,
            return_dict: Optional[bool] = None,
            return_simple: Optional[bool] = False,
            use_last_layer_norm: Optional[bool] = True,
            device=None,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], AllOutput]:
        if return_dict is None and self.config is not None:
            return_dict = self.config.use_return_dict
        if return_dict is None:
            return_dict = False
        if repr_layers is None or len(repr_layers) == 0:
            repr_layers = [-1]
        if need_head_weights is None:
            need_head_weights = True
        representations, \
        mask_gene_token_lm_logits, mask_express_token_lm_logits, mask_express_sorted_lm_logits, \
        mask_gene_token_loss, mask_express_token_loss, mask_express_sorted_loss = self.__forword__(
            nucleotide_input_ids=nucleotide_input_ids,
            nucleotide_input_embeds=nucleotide_input_embeds,
            nucleotide_attention_mask=nucleotide_attention_mask,
            gene_input_ids=gene_input_ids,
            gene_attention_mask=gene_attention_mask,
            gene_type_ids=gene_type_ids,
            express_input_ids=express_input_ids,
            express_attention_mask=express_attention_mask,
            express_sorted_position_ids=express_sorted_position_ids,
            labels=labels,
            repr_layers=repr_layers,
            need_head_weights=need_head_weights,
            use_last_layer_norm=use_last_layer_norm,
            return_simple=return_simple,
            device=device
        )
        if self.alphabet.alphabet_type == "gene" and not self.no_express_sorted_embeddings:
            loss = mask_gene_token_loss + mask_express_token_loss + mask_express_sorted_loss
        elif self.alphabet.alphabet_type == "gene":
            loss = mask_gene_token_loss + mask_express_token_loss
        elif not self.no_express_sorted_embeddings:
            loss = mask_express_token_loss + mask_express_sorted_loss
        else:
            loss = mask_express_token_loss

        if not self.embedding_inference:
            if not return_dict:
                if return_simple:
                    return loss
                else:
                    return [
                        [loss, mask_gene_token_loss, mask_express_token_loss, mask_express_sorted_loss],
                        [mask_gene_token_lm_logits, mask_express_token_lm_logits, mask_express_sorted_lm_logits],
                        [representations]
                    ]
            if return_simple:
                return AllOutput(
                    losses=[loss],
                    outputs=None,
                    hidden_states=None,
                    attentions=None,
                    cross_attentions=None,
                    global_attentions=None
                )
            return AllOutput(
                losses=[loss, mask_gene_token_loss, mask_express_token_loss, mask_express_sorted_loss],
                outputs=[mask_gene_token_lm_logits, mask_express_token_lm_logits, mask_express_sorted_lm_logits],
                hidden_states=representations["representation_matrix"] if "representation_matrix" in representations else None,
                attentions=representations["attentions"] if "attentions" in representations else None,
                cross_attentions=None,
                global_attentions=None
            )
        else:
            if not return_dict:
                return [[None], [None], [representations]]
            return AllOutput(
                losses=None,
                outputs=None,
                hidden_states=representations["representation_matrix"] if "representation_matrix" in representations else None,
                attentions=representations["attentions"] if "attentions" in representations else None,
                cross_attentions=None,
                global_attentions=None
            )
