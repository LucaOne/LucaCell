#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 11:00
@project: LucaOne
@file: alphabet
@desc: alphabet for LucaOne
'''
import os
import os.path
from typing import List, Union, Any

nucleotide_standard_toks = ['A', 'T', 'C', 'G', 'N']

nucleotide_prepend_toks = ['[N_PAD]', '[N_UNK]']

nucleotide_append_toks = ['[N_CLS]', '[N_SEP]', '[N_MASK]']

gene_standard_toks = []

gene_prepend_toks = ['[G_PAD]', '[G_UNK]']

gene_append_toks = ['[G_CLS]', '[G_SEP]', '[G_MASK]']

express_standard_toks = []

express_prepend_toks = ['[E_PAD]', '[E_UNK]']

express_append_toks = ['[E_CLS]', '[E_SEP]', '[E_MASK]']

express_sorted_standard_toks = []

express_sorted_prepend_toks = ['[S_PAD]', '[S_UNK]']

express_sorted_append_toks = ['[S_CLS]', '[S_SEP]', '[S_MASK]']


class Alphabet(object):
    def __init__(
            self,
            alphabet_type,
            nucleotide_standard_toks: Union[List[str], None],
            nucleotide_prepend_toks: Union[List[str], None],
            nucleotide_append_toks: Union[List[str], None],
            nucleotide_prepend_bos:Union[bool, None],
            nucleotide_append_eos: Union[bool, None],
            gene_standard_toks:Union[List[str], None],
            gene_prepend_toks: Union[List[str], None],
            gene_append_toks: Union[List[str], None],
            gene_prepend_bos: Union[bool, None],
            gene_append_eos: Union[bool, None],
            express_standard_toks: List[str] = express_standard_toks,
            express_prepend_toks: List[str] = express_prepend_toks,
            express_append_toks: List[str] = express_append_toks,
            express_sorted_standard_toks: Union[List[str], None] = express_sorted_standard_toks,
            express_sorted_prepend_toks: Union[List[str], None] = express_sorted_prepend_toks,
            express_sorted_append_toks: Union[List[str], None] = express_sorted_append_toks,
            express_prepend_bos: bool = True,
            express_append_eos: bool = True,
            use_express_sorted=False,
            **kwargs
    ):
        self.use_express_sorted = use_express_sorted
        assert alphabet_type in ["gene", "nucleotide"]
        self.project_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )
            )
        )
        if alphabet_type == "nucleotide":
            assert nucleotide_prepend_bos == nucleotide_append_eos
        assert gene_prepend_bos == gene_append_eos == express_prepend_bos == express_append_eos
        self.alphabet_type = alphabet_type
        if alphabet_type == "nucleotide":
            self.nucleotide_standard_toks = list(nucleotide_standard_toks)
            self.nucleotide_prepend_toks = list(nucleotide_prepend_toks)
            self.nucleotide_append_toks = list(nucleotide_append_toks)
            self.nucleotide_prepend_bos = nucleotide_prepend_bos
            self.nucleotide_append_eos = nucleotide_append_eos

            self.all_nucleotide_toks = list(self.nucleotide_prepend_toks)
            self.all_nucleotide_toks.extend(self.nucleotide_append_toks)
            self.all_nucleotide_toks.extend(self.nucleotide_standard_toks)

            self.nucleotide_tok_to_idx = {tok: i for i, tok in enumerate(self.all_nucleotide_toks)}
            self.nucleotide_unk_idx = self.nucleotide_tok_to_idx["[N_UNK]"]
            self.nucleotide_padding_idx = self.get_nucleotide_idx("[N_PAD]")
            self.nucleotide_pad_token_id = self.nucleotide_padding_idx
            self.nucleotide_cls_idx = self.get_nucleotide_idx("[N_CLS]")
            self.nucleotide_mask_idx = self.get_nucleotide_idx("[N_MASK]")
            self.nucleotide_eos_idx = self.get_nucleotide_idx("[N_SEP]")

            self.all_nucleotide_special_tokens = nucleotide_prepend_toks + nucleotide_append_toks
            self.nucleotide_vocab_size = len(self.all_nucleotide_toks)

            self.gene_standard_toks = None
            self.gene_prepend_toks = list(gene_prepend_toks)
            self.gene_append_toks = list(gene_append_toks)
            self.gene_prepend_bos = gene_prepend_bos
            self.gene_append_eos = gene_append_eos

            self.all_gene_toks = list(self.gene_prepend_toks)
            self.all_gene_toks.extend(self.gene_append_toks)

            self.gene_tok_to_idx = {tok: i for i, tok in enumerate(self.all_gene_toks)}
            self.gene_unk_idx = self.gene_tok_to_idx["[G_UNK]"]
            self.gene_padding_idx = self.get_gene_idx("[G_PAD]")
            self.gene_pad_token_id = self.gene_padding_idx
            self.gene_cls_idx = self.get_gene_idx("[G_CLS]")
            self.gene_mask_idx = self.get_gene_idx("[G_MASK]")
            self.gene_eos_idx = self.get_gene_idx("[G_SEP]")
        else:
            self.gene_id_list_filepath = None
            if gene_standard_toks is None or len(gene_standard_toks) == 0:
                if "gene_id_list_filepath" in kwargs and kwargs["gene_id_list_filepath"]:
                    gene_id_list_filepath = kwargs["gene_id_list_filepath"]
                else:
                    gene_id_list_filepath = os.path.join(
                        self.project_dir,
                        "meta",
                        "gene_id_list.meta"
                    )
                self.gene_id_list_filepath = gene_id_list_filepath
                gene_standard_toks = []
                with open(gene_id_list_filepath, "r") as rfp:
                    for line in rfp:
                        line = str(line).strip()
                        gene_standard_toks.append(line)
            self.gene_standard_toks = list(gene_standard_toks)
            self.gene_prepend_toks = list(gene_prepend_toks)
            self.gene_append_toks = list(gene_append_toks)
            self.gene_prepend_bos = gene_prepend_bos
            self.gene_append_eos = gene_append_eos

            self.all_gene_toks = list(self.gene_prepend_toks)
            self.all_gene_toks.extend(self.gene_append_toks)
            self.all_gene_toks.extend(self.gene_standard_toks)

            self.gene_tok_to_idx = {tok: i for i, tok in enumerate(self.all_gene_toks)}
            self.gene_unk_idx = self.gene_tok_to_idx["[G_UNK]"]
            self.gene_padding_idx = self.get_gene_idx("[G_PAD]")
            self.gene_pad_token_id = self.gene_padding_idx
            self.gene_cls_idx = self.get_gene_idx("[G_CLS]")
            self.gene_mask_idx = self.get_gene_idx("[G_MASK]")
            self.gene_eos_idx = self.get_gene_idx("[G_SEP]")

        self.express_bin_list_filepath = None
        if express_standard_toks is None or len(express_standard_toks) == 0:
            if "express_bin_list_filepath" in kwargs and kwargs["express_bin_list_filepath"]:
                express_bin_list_filepath = kwargs["express_bin_list_filepath"]
            else:
                express_bin_list_filepath = os.path.join(
                    self.project_dir,
                    "meta",
                    "express_bin_list.meta"
                )
            self.express_bin_list_filepath = express_bin_list_filepath
            express_standard_toks = []
            with open(express_bin_list_filepath, "r") as rfp:
                for line in rfp:
                    line = str(line).strip()
                    express_standard_toks.append(line)
        self.express_standard_toks = list(express_standard_toks)
        self.express_prepend_toks = list(express_prepend_toks)
        self.express_append_toks = list(express_append_toks)
        self.express_prepend_bos = express_prepend_bos
        self.express_append_eos = express_append_eos

        self.all_express_toks = list(self.express_prepend_toks)
        self.all_express_toks.extend(self.express_append_toks)
        self.all_express_toks.extend(self.express_standard_toks)

        self.express_tok_to_idx = {tok: i for i, tok in enumerate(self.all_express_toks)}
        self.express_unk_idx = self.express_tok_to_idx["[E_UNK]"]
        self.express_padding_idx = self.get_express_idx("[E_PAD]")
        self.express_pad_token_id = self.express_padding_idx
        self.express_cls_idx = self.get_express_idx("[E_CLS]")
        self.express_mask_idx = self.get_express_idx("[E_MASK]")
        self.express_eos_idx = self.get_express_idx("[E_SEP]")

        if use_express_sorted:
            self.express_sorted_list_filepath = None
            if express_sorted_standard_toks is None or len(express_sorted_standard_toks) == 0:
                if "express_sorted_list_filepath" in kwargs and kwargs["express_sorted_list_filepath"]:
                    express_sorted_list_filepath = kwargs["express_sorted_list_filepath"]
                else:
                    express_sorted_list_filepath = os.path.join(
                        self.project_dir,
                        "meta",
                        "express_sorted_list.meta"
                    )
                self.express_sorted_list_filepath = express_sorted_list_filepath
                express_sorted_standard_toks = []
                with open(express_sorted_list_filepath, "r") as rfp:
                    for line in rfp:
                        line = str(line).strip()
                        express_sorted_standard_toks.append(line)
            self.express_sorted_standard_toks = list(express_sorted_standard_toks)
            self.express_sorted_prepend_toks = list(express_sorted_prepend_toks)
            self.express_sorted_append_toks = list(express_sorted_append_toks)

            self.all_express_sorted_toks = list(self.express_sorted_prepend_toks)
            self.all_express_sorted_toks.extend(self.express_sorted_append_toks)
            self.all_express_sorted_toks.extend(self.express_sorted_standard_toks)

            self.express_sorted_tok_to_idx = {tok: i for i, tok in enumerate(self.all_express_sorted_toks)}
            self.express_sorted_unk_idx = self.express_sorted_tok_to_idx["[S_UNK]"]
            self.express_sorted_padding_idx = self.get_express_sorted_idx("[S_PAD]")
            self.express_sorted_pad_token_id = self.express_sorted_padding_idx
            self.express_sorted_cls_idx = self.get_express_sorted_idx("[S_CLS]")
            self.express_sorted_mask_idx = self.get_express_sorted_idx("[S_MASK]")
            self.express_sorted_eos_idx = self.get_express_sorted_idx("[E_SEP]")

        self.all_express_special_tokens = express_prepend_toks + express_append_toks
        if use_express_sorted:
            self.all_express_sorted_special_tokens = express_sorted_prepend_toks + express_sorted_append_toks

        self.all_gene_special_tokens = gene_prepend_toks + gene_append_toks
        self.gene_type_size = 2
        self.gene_vocab_size = len(self.all_gene_toks)
        self.express_vocab_size = len(self.all_express_toks)
        if use_express_sorted:
            self.express_sorted_vocab_size = len(self.all_express_sorted_toks)

    def get_nucleotide_idx(self, tok):
        if self.alphabet_type == "gene":
            raise Exception("Not Support function for nucleotide")
        return self.nucleotide_tok_to_idx.get(tok, self.nucleotide_unk_idx)

    def get_gene_idx(self, tok):
        return self.gene_tok_to_idx.get(tok, self.gene_unk_idx)

    def get_express_idx(self, tok):
        return self.express_tok_to_idx.get(tok, self.express_unk_idx)

    def get_express_sorted_idx(self, tok):
        return self.express_sorted_tok_to_idx.get(tok, self.express_sorted_unk_idx)

    def get_nucleotide_tok(self, ind):
        if self.alphabet_type == "gene":
            raise Exception("Not Support function for nucleotide")
        return self.all_nucleotide_toks[ind]

    def get_gene_tok(self, ind):
        return self.all_gene_toks[ind]

    def get_express_tok(self, ind):
        return self.all_express_toks[ind]

    def get_express_sorted_tok(self, ind):
        return self.all_express_sorted_toks[ind]

    def to_nucleotide_dict(self):
        if self.alphabet_type == "gene":
            raise Exception("Not Support function for nucleotide")
        return self.nucleotide_tok_to_idx.copy()

    def to_gene_dict(self):
        return self.gene_tok_to_idx.copy()

    def to_express_dict(self):
        return self.express_tok_to_idx.copy()

    def to_express_sorted_dict(self):
        return self.express_sorted_tok_to_idx.copy()

    @classmethod
    def from_predefined(
            cls,
            name: str,
            gene_id_list_filepath: str = None,
            express_bin_list_filepath: str = None,
            express_sorted_list_filepath: str = None,
            tokenizer_dir: str = None,
            use_express_sorted=False
    ):
        if tokenizer_dir is not None:
            # 如果不是路径而是文件名，则从tokenizer_dir下找
            if gene_id_list_filepath and not os.path.exists(gene_id_list_filepath):
                gene_id_list_filepath = os.path.join(tokenizer_dir, gene_id_list_filepath)
            if express_bin_list_filepath and not os.path.exists(express_bin_list_filepath):
                express_bin_list_filepath = os.path.join(tokenizer_dir, express_bin_list_filepath)
            if express_sorted_list_filepath and not os.path.exists(express_sorted_list_filepath):
                express_sorted_list_filepath = os.path.join(tokenizer_dir, express_sorted_list_filepath)

        if name == "nucleotide":
            return cls(
                alphabet_type=name,
                nucleotide_standard_toks=nucleotide_standard_toks,
                nucleotide_prepend_toks=nucleotide_prepend_toks,
                nucleotide_append_toks=nucleotide_append_toks,
                nucleotide_prepend_bos=True,
                nucleotide_append_eos=True,
                gene_standard_toks=None,
                gene_prepend_toks=gene_prepend_toks,
                gene_append_toks=gene_append_toks,
                gene_prepend_bos=True,
                gene_append_eos=True,
                express_standard_toks=express_standard_toks,
                express_prepend_toks=express_prepend_toks,
                express_append_toks=express_append_toks,
                express_sorted_standard_toks=express_sorted_standard_toks,
                express_sorted_prepend_toks=express_sorted_prepend_toks,
                express_sorted_append_toks=express_sorted_append_toks,
                express_prepend_bos=True,
                express_append_eos=True,
                gene_id_list_filepath=gene_id_list_filepath,
                express_bin_list_filepath=express_bin_list_filepath,
                express_sorted_list_filepath=express_sorted_list_filepath,
                use_express_sorted=use_express_sorted
            )
        else:
            return cls(
                alphabet_type=name,
                nucleotide_standard_toks=None,
                nucleotide_prepend_toks=None,
                nucleotide_append_toks=None,
                nucleotide_prepend_bos=None,
                nucleotide_append_eos=None,
                gene_standard_toks=gene_standard_toks,
                gene_prepend_toks=gene_prepend_toks,
                gene_append_toks=gene_append_toks,
                gene_prepend_bos=True,
                gene_append_eos=True,
                express_standard_toks=express_standard_toks,
                express_prepend_toks=express_prepend_toks,
                express_append_toks=express_append_toks,
                express_sorted_standard_toks=express_sorted_standard_toks,
                express_sorted_prepend_toks=express_sorted_prepend_toks,
                express_sorted_append_toks=express_sorted_append_toks,
                express_prepend_bos=True,
                express_append_eos=True,
                gene_id_list_filepath=gene_id_list_filepath,
                express_bin_list_filepath=express_bin_list_filepath,
                express_sorted_list_filepath=express_sorted_list_filepath,
                use_express_sorted=use_express_sorted
            )

    @classmethod
    def from_pretrained(cls, dir_path):
        import os, pickle
        return pickle.load(open(os.path.join(dir_path, "alphabet.pkl"), "rb"))

    def save_pretrained(self, save_dir):
        import os, shutil, pickle
        if self.gene_id_list_filepath:
            shutil.copy(self.gene_id_list_filepath, save_dir)
        if self.express_bin_list_filepath:
            shutil.copy(self.express_bin_list_filepath, save_dir)
        if self.express_sorted_list_filepath:
            shutil.copy(self.express_sorted_list_filepath, save_dir)
        with open(os.path.join(save_dir, "alphabet.pkl"), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def nucleotide_tokenize(
            nucleotide,
            add_special_token=False
    ) -> Union[list[str], list[Any], list[Union[list[str], list[Any]]]]:
        if isinstance(nucleotide, str):
            if add_special_token:
                return ["[N_CLS]"] + list(nucleotide) + ["[N_SEP]"]
            else:
                return list(nucleotide)
        else:
            res = []
            for item in nucleotide:
                if add_special_token:
                    res.append(["[N_CLS]"] + list(item) + ["[N_SEP]"])
                else:
                    res.append(list(item))
            return res

    def nucleotide_encode(
            self,
            nucleotide,
            add_special_token=False
    ) -> Union[list[Any], list[list[Any]]]:
        if isinstance(nucleotide, str):
            return [self.nucleotide_tok_to_idx[tok] for tok in self.nucleotide_tokenize(nucleotide, add_special_token)]
        else:
            res = []
            for item in nucleotide:
                res.append([self.nucleotide_tok_to_idx[tok] for tok in self.nucleotide_tokenize(item, add_special_token)])
            return res

    @staticmethod
    def gene_tokenize(
            gene,
            add_special_token=False
    ) -> Union[list[list[str]], list[str]]:
        if isinstance(gene[0], list):
            res = []
            for item in gene:
                items = [str(v) for v in item]
                if add_special_token:
                    res.append(["[G_CLS]"] + items + ["[G_SEP]"])
                else:
                    res.append(items)
            return res
        else:
            items = [str(v) for v in gene]
            if add_special_token:
                return ["[G_CLS]"] + items + ["[G_SEP]"]
            else:
                return items

    def gene_encode(
            self,
            gene,
            add_special_token=False
    ) -> Union[list[list[Any]], list[Any]]:
        if isinstance(gene[0], list):
            res = []
            for item in gene:
                res.append([self.gene_tok_to_idx[tok] for tok in self.gene_tokenize(item, add_special_token)])
            return res
        else:
            return [self.gene_tok_to_idx[tok] for tok in self.gene_tokenize(gene, add_special_token)]

    @staticmethod
    def express_tokenize(
            express,
            add_special_token=False
    ) -> Union[list[list[str]], list[str]]:
        if isinstance(express[0], list):
            res = []
            for item in express:
                items = [str(v) for v in item]
                if add_special_token:
                    res.append(["[E_CLS]"] + items + ["[E_SEP]"])
                else:
                    res.append(items)
            return res
        else:
            items = [str(v) for v in express]
            if add_special_token:
                return ["[E_CLS]"] + items + ["[E_SEP]"]
            else:
                return items

    def express_encode(
            self,
            express,
            add_special_token=False
    ) -> Union[list[list[Any]], list[Any]]:
        if isinstance(express[0], list):
            res = []
            for item in express:
                res.append([self.express_tok_to_idx[tok] for tok in self.express_tokenize(item, add_special_token)])
            return res
        else:
            return [self.express_tok_to_idx[tok] for tok in self.express_tokenize(express, add_special_token)]

    def express_encode_for_eval_mask(
            self,
            express,
            add_special_token=False
    ):
        return [
            self.express_tok_to_idx[tok] if tok != '-' else self.express_tok_to_idx["[E_MASK]"]
            for tok in self.express_tokenize(express, add_special_token=add_special_token)
        ]



