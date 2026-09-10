#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/12/10 20:18
@project: LucaOnePlusTasks
@file: predict
@desc: predict or inference for trained downstream models
'''
import csv
import json
import os, sys
import torch
import time
import numpy as np
import argparse
from collections import OrderedDict
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import to_device, device_memory, available_gpu_id, load_labels, seq_type_is_match_seq, \
        download_trained_checkpoint_lucaone_v2, download_trained_checkpoint_lucacell
    from common.multi_label_metrics import relevant_indexes
    from encoder import Encoder
    from batch_converter import BatchConverter
    from llm.lucaone.v2_0.alphabet import Alphabet as AlphabetLucaOne
    from llm.lucacell.models.alphabet import Alphabet as AlphabetLucaCell
    from file_operator import csv_reader, fasta_reader, csv_writer, file_reader
    from common.model_config import LucaConfig
    from lucasingle.models.LucaSingle import LucaSingle
    from lucapair.models.LucaPairHomo import LucaPairHomo
    from lucapair.models.LucaPairHeter import LucaPairHeter
    from lucapair.models.LucaPairIntraInter import LucaPairIntraInter
except ImportError:
    from src.utils import to_device, device_memory, available_gpu_id, load_labels, seq_type_is_match_seq, \
        download_trained_checkpoint_lucaone_v2, download_trained_checkpoint_lucacell
    from src.common.multi_label_metrics import relevant_indexes
    from src.encoder import Encoder
    from src.batch_converter import BatchConverter
    from src.llm.lucaone.v2_0.alphabet import Alphabet as AlphabetLucaOne
    from src.llm.lucacell.models.alphabet import Alphabet as AlphabetLucaCell
    from src.file_operator import csv_reader, fasta_reader, csv_writer, file_reader
    from src.common.model_config import LucaConfig
    from src.lucasingle.models.LucaSingle import LucaSingle
    from src.lucapair.models.LucaPairHomo import LucaPairHomo
    from src.lucapair.models.LucaPairHeter import LucaPairHeter
    from src.lucapair.models.LucaPairIntraInter import LucaPairIntraInter


def transform_one_sample_2_feature(
        device,
        input_mode,
        encoder,
        batch_convecter,
        row
):
    batch_info = []

    if input_mode == "pair" and args.input_type in ["matrix_vs_matrix", "vector_vs_vector", "vector_vs_matrix", "matrix_vs_vector"]:
        sample_ids = [row[0] + "#" + row[1]]
        en = encoder.encode_pair_cell(
            *row[0:12],
            label=None
        )
        en_list = en
        batch_info.append([row[0], row[1], row[2], row[3]])
        gene_nums = [len(row[4]), len(row[5])]
    elif input_mode == "pair" and args.input_type in ["gene_vs_matrix", "matrix_vs_gene"]:
        sample_ids = [row[0] + "#" + row[1]]
        en = encoder.encode_pair_gene_cell(
            *row[0:12],
            label=None
        )
        en_list = en
        batch_info.append([row[0], row[1], row[2], row[3]])
        gene_nums = [len(row[4]), len(row[5])]
    else:
        sample_ids = [row[0]]
        en = encoder.encode_single_cell(
            *row[0:6],
            label=None
        )
        en_list = en
        gene_nums = len(row[2])
        batch_info.append([row[0], row[1]])
    batch = [en_list]
    if isinstance(batch[0], list):
        batch_features = []
        for cur_batch in batch[0]:
            cur_batch_features = batch_convecter([cur_batch])
            cur_batch_features, cur_sample_num = to_device(device, cur_batch_features)
            batch_features.append(cur_batch_features)

    else:
        batch_features = batch_convecter(batch)
        batch_features, cur_sample_num = to_device(device, batch_features)
    return batch_info, batch_features, [gene_nums], sample_ids


def predict_probs(args, encoder, batch_convecter, model, row):
    batch_info, batch_features, gene_nums, sample_ids = transform_one_sample_2_feature(
        args.device,
        args.input_mode,
        encoder,
        batch_convecter,
        row
    )
    if isinstance(batch_features, list):
        probs = []
        for cur_batch_features in batch_features:
            cur_probs = model(
                **cur_batch_features,
                sample_ids=sample_ids,
                attention_scores_savepath=args.output_attention_scores_dirpath,
                attention_pooling_scores_savepath=args.output_attention_pooling_scores_dirpath,
                output_classification_vector_dirpath=args.output_classification_vector_dirpath
            )[1]
            if cur_probs.is_cuda:
                cur_probs = cur_probs.detach().cpu().numpy()
            else:
                cur_probs = cur_probs.detach().numpy()
            probs.append(cur_probs)
    else:
        probs = model(
            **batch_features,
            sample_ids=sample_ids,
            attention_scores_savepath=args.output_attention_scores_dirpath,
            attention_pooling_scores_savepath=args.output_attention_pooling_scores_dirpath,
            output_classification_vector_dirpath=args.output_classification_vector_dirpath
        )[1]
        if probs.is_cuda:
            probs = probs.detach().cpu().numpy()
        else:
            probs = probs.detach().numpy()
    return batch_info, probs, gene_nums


def predict_cell_level_binary_class(args, encoder, batch_convecter, label_id_2_name, model, row):
    batch_info, probs, gene_nums = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    preds = (probs >= args.threshold).astype(int).flatten()
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "pair":
            cur_res = [info[0], info[1], info[2], info[3], float(probs[idx][0]), label_id_2_name[preds[idx]]]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [info[0], info[1], float(probs[idx][0]), label_id_2_name[preds[idx]]]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_cell_level_multi_class(args, encoder, batch_convecter, label_id_2_name, model, row, topk=5):
    batch_info, probs, gene_nums = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)

    if topk is not None and topk > 1:
        # print("topk: %d" % topk)
        preds = np.argmax(probs, axis=-1)
        preds_topk = np.argsort(probs, axis=-1)[:, ::-1][:, :topk]
        res = []
        for idx, info in enumerate(batch_info):
            cur_topk_probs = []
            cur_topk_labels = []
            for label_idx in preds_topk[idx]:
                cur_topk_probs.append(float(probs[idx][label_idx]))
                cur_topk_labels.append(label_id_2_name[label_idx])
            if args.input_mode == "pair":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    float(probs[idx][preds[idx]]),
                    label_id_2_name[preds[idx]],
                    cur_topk_probs,
                    cur_topk_labels
                ]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [
                    info[0],
                    info[1],
                    float(probs[idx][preds[idx]]),
                    label_id_2_name[preds[idx]],
                    cur_topk_probs,
                    cur_topk_labels
                ]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res
    else:
        preds = np.argmax(probs, axis=-1)
        res = []
        for idx, info in enumerate(batch_info):
            if args.input_mode == "pair":
                cur_res = [info[0], info[1], info[2], info[3],  float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]]]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [info[0], info[1], float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]]]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res


def predict_cell_level_multi_label(args, encoder, batch_convecter, label_id_2_name, model, row):
    batch_info, probs, gene_nums = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    preds = relevant_indexes((probs >= args.threshold).astype(int))
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "pair":
            cur_res = [info[0], info[1], info[2], info[3], [float(probs[idx][label_index]) for label_index in preds[idx]], [label_id_2_name[label_index] for label_index in preds[idx]]]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [info[0], info[1], [float(probs[idx][label_index]) for label_index in preds[idx]], [label_id_2_name[label_index] for label_index in preds[idx]]]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_cell_level_regression(args, encoder, batch_convecter, label_id_2_name, model, row):
    batch_info, probs, gene_nums = predict_probs(args, encoder, batch_convecter, model, row)
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "pair":
            if args.num_labels == 1:
                cur_res = [info[0], info[1], info[2], info[3], float(probs[idx][0]), str(probs[idx][0])]
            else:
                res_list = [float(v) for v in probs[idx]]
                cur_res = [info[0], info[1], info[2], info[3], res_list, str(res_list)]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            if args.num_labels == 1:
                cur_res = [info[0], info[1], float(probs[idx][0]), str(probs[idx][0])]
            else:
                res_list = [float(v) for v in probs[idx]]
                cur_res = [info[0], info[1], res_list, str(res_list)]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def load_tokenizer(args, seq_tokenizer_class, cell_tokenizer_class):
    seq_tokenizer = seq_tokenizer_class.from_predefined("gene_prot")
    if args.not_seq_prepend_bos:
        seq_tokenizer.seq_prepend_bos = False
    if args.not_seq_append_eos:
        seq_tokenizer.seq_append_eos = False
    cell_tokenizer = cell_tokenizer_class.from_predefined(
        "nucleotide",
        gene_id_list_filepath=None,
        express_bin_list_filepath=None,
        express_sorted_list_filepath=None,
    )
    if args.not_cell_prepend_bos:
        cell_tokenizer.nucleotide_prepend_bos = False
        cell_tokenizer.gene_prepend_bos = False
        cell_tokenizer.express_prepend_bos = False
    if args.not_cell_append_eos:
        cell_tokenizer.nucleotide_append_eos = False
        cell_tokenizer.gene_append_eos = False
        cell_tokenizer.express_append_eos = False
    return seq_tokenizer, cell_tokenizer


def load_trained_model(model_config, args, model_class, model_dirpath):
    # load exists checkpoint
    print("load pretrained model: %s" % model_dirpath)
    try:
        model = model_class.from_pretrained(model_dirpath, config=model_config, args=args)
    except Exception as e:
        model = model_class(model_config, args=args)
        pretrained_net_dict = torch.load(
            os.path.join(model_dirpath, "pytorch.pth"),
            map_location=torch.device("cpu"),
            weights_only=True
        )
        model_state_dict_keys = set()
        for key in model.state_dict():
            model_state_dict_keys.add(key)
        new_state_dict = OrderedDict()
        for k, v in pretrained_net_dict.items():
            if k.startswith("module."):
                # remove `module.`
                name = k[7:]
            else:
                name = k
            if name in model_state_dict_keys:
                new_state_dict[name] = v
        diff = model_state_dict_keys.difference(new_state_dict.keys())
        if len(diff) > 0:
            print("diff:")
            print(diff)
        model.load_state_dict(new_state_dict)
    model.to(args.device)
    model.eval()
    return model


def load_model(args, model_dir):
    # load tokenizer and model
    begin_time = time.time()
    device = torch.device(args.device)
    print("load model on cuda:", device)
    if args.model_type == "lucapair_homo":
        config_class, seq_tokenizer_class, cell_tokenizer_class, model_class = LucaConfig, AlphabetLucaOne, AlphabetLucaCell, LucaPairHomo
    elif args.model_type == "lucapair_heter":
        config_class, seq_tokenizer_class, cell_tokenizer_class, model_class = LucaConfig, AlphabetLucaOne, AlphabetLucaCell, LucaPairHeter
    elif args.model_type == "lucapair_intrainter":
        config_class, seq_tokenizer_class, cell_tokenizer_class, model_class = LucaConfig, AlphabetLucaOne, AlphabetLucaCell, LucaPairIntraInter
    elif args.model_type == "lucasingle":
        config_class, seq_tokenizer_class, cell_tokenizer_class, model_class = LucaConfig, AlphabetLucaOne, AlphabetLucaCell, LucaSingle
    else:
        raise Exception("Not support the model_type=%s" % args.model_type)
    seq_tokenizer, cell_tokenizer = load_tokenizer(args, seq_tokenizer_class, cell_tokenizer_class)
    model_config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r")))
    model_config.num_labels = args.num_labels
    model = load_trained_model(model_config, args, model_class, model_dir)
    print("the time for loading model:", time.time() - begin_time)

    return model_config, seq_tokenizer, cell_tokenizer, model


def create_encoder_batch_convecter(
        model_args,
        seq_tokenizer,
        cell_tokenizer
):
    encoder_config = {
        "input_type": model_args.input_type,
        "trunc_type": model_args.trunc_type,
        "seq_llm_dirpath": model_args.seq_llm_dirpath,
        "seq_llm_type": model_args.seq_llm_type,
        "seq_llm_version": model_args.seq_llm_version,
        "seq_llm_step": model_args.seq_llm_step,
        "seq_tokenizer": seq_tokenizer,
        "seq_max_length": model_args.seq_max_length,
        "seq_prepend_bos": True,
        "seq_append_eos": True,
        "seq_matrix_add_special_token": model_args.seq_matrix_add_special_token,
        "seq_vector_dirpath": model_args.seq_vector_dirpath,
        "seq_matrix_dirpath": model_args.seq_matrix_dirpath,
        "seq_embedding_vector_type": model_args.seq_embedding_vector_type,
        "seq_embedding_complete": model_args.seq_embedding_complete,
        "seq_embedding_complete_seg_overlap": model_args.seq_embedding_complete_seg_overlap,
        "seq_embedding_fixed_len_a_time": model_args.seq_embedding_fixed_len_a_time,
        "seq_vector_embedding_exists": model_args.seq_vector_embedding_exists,
        "seq_matrix_embedding_exists": model_args.seq_matrix_embedding_exists,
        "seq_meta_fasta": model_args.seq_meta_fasta,
        "cell_llm_dirpath": model_args.cell_llm_dirpath,
        "cell_llm_type": model_args.cell_llm_type,
        "cell_llm_version": model_args.cell_llm_version,
        "cell_llm_step": model_args.cell_llm_step,
        "cell_tokenizer": cell_tokenizer,
        "cell_max_length": model_args.cell_max_length,
        "cell_prepend_bos": True,
        "cell_append_eos": True,
        "cell_matrix_add_special_token": model_args.cell_matrix_add_special_token,
        "cell_vector_dirpath": model_args.cell_vector_dirpath,
        "cell_matrix_dirpath": model_args.cell_matrix_dirpath,
        "cell_embedding_vector_type": model_args.cell_embedding_vector_type,
        "cell_embedding_complete": model_args.cell_embedding_complete,
        "cell_embedding_complete_seg_overlap": model_args.cell_embedding_complete_seg_overlap,
        "cell_embedding_fixed_len_a_time": model_args.cell_embedding_fixed_len_a_time,
        "cell_matrix_embedding_exists": model_args.cell_matrix_embedding_exists,
        "gpu_id": model_args.gpu_id,
        "lucacell_finetune": model_args.model_type == "lucacell_finetune",
        "not_frozen_gene_express_bin_embedding": model_args.not_frozen_gene_express_bin_embedding
    }
    encoder = Encoder(**encoder_config)

    batch_converter = BatchConverter(
        task_level_type=model_args.task_level_type,
        label_size=model_args.label_size,
        output_mode=model_args.output_mode,
        seq_tokenizer=seq_tokenizer,
        seq_no_position_embeddings=model_args.seq_no_position_embeddings,
        seq_no_token_type_embeddings=model_args.seq_no_token_type_embeddings,
        seq_truncation_length=model_args.seq_max_length,
        seq_matrix_add_special_token=model_args.seq_matrix_add_special_token,
        seq_prepend_bos=not model_args.not_seq_prepend_bos,
        seq_append_eos=not model_args.not_seq_append_eos,
        cell_tokenizer=cell_tokenizer,
        cell_no_position_embeddings=model_args.cell_no_position_embeddings,
        cell_no_token_type_embeddings=model_args.cell_no_token_type_embeddings,
        cell_truncation_length=model_args.cell_max_length,
        cell_matrix_add_special_token=model_args.cell_matrix_add_special_token,
        cell_prepend_bos=not model_args.not_cell_prepend_bos,
        cell_append_eos=not model_args.not_cell_append_eos,
        ignore_index=model_args.ignore_index,
        padding_idx=seq_tokenizer.padding_idx,
        unk_idx=seq_tokenizer.unk_idx,
        cls_idx=seq_tokenizer.cls_idx,
        eos_idx=seq_tokenizer.eos_idx,
        mask_idx=seq_tokenizer.mask_idx,
        non_ignore=model_args.non_ignore,
        input_type=model_args.input_type,
        trunc_type=model_args.trunc_type,
        lucacell_finetune=model_args.model_type == "lucacell_finetune",
        not_frozen_gene_express_bin_embedding=model_args.not_frozen_gene_express_bin_embedding
    )
    return encoder, batch_converter

# global
global_model_config, global_seq_tokenizer, global_cell_tokenizer, global_trained_model = None, None, None, None


def run(
        samples,
        seq_max_length,
        seq_emb_dir,
        seq_vector_embedding_exists,
        seq_matrix_embedding_exists,
        cell_max_length,
        cell_emb_dir,
        cell_matrix_embedding_exists,
        model_path,
        dataset_name,
        dataset_type,
        task_type,
        task_level_type,
        model_type,
        input_type,
        time_str,
        step,
        gpu_id,
        threshold,
        topk,
        output_attention_scores_dirpath,
        output_attention_pooling_scores_dirpath,
        output_classification_vector_dirpath
):
    global global_model_config, global_seq_tokenizer, global_cell_tokenizer, global_trained_model
    model_dir = "%s/models/%s/%s/%s/%s/%s/%s/%s" % (
        model_path, dataset_name, dataset_type, task_type, model_type, input_type,
        time_str, step if step == "best" else "checkpoint-{}".format(step)
    )
    config_dir = "%s/logs/%s/%s/%s/%s/%s/%s" % (
        model_path, dataset_name, dataset_type, task_type, model_type, input_type,
        time_str
    )
    model_args = torch.load(os.path.join(model_dir, "training_args.bin"), weights_only=False)

    print("-" * 25 + "Trained Model Args" + "-" * 25)
    print(model_args.__dict__)
    print("-" * 50)
    model_args.output_attention_scores_dirpath = output_attention_scores_dirpath
    model_args.output_attention_pooling_scores_dirpath = output_attention_pooling_scores_dirpath
    model_args.output_classification_vector_dirpath = output_classification_vector_dirpath

    model_args.seq_max_length = seq_max_length
    model_args.seq_vector_embedding_exists = seq_vector_embedding_exists
    model_args.seq_matrix_embedding_exists = seq_matrix_embedding_exists
    model_args.seq_emb_dir = seq_emb_dir
    if model_args.seq_emb_dir:
        seq_emb_dirs = model_args.seq_emb_dir.split("#")
    else:
        seq_emb_dirs = None
    if not seq_vector_embedding_exists:
        if seq_emb_dirs:
            for seq_emb_dir in seq_emb_dirs:
                if not os.path.exists(seq_emb_dir):
                    os.makedirs(seq_emb_dir)
            model_args.seq_vector_dirpath = seq_emb_dirs[0] if len(seq_emb_dirs) == 1 else model_args.seq_emb_dir
        else:
            model_args.seq_vector_dirpath = None
    else:
        if seq_emb_dirs:
            for seq_emb_dir in seq_emb_dirs:
                assert os.path.exists(seq_emb_dir)
            model_args.seq_vector_dirpath = model_args.seq_emb_dir
        else:
            raise ValueError("seq_vector_embedding_exists=True, but --seq_emb_dir is None or not exists")
    if not seq_matrix_embedding_exists:
        if seq_emb_dirs:
            for seq_emb_dir in seq_emb_dirs:
                if not os.path.exists(seq_emb_dir):
                    os.makedirs(seq_emb_dir)
            model_args.seq_matrix__dirpath = seq_emb_dirs[0] if len(seq_emb_dirs) == 1 else model_args.seq_emb_dir
        else:
            model_args.seq_matrix__dirpath = None
    else:
        if seq_emb_dirs:
            for seq_emb_dir in seq_emb_dirs:
                assert os.path.exists(seq_emb_dir)
            model_args.seq_matrix__dirpath = model_args.seq_emb_dir
        else:
            raise ValueError("seq_matrix_embedding_exists=True, but --seq_emb_dir is None or not exists")

    model_args.cell_max_length = cell_max_length
    model_args.cell_matrix_embedding_exists = cell_matrix_embedding_exists
    model_args.cell_emb_dir = cell_emb_dir
    if not cell_matrix_embedding_exists:
        if model_args.cell_emb_dir:
            cell_emb_dirs = model_args.cell_emb_dir.split("#")
            for cell_emb_dir in cell_emb_dirs:
                if not os.path.exists(cell_emb_dir):
                    os.makedirs(cell_emb_dir)
            model_args.cell_vector_dirpath = cell_emb_dirs[0] if len(cell_emb_dirs) == 1 else cell_emb_dirs
            model_args.cell_matrix_dirpath = cell_emb_dirs[0] if len(cell_emb_dirs) == 1 else cell_emb_dirs
        else:
            model_args.cell_vector_dirpath = None
            model_args.cell_matrix_dirpath = None

    model_args.dataset_name = dataset_name
    model_args.dataset_type = dataset_type
    model_args.task_type = task_type
    model_args.task_level_type = task_level_type
    model_args.model_type = model_type
    model_args.input_type = input_type
    model_args.time_str = time_str
    model_args.step = step
    model_args.threshold = threshold
    model_args.gpu_id = gpu_id

    if model_args.label_filepath:
        model_args.label_filepath = model_args.label_filepath.replace("../", "%s/" % model_path)
    if not os.path.exists(model_args.label_filepath):
        model_args.label_filepath = os.path.join(config_dir, "label.txt")

    if gpu_id is None or gpu_id < 0:
        # gpu_id = available_gpu_id()
        gpu_id = -1
        model_args.gpu_id = gpu_id
    print("------Before loading the model:------")
    print("GPU ID: %d" % gpu_id)
    model_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    device_memory(None if gpu_id == -1 else gpu_id)

    # Step2: loading the tokenizer and model
    if global_trained_model is None or next(global_trained_model.parameters()).device != model_args.device:
        global_trained_model = None
        model_config, seq_tokenizer, cell_tokenizer, trained_model = load_model(model_args, model_dir=model_dir)
        global_model_config = model_config
        global_seq_tokenizer = seq_tokenizer
        global_cell_tokenizer = cell_tokenizer
        global_trained_model = trained_model
    else:
        model_config = global_model_config
        seq_tokenizer = global_seq_tokenizer
        cell_tokenizer = global_cell_tokenizer
        trained_model = global_trained_model

    print("------After loaded the model:------")
    device_memory(None if gpu_id == -1 else gpu_id)
    encoder, batch_convecter = create_encoder_batch_convecter(model_args, seq_tokenizer, cell_tokenizer)

    label_list = load_labels(model_args.label_filepath)
    label_id_2_name = {idx: name for idx, name in enumerate(label_list)}

    # Step 3: prediction
    if model_args.task_level_type in ["cell_level", "cell-level"] and task_type in ["binary_class", "binary-class"]:
        predict_func = predict_cell_level_binary_class
    elif model_args.task_level_type in ["cell_level", "cell-level"] and task_type in ["multi_class", "multi-class"]:
        predict_func = predict_cell_level_multi_class
    elif model_args.task_level_type in ["cell_level", "cell-level"] and task_type in ["multi_label", "multi-label"]:
        predict_func = predict_cell_level_multi_label
    elif model_args.task_level_type in ["cell_level", "cell-level"] and task_type in ["regression"]:
        predict_func = predict_cell_level_regression
    else:
        raise Exception("the task_type=%s or task_level_type=%s error" % (task_type, model_args.task_level_type))

    predicted_results = []
    print()
    print("Device:", model_args.device)
    if hasattr(model_args, "input_mode") and model_args.input_mode in ["pair"]:
        for sample in samples:
            if task_level_type in ["cell_level", "cell-level"] and task_type in ["multi_class", "multi-class"]:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    sample,
                    topk=topk
                )
                if topk is not None and topk > 1:
                    predicted_results.append([
                        sample[0],
                        sample[1],
                        sample[2],
                        sample[3],
                        cur_res[0][4],
                        cur_res[0][5],
                        cur_res[0][6],
                        cur_res[0][7]
                    ])
                else:
                    predicted_results.append([
                        sample[0],
                        sample[1],
                        sample[2],
                        sample[3],
                        cur_res[0][4],
                        cur_res[0][5]
                    ])
            else:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    sample
                )
                predicted_results.append([
                    sample[0],
                    sample[1],
                    sample[2],
                    sample[3],
                    cur_res[0][4],
                    cur_res[0][5]
                ])
    else:
        for sample in samples:
            if task_level_type in ["cell_level", "cell-level"] and task_type in ["multi_class", "multi-class"]:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    sample,
                    topk=topk
                )
                if topk is not None and topk > 1:
                    predicted_results.append([sample[0], sample[1], cur_res[0][2], cur_res[0][3], cur_res[0][4], cur_res[0][5]])
                else:
                    predicted_results.append([sample[0], sample[1], cur_res[0][2], cur_res[0][3]])
            else:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    sample
                )
                predicted_results.append([sample[0], sample[1], cur_res[0][2], cur_res[0][3]])
    torch.cuda.empty_cache()
    return predicted_results


def run_args():
    parser = argparse.ArgumentParser(description="Prediction")
    # for one single sample of the input
    parser.add_argument("--sample_id", default=None, type=str,  help="the sample id")
    parser.add_argument("--sample_type", default=None, type=str, choices=["cell", "gene"], help="the sample type.")
    parser.add_argument("--sample_gene_id_list", default=None, type=str,  help="the sample gene id list")
    parser.add_argument("--sample_gene_seq_type_list", default=None, type=str,  help="the sample gene seq type list")
    parser.add_argument("--sample_gene_seq_list", default=None, type=str,  help="the sample gene seq list")
    parser.add_argument("--sample_gene_express_bin_list", default=None, type=str,  help="the sample gene express bin list")

    # for one pair sample of the input
    parser.add_argument("--sample_id_a", default=None, type=str,  help="the sample a id")
    parser.add_argument("--sample_type_a", default=None, type=str, choices=["cell", "gene"], help="the sample a type.")
    parser.add_argument("--sample_gene_id_list_a", default=None, type=str,  help="the sample a gene id list")
    parser.add_argument("--sample_gene_seq_type_list_a", default=None, type=str,  help="the sample a gene seq type list")
    parser.add_argument("--sample_gene_seq_list_a", default=None, type=str,  help="the sample a gene seq list")
    parser.add_argument("--sample_gene_express_bin_list_a", default=None, type=str,  help="the sample a gene express bin list")
    parser.add_argument("--sample_id_b", default=None, type=str,  help="the sample b id")
    parser.add_argument("--sample_type_b", default=None, type=str, choices=["cell", "gene"], help="the sample b type.")
    parser.add_argument("--sample_gene_id_list_b", default=None, type=str,  help="the sample b gene id list")
    parser.add_argument("--sample_gene_seq_type_list_b", default=None, type=str,  help="the sample b gene seq type list")
    parser.add_argument("--sample_gene_seq_list_b", default=None, type=str,  help="the sample gene b seq list")
    parser.add_argument("--sample_gene_express_bin_list_b", default=None, type=str,  help="the sample b gene express bin list")

    # for many samples
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="the fasta or csv format file for single-seq model, or the csv format file for pair-seq model"
    )

    # for seq embedding
    parser.add_argument("--seq_fasta_filepath", default=None, type=str, help="the gene seq fasta filepath")
    parser.add_argument("--seq_max_length", default=10242, type=int, required=True,
                        help="the max seq-length for seq-llm embedding")
    parser.add_argument("--seq_vector_embedding_exists", action="store_true",
                        help="the seq vector embedding is or not in advance. default: False")
    parser.add_argument("--seq_matrix_embedding_exists", action="store_true",
                        help="the seq matrix embedding is or not in advance. default: False")
    parser.add_argument("--seq_emb_dir", default=None, type=str,
                        help="the seq embedding save dir. default: None")
    # for cell embedding
    parser.add_argument("--cell_max_length", default=10242, type=int, required=True,
                        help="the max cell-length for seq-llm embedding")
    parser.add_argument("--cell_matrix_embedding_exists", action="store_true",
                        help="the cell embedding is or not in advance. default: False")
    parser.add_argument("--cell_emb_dir", default=None, type=str,
                        help="the cell embedding save dir. default: None")

    # for downstream tasks trained model
    parser.add_argument("--model_path", default=None, type=str, help="the model dir. default: None")
    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="the dataset name for model building.")
    parser.add_argument("--dataset_type", default=None, type=str, required=True,
                        help="the dataset type for model building.")
    parser.add_argument("--task_type", default=None, type=str, required=True,
                        choices=["multi_label", "multi_class", "binary_class", "regression"],
                        help="the task type for model building.")
    parser.add_argument("--task_level_type", default=None, type=str, required=True,
                        choices=["cell_level", "gene_level"],
                        help="the task level type for model building.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        choices=["lucasingle", "lucapair_homo", "lucapair_inter", "lucapair_heter", "lucapair_intrainter"],
                        help="the model type.")
    parser.add_argument("--input_type", default=None, type=str, required=True,
                        choices=["matrix", "vector", "matrix_vs_matrix", "gene_vs_matrix"],
                        help="the input type.")
    parser.add_argument("--input_mode", default=None, type=str, required=True,
                        choices=["single", "pair"],
                        help="the input mode.")
    parser.add_argument("--time_str", default=None, type=str, required=True,
                        help="the running time string(yyyymmddHimiss) of model building.")
    parser.add_argument("--step", default=None, type=str, required=True,
                        help="the training global checkpoint step of model finalization.")

    parser.add_argument("--topk", default=None, type=int, help="the topk labels for multi-class")
    parser.add_argument("--threshold",  default=0.5, type=float,
                        help="sigmoid threshold for binary-class or multi-label classification, "
                             "None for multi-class classification or regression, default: 0.5.")
    parser.add_argument("--ground_truth_idx", default=None, type=int,
                        help="the ground truth idx, when the input file contains")
    # for results(csv format, contain header)
    parser.add_argument("--save_path", default=None, type=str, help="the result save path")
    # for print info
    parser.add_argument("--print_per_num", default=10000, type=int, help="per num to print")
    parser.add_argument("--gpu_id", default=None, type=int, help="the used gpu index, -1 for cpu")

    parser.add_argument(
        "--output_attention_scores_dirpath",
        default=None,
        type=str,
        help="the save path output the attention scores(one file for each one sample)"
    )
    parser.add_argument(
        "--output_attention_pooling_scores_dirpath",
        default=None,
        type=str,
        help="the save path output the attention pooling scores(one file for each one sample)"
    )
    parser.add_argument(
        "--output_classification_vector_dirpath",
        default=None,
        type=str,
        help="the save path output the attention pooling scores(one file for each one sample)"
    )
    args_info = parser.parse_args()
    return args_info


def read_gene_fasta(gene_seq_fasta_filepath):
    seq_fasta = {}
    if gene_seq_fasta_filepath:
        if not os.path.exists(gene_seq_fasta_filepath):
            sys.exit("%s not exists" % gene_seq_fasta_filepath)
        for row in fasta_reader(gene_seq_fasta_filepath):
            seq_id = row[0]
            if seq_id[0] == ">":
                seq_id = seq_id[1:]
            seq = row[1].strip().upper()
            seq_fasta[seq_id] = seq
    return seq_fasta


if __name__ == "__main__":
    args = run_args()
    print("-" * 25 + "Run Args" + "-" * 25)
    args_dict = {}
    for item in args.__dict__.items():
        if item[1] is not None:
            args_dict[item[0]] = item[1]
    print(args_dict)
    print("-" * 50)

    if args.output_attention_scores_dirpath:
        args.output_attention_scores = True
        if not os.path.exists(args.output_attention_scores_dirpath):
            os.makedirs(args.output_attention_scores_dirpath)
    if args.output_attention_pooling_scores_dirpath:
        args.output_attention_pooling_scores = True
        if not os.path.exists(args.output_attention_pooling_scores_dirpath):
            os.makedirs(args.output_attention_pooling_scores_dirpath)
    if args.output_classification_vector_dirpath:
        args.output_classification_vectors = True
        if not os.path.exists(args.output_classification_vector_dirpath):
            os.makedirs(args.output_classification_vector_dirpath)


    seq_fasta = read_gene_fasta(args.seq_fasta_filepath)

    if args.input_mode == "pair" and args.input_file is not None:
        input_file_suffix = os.path.basename(args.input_file).split(".")[-1]
        if input_file_suffix not in ["csv", "tsv"]:
            print("Error! the input file is not in .csv or .tsv format for the pair seqs task.")
            sys.exit(-1)

    model_dir = "%s/models/%s/%s/%s/%s/%s/%s/%s" % (
        args.model_path, args.dataset_name, args.dataset_type, args.task_type, args.model_type, args.input_type,
        args.time_str, args.step if args.step == "best" else "checkpoint-{}".format(args.step)
    )
    model_args = torch.load(os.path.join(model_dir, "training_args.bin"), weights_only=False)

    # download LLM(LucaOne)
    print("#" * 25 + "LucaOne for Gene Seq Embedding" + "#" * 25)
    model_args.seq_llm_dirpath = download_trained_checkpoint_lucaone_v2(
        llm_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm/"),
        llm_type=model_args.seq_llm_type,
        llm_version=model_args.seq_llm_version,
        llm_step=model_args.seq_llm_step
    )

    # download LLM(LucaCell)
    print("#" * 25 + "LucaCell for Cell(Gene Seq List + Gene Expression List) Embedding" + "#" * 25)
    model_args.cell_llm_dirpath = download_trained_checkpoint_lucacell(
        llm_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm/"),
        llm_type=model_args.cell_llm_type,
        # llm_time_str=model_args.cell_llm_time_str,
        llm_version=model_args.cell_llm_version,
        llm_step=model_args.cell_llm_step
    )

    # download trained downstream task models
    '''
    download_trained_checkpoint_downstream_tasks(
        save_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    '''

    if args.input_file is not None and os.path.exists(args.input_file):
        exists_ids = set()
        exists_res = []
        if os.path.exists(args.save_path):
            print("save_path=%s exists." % args.save_path)
            if args.input_mode == "pair":
                for row in csv_reader(args.save_path, header=True, header_filter=True):
                    if len(row) < 12:
                        continue
                    exists_ids.add(row[0] + "#" + row[1])
                    exists_res.append(row)
                print("exists records: %d" % len(exists_res))
            else:
                for row in csv_reader(args.save_path, header=True, header_filter=True):
                    if len(row) < 6:
                        continue
                    exists_ids.add(row[0])
                    exists_res.append(row)
                print("exists records: %d" % len(exists_res))
        elif not os.path.exists(os.path.dirname(args.save_path)):
            os.makedirs(os.path.dirname(args.save_path))
        with open(args.save_path, "w") as wfp:
            writer = csv.writer(wfp)
            if args.input_mode == "pair":
                if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "cell_level":
                    header = ["sample_id_a", "sample_id_b", "sample_type_a", "sample_type_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                else:
                    header = ["sample_id_a", "sample_id_b", "sample_type_a", "sample_type_b", "prob", "label"]
            else:
                if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "cell_level":
                    header = ["sample_id", "sample_type", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                else:
                    header = ["sample_id", "sample_type", "prob", "label"]
            if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                header.append("ground_truth")
            writer.writerow(header)
            for item in exists_res:
                writer.writerow(item)
            exists_res = []
            batch_data = []
            batch_ground_truth = []
            had_done = 0

            reader = file_reader(args.input_file) if args.input_file.endswith(".csv") \
                                                     or args.input_file.endswith(".tsv") else fasta_reader(args.input_file)
            for row in reader:
                if args.input_mode == "pair":
                    if row[0] + "#" + row[1] in exists_ids:
                        continue
                    for col_idx in range(4, 12):
                        # 序列类型
                        if col_idx in [6, 7] and row[col_idx] and '[' not in row[col_idx]:
                            row[col_idx] = [row[col_idx]] * len(row[col_idx - 2])
                        else:
                            row[col_idx] = eval(row[col_idx]) if row[col_idx] else None
                    # the gene seq list not in csv, the get by gene id list
                    if row[8] is None or len(row[8]) == 0 and seq_fasta:
                        assert row[4] is not None and len(row[4]) > 0 or (row[0] is not None and args.seq_emb_dir is not None)
                        if row[4] is not None and len(row[4]) > 0:
                            row[8] = [seq_fasta[gene_id] for gene_id in row[4]] if seq_fasta else None
                    if row[9] is None or len(row[9]) == 0 and seq_fasta:
                        assert row[5] is not None and len(row[5]) > 0 or (row[1] is not None and args.cell_emb_dir is not None)
                        if row[5] is not None and len(row[5]) > 0:
                            row[9] = [seq_fasta[gene_id] for gene_id in row[5]] if seq_fasta else None
                    batch_data.append([*row[0:12]])
                    if args.ground_truth_idx is not None and args.ground_truth_idx >= 12:
                        batch_ground_truth.append(row[args.ground_truth_idx])
                else:
                    if row[0] in exists_ids:
                        continue
                    for col_idx in range(2, 6):
                        row[col_idx] = eval(row[col_idx]) if row[col_idx] is not None else None
                    # the gene seq list not in csv, the get by gene id list
                    if row[4] is None or len(row[4]) == 0:
                        assert row[2] is not None and len(row[2]) > 0
                        row[4] = [seq_fasta[gene_id] for gene_id in row[2]]

                    batch_data.append([*row[0:6]])
                    if args.ground_truth_idx is not None and args.ground_truth_idx >= 6:
                        batch_ground_truth.append(row[args.ground_truth_idx])

                if len(batch_data) % args.print_per_num == 0:
                    batch_results = run(
                        samples=batch_data,
                        seq_max_length=args.seq_max_length,
                        seq_emb_dir=args.seq_emb_dir,
                        seq_vector_embedding_exists=args.seq_vector_embedding_exists,
                        seq_matrix_embedding_exists=args.seq_matrix_embedding_exists,
                        cell_max_length=args.cell_max_length,
                        cell_emb_dir=args.cell_emb_dir,
                        cell_matrix_embedding_exists=args.cell_matrix_embedding_exists,
                        model_path=args.model_path,
                        dataset_name=args.dataset_name,
                        dataset_type=args.dataset_type,
                        task_type=args.task_type,
                        task_level_type=args.task_level_type,
                        model_type=args.model_type,
                        input_type=args.input_type,
                        time_str=args.time_str,
                        step=args.step,
                        gpu_id=args.gpu_id,
                        threshold=args.threshold,
                        topk=args.topk,
                        output_attention_scores_dirpath=args.output_attention_scores_dirpath,
                        output_attention_pooling_scores_dirpath=args.output_attention_pooling_scores_dirpath,
                        output_classification_vector_dirpath=args.output_classification_vector_dirpath
                    )
                    for item_idx, item in enumerate(batch_results):
                        if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                            item.append(batch_ground_truth[item_idx])
                        writer.writerow(item)
                    wfp.flush()
                    had_done += len(batch_data)
                    print("done %d, had_done: %d" % (len(batch_data), had_done))
                    batch_data = []
                    batch_ground_truth = []
            if len(batch_data) > 0:
                batch_results = run(
                    samples=batch_data,
                    seq_max_length=args.seq_max_length,
                    seq_emb_dir=args.seq_emb_dir,
                    seq_vector_embedding_exists=args.seq_vector_embedding_exists,
                    seq_matrix_embedding_exists=args.seq_matrix_embedding_exists,
                    cell_max_length=args.cell_max_length,
                    cell_emb_dir=args.cell_emb_dir,
                    cell_matrix_embedding_exists=args.cell_matrix_embedding_exists,
                    model_path=args.model_path,
                    dataset_name=args.dataset_name,
                    dataset_type=args.dataset_type,
                    task_type=args.task_type,
                    task_level_type=args.task_level_type,
                    model_type=args.model_type,
                    input_type=args.input_type,
                    time_str=args.time_str,
                    step=args.step,
                    gpu_id=args.gpu_id,
                    threshold=args.threshold,
                    topk=args.topk,
                    output_attention_scores_dirpath=args.output_attention_scores_dirpath,
                    output_attention_pooling_scores_dirpath=args.output_attention_pooling_scores_dirpath,
                    output_classification_vector_dirpath=args.output_classification_vector_dirpath
                )
                for item_idx, item in enumerate(batch_results):
                    if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                        item.append(batch_ground_truth[item_idx])
                    writer.writerow(item)
                wfp.flush()
                batch_data = []
                batch_ground_truth = []
    elif args.sample_id is not None and args.sample_type is not None:
        data = [
            [
                args.sample_id, args.sample_type,
                args.sample_gene_id_list,
                args.sample_gene_seq_type_list,
                args.sample_gene_seq_list,
                args.sample_gene_express_bin_list
            ]
        ]
        results = run(
            samples=data,
            seq_max_length=args.seq_max_length,
            seq_emb_dir=args.seq_emb_dir,
            seq_vector_embedding_exists=args.seq_vector_embedding_exists,
            seq_matrix_embedding_exists=args.seq_matrix_embedding_exists,
            cell_max_length=args.cell_max_length,
            cell_emb_dir=args.cell_emb_dir,
            cell_matrix_embedding_exists=args.cell_matrix_embedding_exists,
            model_path=args.model_path,
            dataset_name=args.dataset_name,
            dataset_type=args.dataset_type,
            task_type=args.task_type,
            task_level_type=args.task_level_type,
            model_type=args.model_type,
            input_type=args.input_type,
            time_str=args.time_str,
            step=args.step,
            gpu_id=args.gpu_id,
            threshold=args.threshold,
            topk=args.topk,
            output_attention_scores_dirpath=args.output_attention_scores_dirpath,
            output_attention_pooling_scores_dirpath=args.output_attention_pooling_scores_dirpath,
            output_classification_vector_dirpath=args.output_classification_vector_dirpath
        )
        print("Predicted Result:")
        print("sample_id=%s" % args.sample_id)
        print("sample_type=%s" % args.sample_type)
        print("prob=%f" % results[0][2])
        print("label=%s" % results[0][3])
    elif args.sample_id_a is not None and args.sample_id_b is not None and args.sample_type_a is not None and args.sample_type_b is not None:
        data = [[
            args.sample_id_a, args.sample_id_b,
            args.sample_type_a, args.sample_type_b,
            args.sample_gene_id_list_a, args.sample_gene_id_list_b,
            args.sample_gene_seq_type_list_a, args.sample_gene_seq_type_list_b,
            args.sample_gene_seq_list_a, args.sample_gene_seq_list_b,
            args.sample_gene_express_bin_list_a, args.sample_gene_express_bin_list_b
        ]]
        results = run(
            samples=data,
            seq_max_length=args.seq_max_length,
            seq_emb_dir=args.seq_emb_dir,
            seq_vector_embedding_exists=args.seq_vector_embedding_exists,
            seq_matrix_embedding_exists=args.seq_matrix_embedding_exists,
            cell_max_length=args.cell_max_length,
            cell_emb_dir=args.cell_emb_dir,
            cell_matrix_embedding_exists=args.cell_matrix_embedding_exists,
            model_path=args.model_path,
            dataset_name=args.dataset_name,
            dataset_type=args.dataset_type,
            task_type=args.task_type,
            task_level_type=args.task_level_type,
            model_type=args.model_type,
            input_type=args.input_type,
            time_str=args.time_str,
            step=args.step,
            gpu_id=args.gpu_id,
            threshold=args.threshold,
            topk=args.topk,
            output_attention_scores_dirpath=args.output_attention_scores_dirpath,
            output_attention_pooling_scores_dirpath=args.output_attention_pooling_scores_dirpath,
            output_classification_vector_dirpath=args.output_classification_vector_dirpath
        )
        print("Predicted Result:")
        print("sample_id_a=%s, sample_id_b" % args.sample_id_a)
        print("sample_type=%s" % args.sample_type)
        print("prob=%f" % results[0][2])
        print("label=%s" % results[0][3])
        print("Predicted Result:")
        print("seq_id=%s" % args.seq_id)
        print("seq=%s" % args.seq)
        print("prob=%f" % results[0][4])
        print("label=%s" % results[0][5])
    else:
        raise Exception("input error, usage: --hep")


"""
    python predict_v2.py \
        --model_path ../ \
        --dataset_name xxxx \
        --dataset_type xxxx \
        --task_type xxxx \
        --task_level_type xxxxx \
        --model_type lucasingle or lucapair_heter \
        --input_type gene_vs_matrix, matrix, matrix_vs_matrix \
        --input_mode single or pair \
        --time_str xxxx \
        --step xxxx \
        --input_file xxxx.csv \
        --seq_fasta_filepath xxxx \
        --seq_max_length xxxxx \
        --seq_emb_dir xxxx \
        --cell_max_length xxxx \
        --cell_emb_dir xxxx \
        --save_path xxxx \
        --output_attention_scores_dirpath xxxx \
        --output_attention_pooling_scores_dirpath \
        --output_classification_vector_dirpath \
        --print_per_num 1000 \
        --ground_truth_idx xxx(如果有) \
        --gpu_id xxxx
        
"""