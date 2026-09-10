#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/26 14:48
@project: LucaOne
@file: utils
@desc: utils for LucaOne
'''
import requests
import random, json
import os, sys, math
import pynvml, torch
import numpy as np
from collections import OrderedDict
sys.path.append("..")
sys.path.append("../src")
sys.path.append("../src/common")
try:
    from metrics import metrics_multi_class, metrics_binary, metrics_regression
except ImportError:
    from src.common.metrics import metrics_multi_class, metrics_binary, metrics_regression


def set_seed(args):
    '''
    Set seed for reproducibility
    :param args:
    :return:
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def to_device(
        device,
        batch,
        exclude=[
            "nucleotide_input_ids",
            "nucleotide_attention_mask",
            "nucleotide_position_ids",
            "nucleotide_input_embeds"
        ]
):
    '''
    input to device
    :param device:
    :param batch:
    :param exclude:
    :return:
    '''
    new_batch = {}
    sample_num = 0
    tens = None
    for item1 in batch.items():
        if isinstance(item1[1], dict):
            new_batch[item1[0]] = {}
            for item2 in item1[1].items():
                new_batch[item1[0]][item2[0]] = {}
                if isinstance(item2[1], dict):
                    for item3 in item2[1].items():
                        if item3[1] is not None:
                            if exclude and item3[0] in exclude:
                                new_batch[item1[0]][item2[0]][item3[0]] = item3[1]
                            else:
                                new_batch[item1[0]][item2[0]][item3[0]] = item3[1].to(device)
                            tens = item3[1]
                        else:
                            new_batch[item1[0]][item2[0]][item3[0]] = item3[1]
                else:
                    if item2[1] is not None:
                        if exclude and item2[0] in exclude:
                            new_batch[item1[0]][item2[0]] = item2[1]
                        else:
                            new_batch[item1[0]][item2[0]] = item2[1].to(device)
                        tens = item2[1]
                    else:
                        new_batch[item1[0]][item2[0]] = item2[1]
        else:
            if item1[1] is not None:
                if exclude and item1[0] in exclude:
                    new_batch[item1[0]] = item1[1]
                else:
                    new_batch[item1[0]] = item1[1].to(device)
                tens = item1[1]
            else:
                new_batch[item1[0]] = item1[1]
    if tens is not None:
        sample_num = tens.shape[0]
    return new_batch, sample_num


def is_scalar(variable):
    """
    判断一个变量是否为标量（单个值，没有或空的 shape）。
    支持的类型:
    - Python 内置 int, float, bool, complex
    - NumPy 标量或 0 维数组
    - PyTorch 0 维张量
    - TensorFlow 0 维张量 (如果已安装)
    """
    # 1. 首先判断是否是 Python 内置的数值或布尔类型
    if isinstance(variable, (int, float, bool, complex, str)):
        return True

    # 2. 接着处理类似数组/张量的对象
    # 使用 hasattr 避免在非数组对象上调用 .ndim 或 .shape 时出错
    if hasattr(variable, 'ndim'): # NumPy, PyTorch, TensorFlow 都有 .ndim 属性
        return variable.ndim == 0

    return False


def print_batch_input(batch):
    '''
    print output batch
    :param batch:
    :return:
    '''
    if isinstance(batch, list):
        for item in batch:
            print_batch_input(item)
    elif isinstance(batch, dict):
        for item in batch.items():
            print(item[0] + ":")
            print_batch_input(item[1])
    else:
        if batch is None:
            print("None")
        elif is_scalar(batch):
            print(batch)
        else:
            print(batch.shape)
            print(batch)
            print(torch.nonzero(torch.ne(batch, -100)))
        print("*" * 10)


def print_batch_output(batch):
    '''
    print output batch
    :param batch:
    :return:
    '''
    if isinstance(batch, list):
        for item in batch:
            print_batch_output(item)
    elif isinstance(batch, dict):
        for item in batch.items():
            print(item[0] + ":")
            print_batch_output(item[1])
    else:
        if batch is None:
            print("None")
        elif is_scalar(batch):
            print(batch)
        else:
            print(batch.shape)
            print(batch)
        print("*" * 10)


def print_batch_device(batch_input, level):
    for item in batch_input.items():
        print("level-%d:" % level + item[0])
        if isinstance(item[1], dict):
            print_batch_device(item[1], level=level+1)
        else:
            if item[1] is not None:
                print(item[1].shape)
                print(item[1].device)
            else:
                print("None")
            print("*" * 10)


def process_outputs(
        output_mode,
        truth,
        pred,
        output_truth,
        output_pred,
        ignore_index,
        keep_seq=False,
        return_masked_truth_pred=False
):
    # token_level/mask
    # truth: [N, max_seq_len] pred: [N, max_seq_len, vocab_size]
    # span_level/gene_type
    # truth: [N, max_seq_len] pred: [N, max_seq_len, label_size]
    # seq_level/gene_taxonomy
    # truth: [N, 1] pred: [N, label_size]

    # token_level/mask
    # truth: [N, max_seq_len] pred: [N, max_seq_len, vocab_size]
    # span_level/prot_homo
    # truth: [N, max_seq_len] pred: [N, max_seq_len, label_size]
    # span_level/prot_site
    # truth: [N, max_seq_len] pred: [N, max_seq_len, label_size]
    # seq_level/prot_taxonomy
    # truth: [N, 1] pred: [N, label_size]
    # seq_level/prot_keyword
    # truth: [N, label_size] pred: [N, label_size]
    # structure_level/prot_structure
    # truth: [N, max_seq_len, 3] pred: [N, max_seq_len, 3]

    # pair_level/trans
    # truth: [N, 1] pred: [N, 1]
    masked_preds = {}
    masked_truths = {}
    if keep_seq:
        # todo
        # token_level/mask
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # span_level/gene_type
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # seq_level/gene_taxonomy
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # token_level/mask
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # span_level/prot_homo
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # span_level/prot_site
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # seq_level/prot_taxonomy
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # seq_level/prot_keyword
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1)
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask]

        # structure_level/prot_structure
        truth = truth.view(-1, 3)
        truth_mask = truth[:, 0] != ignore_index
        pred = pred.view(-1, 3)
        masked_truth = truth[truth_mask, :]
        masked_pred = pred[truth_mask, :]

        # pair_level/trans
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1)
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask]
    else:
        for item1 in truth.items():
            task_name = item1[0]
            cur_truth = truth[task_name]
            cur_pred = pred[task_name]
            if output_mode[task_name] in ["multi_class", "multi-class"]:
                cur_truth = cur_truth.view(-1)
                cur_mask = cur_truth != ignore_index
                cur_pred = cur_pred.view(-1, cur_pred.shape[-1])
                cur_truth = cur_truth[cur_mask]
                cur_pred = cur_pred[cur_mask, :]
            elif output_mode[task_name] in ["multi_label", "multi-label", "binary_class", "binary-class"]:
                cur_truth = cur_truth.view(-1)
                cur_mask = cur_truth != ignore_index
                cur_pred = cur_pred.view(-1)
                cur_truth = cur_truth[cur_mask]
                cur_pred = cur_pred[cur_mask]
            elif output_mode[task_name] in ["regression"]:
                cur_truth = cur_truth.view(-1, 3)
                cur_mask = (cur_truth[:, 0] != ignore_index) | (cur_truth[:, 1] != ignore_index) | (cur_truth[:, 2] != ignore_index)
                cur_pred = cur_pred.view(-1, 3)
                cur_truth = cur_truth[cur_mask, :]
                cur_pred = cur_pred[cur_mask, :]
            else:
                raise Exception("not output mode: task_name=%s,, mode:%s" % (
                     task_name, output_mode[task_name]
                ))

            if cur_mask.sum().item() > 0:
                cur_truth = cur_truth.detach().cpu().numpy()
                cur_pred = cur_pred.detach().cpu().numpy()
                if task_name not in output_truth:
                    output_truth[task_name] = cur_truth
                    output_pred[task_name] = cur_pred
                else:
                    output_truth[task_name] = np.append(
                        output_truth[task_name],
                        cur_truth,
                        axis=0
                    )
                    output_pred[task_name] = np.append(
                        output_pred[task_name],
                        cur_pred,
                        axis=0
                    )
            else:
                cur_truth = None
                cur_pred = None
            masked_preds[task_name] = cur_pred
            masked_truths[task_name] = cur_truth
    if return_masked_truth_pred:
        return output_truth, output_pred, masked_truths, masked_preds
    return output_truth, output_pred


def concat_output(
        ground_truth,
        outputs,
        ground_truth_ids,
        pred_scores
):
    '''
    :param ground_truth:
    :param outputs:
    :param ground_truth_ids:
    :param pred_scores:
    :return:
    '''
    if pred_scores is None:
        pred_scores = outputs.detach().cpu().numpy()
        ground_truth_ids = ground_truth.detach().cpu().numpy()
    else:
        pred_scores = np.append(pred_scores, outputs.detach().cpu().numpy(), axis=0)
        ground_truth_ids = np.append(ground_truth_ids, ground_truth.detach().cpu().numpy(), axis=0)
    return ground_truth_ids, pred_scores


def concat_output_tensor(ground_truth, outputs, ground_truth_ids, pred_scores):
    '''
    :param ground_truth:
    :param outputs:
    :param ground_truth_ids:
    :param pred_scores:
    :return:
    '''
    if pred_scores is None:
        pred_scores = outputs
        ground_truth_ids = ground_truth
    else:
        pred_scores = torch.cat((pred_scores, outputs), dim=0)
        ground_truth_ids = torch.cat((ground_truth_ids, ground_truth), dim=0)
    return ground_truth_ids, pred_scores


def get_labels(label_filepath, header=True):
    '''
    get labels from file, exists header
    :param label_filepath:
    :return:
    '''
    with open(label_filepath, "r") as fp:
        labels = []
        multi_cols = False
        cnt = 0
        for line in fp.readlines():
            cnt += 1
            if cnt == 1 and header:
                if line.find(",") > 0:
                    multi_cols = True
                continue
            line = line.strip()
            if multi_cols:
                idx = line.find(",")
                if idx > 0:
                    label_name = line[idx + 1:].strip()
                else:
                    label_name = line
            else:
                label_name = line
            labels.append(label_name)
        return labels


def available_gpu_id():
    """
    计算可用的GPU id
    :return:
    """
    pynvml.nvmlInit()
    if not torch.cuda.is_available():
        print("GPU not available")
        return -1
    # 获取GPU数量
    device_count = pynvml.nvmlDeviceGetCount()
    max_available_gpu = -1
    max_available_rate = 0

    # 遍历所有GPU并检查可用性
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # 假设如果GPU利用率小于某个阈值（例如10%），我们认为这个GPU目前是空闲的
        if utilization.gpu < 10 and max_available_rate < 100 - utilization.gpu:
            max_available_rate = 100 - utilization.gpu
            max_available_gpu = i
    # 打印可用的GPU ID
    if max_available_gpu > -1:
        print("Available GPU ID: %d, Free Rate: %0.2f%%" % (max_available_gpu, max_available_rate))
    else:
        print("No Available GPU!")

    # Shutdown NVML
    pynvml.nvmlShutdown()
    return max_available_gpu


def eval_metrics(output_mode, truths, preds, threshold=0.5):
    '''
    eval metrics
    :param output_mode:
    :param truths:
    :param preds:
    :param threshold:
    :return:
    '''
    result = {}
    for item1 in truths.items():
        task_name = item1[0]
        cur_output_mode = output_mode[task_name]
        cur_truths = truths[task_name]
        cur_preds = preds[task_name]
        if cur_output_mode in ["multi-label", "multi_label"]:
            cur_result = metrics_binary(cur_truths, cur_preds, threshold=threshold)
        elif cur_output_mode in ["multi-class", "multi_class"]:
            cur_result = metrics_multi_class(cur_truths, cur_preds)
        elif cur_output_mode == "regression":
            cur_result = metrics_regression(cur_truths, cur_preds)
        elif cur_output_mode in ["binary-class", "binary_class"]:
            cur_result = metrics_binary(cur_truths, cur_preds, threshold=threshold)
        else:
            raise Exception("Not Support this output mode: %s, task_name=%s" % (cur_output_mode, task_name))
        result[task_name] = cur_result

    return result


def metrics_merge(results, all_results):
    for item1 in results.items():
        if item1[0] not in all_results:
            all_results[item1[0]] = {}
        for item2 in item1[1].items():
            if item2[0] not in all_results[item1[0]]:
                all_results[item1[0]][item2[0]] = item2[1]
            else:
                all_results[item1[0]][item2[0]] += item2[1]
    return all_results


def get_lr(optimizer):
    """
    get learning rate
    :param optimizer:
    :return:
    """
    for p in optimizer.param_groups:
        return p["lr"]


def get_parameter_number(model):
    '''
    colc the parameter number of the model
    :param model:
    :return:
    '''
    param_size = 0
    param_sum = 0
    trainable_size = 0
    trainable_num = 0
    for param in model.parameters():
        cur_size = param.nelement() * param.element_size()
        cur_num = param.nelement()
        param_size += cur_size
        param_sum += cur_num
        if param.requires_grad:
            trainable_size += cur_size
            trainable_num += cur_num
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    '''
    total_num = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    total_num += sum(p.numel() for p in model.buffers())
    total_size += sum(p.numel() * p.element_size() for p in model.buffers())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    '''
    return {
        'total_num': "%fM" % round((buffer_sum + param_sum) / (1024 * 1024), 2),
        'total_size': "%fMB" % round((buffer_size + param_size) / (1024 * 1024), 2),
        'param_sum': "%fM" % round(param_sum / (1024 * 1024), 2),
        'param_size': "%fMB" % round(param_size / (1024 * 1024), 2),
        'buffer_sum': "%fM" % round(buffer_sum / (1024 * 1024), 2),
        'buffer_size': "%fMB" % round(buffer_size / (1024 * 1024), 2),
        'trainable_num': "%fM" % round(trainable_num / (1024 * 1024), 2),
        'trainable_size': "%fMB" % round(trainable_size / (1024 * 1024), 2)
    }


def calc_loss_index(args):
    if not hasattr(args, "index_list"):
        index_list = []
        if "token_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
            index_list.append(0)
        if "span_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
            index_list.append(1)
        if "seq_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
            index_list.append(2)
        args.index_list = index_list
        args.task_num = len(index_list)


def writer_info_tb(tb_writer, logs, global_step, prefix=None):
    '''
    write info to tensorboard
    :param tb_writer:
    :param logs:
    :param global_step:
    :param prefix:
    :return:
    '''
    if prefix is None:
        prefix = ""
    elif prefix != "":
        prefix = prefix + "_"
    for key, value in logs.items():
        if isinstance(value, dict):
            writer_info_tb(tb_writer, value, global_step, prefix=prefix + key)
        elif isinstance(value, list):
            print("writer_info_tb List, Key-Value: %s=%s" % (key, str(value)))
        elif not math.isnan(value) and not math.isinf(value):
            try:
                tb_writer.add_scalar(prefix + key, value, global_step)
            except Exception as e:
                print(e)
        else:
            print("writer_info_tb NaN or Inf, Key-Value: %s=%s" % (key, value))


def calc_detail_losses(
        losses,
        total_losses,
        total_steps,
        log_total_losses=None,
        log_total_steps=None
):
    current_losses = {}
    for cur_losses in losses:
        for item in cur_losses.items():
            key = item[0]
            if key not in total_losses:
                total_losses[key] = 0
            if key not in current_losses:
                current_losses[key] = 0
            if key not in total_steps:
                total_steps[key] = 0
            if log_total_losses is not None:
                if key not in log_total_losses:
                    log_total_losses[key] = 0
            if log_total_steps is not None:
                if key not in log_total_steps:
                    log_total_steps[key] = 0
            if item[1] is not None:
                if torch.is_tensor(item[1]):
                    v = item[1].item()
                else:
                    v = item[1]
                total_losses[key] += v
                if log_total_losses is not None:
                    log_total_losses[key] += v
                current_losses[key] += v
                if v > 0.0:
                    total_steps[key] += 1
                    if log_total_steps is not None:
                        log_total_steps[key] += 1

    return current_losses, total_losses, total_steps, log_total_losses, log_total_steps


def calc_avg_loss(total_losses, nb_steps, total_steps=None):
    '''
    计算多种loss得平均loss与总loss
    :param total_losses:
    :param nb_steps:
    :param total_steps
    :return:
    '''
    loss_detail = {}
    loss = 0
    for item in total_losses.items():
        key = item[0]
        steps = total_steps[key] if total_steps is not None and key in total_steps and total_steps[key] > 0 else nb_steps
        if steps > 0:
            v = item[1] / steps
        else:
            v = 0.0
        if key not in loss_detail:
            loss_detail[key] = float(v)
        else:
            loss_detail[key] += float(v)
        loss += float(v)
    all_result = {
        "loss_detail": loss_detail,
        "loss": loss
    }
    return all_result, loss, loss_detail


aa_d3to1 = {
    'CYS': 'C',
    'ASP': 'D',
    'SER': 'S',
    'GLN': 'Q',
    'LYS': 'K',
    'ILE': 'I',
    'PRO': 'P',
    'THR': 'T',
    'PHE': 'F',
    'ASN': 'N',
    'GLY': 'G',
    'HIS': 'H',
    'LEU': 'L',
    'ARG': 'R',
    'TRP': 'W',
    'ALA': 'A',
    'VAL': 'V',
    'GLU': 'E',
    'TYR': 'Y',
    'MET': 'M',
    'SEC': 'U',
    'PYL': 'O'
}


def seq_type_is_match_seq(seq_type, seq):
    """
    判断序列内容与序列类型是否匹配
    :param seq_type:
    :param seq:
    :return:
    """
    if seq_type is None or seq is None:
        return False
    seq = seq.strip().upper()
    atcgu_num = 0
    total_num = 0
    for ch in seq:
        if ch < 'A' or ch > 'Z':
            continue
        total_num += 1
        if ch in {"A", "T", "C", "G", "U", "N"}:
            atcgu_num += 1

    if len(seq) > 0 and seq[0] == "M" and seq_type == "prot":
        return True
    is_gene = False
    if total_num == atcgu_num or atcgu_num >= 0.8 * total_num:
        is_gene = True

    if is_gene and seq_type in ["gene", "dna", "rna", "nucl"]:
        return True
    if not is_gene and seq_type == "prot":
        return True
    return False


def gene_seq_replace_eval_mask(seq):
    '''
    Nucleic acid （gene replace: A->1, U/T->2, C->3, G->4, N->5
    :param seq:
    :return:
    '''
    new_seq = ""
    for ch in seq:
        if ch in ["A", "a"]:
            new_seq += "1"
        elif ch in ["T", "U", "t", "u"]:
            new_seq += "2"
        elif ch in ["C", "c"]:
            new_seq += "3"
        elif ch in ["G", "g"]:
            new_seq += "4"
        elif ch in ['-', '_']:
            new_seq += '-'
        else:
            # unknown
            new_seq += "5"
    return new_seq


def span_merge(spans, start_index=0, end_index=1, value_index=2, merge_type="intersection"):
    '''
    区间合并，删除子区间，合并连在一起的
    :param spans:
    :param start_index:
    :param end_index:
    :param value_index:
    :param merge_type: 合并类型，intersection：只要有交集就合并， sub: 要是子集才合并； join: 包括首尾相接的， sub-join: 子集或者首尾相接的情况
    :return:
    '''
    sorted_spans = sorted(spans, key=lambda x:(x[start_index], -x[end_index]))
    result = []
    for span in sorted_spans:
        if result:
            if merge_type == "intersection" and result[-1][end_index] > span[start_index] and result[-1][value_index] == span[value_index]:
                # result中最后一个区间的右值>新区间的左值，说明两个区间有重叠，这种有交集，但是交集不是首尾相接
                # 将result中最后一个区间更新为合并之后的新区间
                result[-1][end_index] = max(result[-1][end_index], span[end_index])
            elif merge_type == "sub" and result[-1][end_index] >= span[end_index] and result[-1][value_index] == span[value_index]:
                # 要是子集包含
                result[-1][end_index] = max(result[-1][end_index], span[end_index])
            elif merge_type == "join" and result[-1][end_index] >= span[start_index] and result[-1][value_index] == span[value_index]:
                # 有交集或者首尾相接的情况
                result[-1][end_index] = max(result[-1][end_index], span[end_index])
            elif merge_type == "sub-join" and (result[-1][end_index] == span[start_index] or result[-1][end_index] >= span[end_index]) and result[-1][value_index] == span[value_index]:
                # 子集或者首尾相接的情况
                result[-1][end_index] = max(result[-1][end_index], span[end_index])
            else:
                result.append(span)
        else:
            result.append(span)

    return result


def calc_eval_test_loss(losses, total_losses, total_steps, total_loss):
    cur_loss = 0.0
    current_losses = {}
    for cur_losses in losses:
        for item1 in cur_losses.items():
            key1 = item1[0]
            if key1 not in total_losses:
                total_losses[key1] = {}
            if key1 not in current_losses:
                current_losses[key1] = {}
            if key1 not in total_steps:
                total_steps[key1] = {}
            for item2 in item1[1].items():
                key2 = item2[0]
                if item2[1] is not None:
                    if torch.is_tensor(item2[1]):
                        v = item2[1].item()
                    else:
                        v = item2[1]
                    if key2 not in total_losses[key1]:
                        total_losses[key1][key2] = v
                    else:
                        total_losses[key1][key2] += v
                    if key2 not in current_losses[key1]:
                        current_losses[key1][key2] = v
                    else:
                        current_losses[key1][key2] += v
                    if v > 0.0:
                        if key2 not in total_steps[key1]:
                            total_steps[key1][key2] = 1
                        else:
                            total_steps[key1][key2] += 1
                    total_loss += v
                    cur_loss += v
    return current_losses, total_losses, total_steps, total_loss, cur_loss


def print_shape(item):
    '''
    print shape
    :param item:
    :return:
    '''
    if isinstance(item, dict):
        for item1 in item.items():
            print(item1[0] + ":")
            print_shape(item1[1])
    elif isinstance(item, list):
        for idx, item1 in enumerate(item):
            print("idx: %d" % idx)
            print_shape(item1)
    else:
        if item is None:
            print("None")
        else:
            print("shape:", item.shape)
        print("*" * 10)


def print_batch(
        value,
        key=None,
        debug_path=None,
        wfp=None,
        local_rank=-1
):
    '''
    print a batch
    :param value:
    :param key:
    :param debug_path:
    :param wfp:
    :param local_rank:
    :return:
    '''
    if key is None:
        key = ""
    if isinstance(value, list):
        wfp.write("list size: %d\n" % len(value))
        for idx, v in enumerate(value):
            if wfp is not None:
                if v is not None:
                    wfp.write(str([
                        torch.min(v),
                        torch.min(torch.where(v == -100, 10000, v)),
                        torch.max(v)]) + "\n")
                    wfp.write(str(v.shape) + "\n")
                else:
                    wfp.write("None\n")
                wfp.write("-" * 10 + "\n")
            else:
                if v is not None:
                    print([torch.min(v), torch.min(torch.where(v == -100, 10000, v)), torch.max(v)])
                    print(v.shape)
                else:
                    print("None")
                print("-" * 50)
            if v is not None:
                try:
                    dtype = v.dtype
                    if dtype in [torch.float32, torch.float16, torch.float64, torch.float]:
                        fmt = '%0.4f'
                    else:
                        fmt = '%i'
                    value = v.detach().cpu().numpy()
                    if debug_path is not None:
                        if value.ndim == 3:
                            for dim_1_idx in range(value.shape[0]):
                                np.savetxt(
                                    os.path.join(debug_path, "%s_batch_idx_%d.txt" % (key, dim_1_idx)),
                                    value[dim_1_idx, :, :],
                                    fmt=fmt,
                                    delimiter=","
                                )
                        else:
                            np.savetxt(
                                os.path.join(debug_path, "%s_idx_%d.txt" % (key, idx)),
                                value,
                                fmt=fmt,
                                delimiter=","
                            )
                    else:
                        if value.ndim == 3:
                            for dim_1_idx in range(value.shape[0]):
                                np.savetxt(
                                    os.path.join(debug_path, "%s_batch_idx_%d.txt" % (key, dim_1_idx)),
                                    value[dim_1_idx, :, :],
                                    fmt=fmt,
                                    delimiter=","
                                )
                        else:
                            np.savetxt(
                                "%s_idx_%d.txt" % (key, idx),
                                value,
                                fmt=fmt,
                                delimiter=","
                            )
                except Exception as e:
                    print(e)
    elif isinstance(value, dict):
        for item in value.items():
            if wfp is not None:
                wfp.write(str(item[0]) + ":\n")
            else:
                print(str(item[0]) + ':')
            print_batch(item[1], item[0], debug_path, wfp, local_rank)
    else:
        if wfp is not None:
            if value is not None:
                wfp.write(
                    str([torch.min(value), torch.min(torch.where(value == -100, 10000, value)), torch.max(value)]) + "\n")
                wfp.write(str(value.shape) + "\n")
            else:
                wfp.write("None\n")
            wfp.write("-" * 10 + "\n")
        else:
            if value is not None:
                print([torch.min(value), torch.min(torch.where(value == -100, 10000, value)), torch.max(value)])
                print(value)
                print(value.shape)
            else:
                print("None")
            print("-" * 10)
        if value is not None:
            dtype = value.dtype
            if dtype in [torch.float32, torch.float16, torch.float64, torch.float]:
                fmt = '%0.4f'
            else:
                fmt = '%i'
            try:
                value = value.detach().cpu().numpy()
                if debug_path is not None:
                    if value.ndim == 3:
                        for dim_1_idx in range(value.shape[0]):
                            np.savetxt(
                                os.path.join(debug_path, "%s_batch_idx_%d.txt" % (key, dim_1_idx)),
                                value[dim_1_idx, :, :],
                                fmt=fmt,
                                delimiter=","
                            )
                    else:
                        np.savetxt(os.path.join(debug_path, "%s.txt" % key), value, fmt=fmt, delimiter=",")
                else:
                    if value.ndim == 3:
                        for dim_1_idx in range(value.shape[0]):
                            np.savetxt(
                                "%s_batch_idx_%d.txt" % (key, dim_1_idx),
                                value[dim_1_idx, :, :],
                                fmt=fmt,
                                delimiter=","
                            )
                    else:
                        np.savetxt("%s.txt" % key, value, fmt=fmt, delimiter=",")
            except Exception as e:
                print(e)


def save_model_parameters(model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name, param in model.named_parameters():
        weights = param.data.numpy()
        if param.requires_grad:
            np.savetxt(
                os.path.join(save_path, "%s_grad.txt" % name),
                weights,
                fmt="%.6f",
                delimiter="\n"
            )
        else:
            np.savetxt(
                os.path.join(save_path, "%s_no_grad.txt" % name),
                weights,
                fmt="%.6f",
                delimiter="\n"
            )


def load_trained_model(model_config, args, model_class, model_dirpath):
    # load exists checkpoint
    print("Load pretrained model: %s" % model_dirpath)
    try:
        model = model_class.from_pretrained(model_dirpath, config=model_config, args=args)
    except Exception as e:
        print(e)
        print("Retry loading...")
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
            print("Load trained model diff:")
            print(diff)
        model.load_state_dict(new_state_dict)
    return model


def load_trained_logs(log_filepath):
    args_info = None
    obj_str = None
    obj_end = True
    with open(log_filepath, "r") as rfp:
        for line in rfp:

            line = line.strip()
            if "{" == line[0]:
                obj_str = line.strip().replace(" ", "")
                try:
                    args_info = json.loads(obj_str)
                    return args_info
                except Exception as e:
                    obj_end = False
            elif not obj_end:
                obj_str += line.strip().replace(" ", "")
                try:
                    args_info = json.loads(obj_str)
                    return args_info
                except Exception as e:
                    obj_end = False
    return args_info


def matrix_2_vector(matrix, matrix_has_special_token, vector_type, save_type):
    if vector_type == "cls":
        if save_type == "numpy":
            return matrix[0, :].copy()
        else:
            return matrix[0, :].clone()
    elif vector_type == "max":
        if matrix_has_special_token:
            if save_type == "numpy":
                return np.max(matrix[1:-1, :], axis=0)
            else:
                return torch.amax(matrix[1:-1, :], dim=0)
        else:
            if save_type == "numpy":
                return np.max(matrix, axis=0)
            else:
                return torch.amax(matrix, dim=0)
    else:
        if matrix_has_special_token:
            if save_type == "numpy":
                return np.mean(matrix[1:-1, :], axis=0)
            else:
                return torch.mean(matrix[1:-1, :], dim=0)
        else:
            if save_type == "numpy":
                return np.mean(matrix, axis=0)
            else:
                return torch.mean(matrix, dim=0)


def print_args(args):
    if isinstance(args, dict):
        print(args)
    else:
        args_dict = {}
        for attr, value in sorted(args.__dict__.items()):
            if attr != "device":
                args_dict[attr] = value
        print(args_dict)


def clean_seq_esm(seq_id, seq, return_rm_index=False):
    seq = seq.upper()
    new_seq = ""
    has_invalid_char = False
    invalid_char_set = set()
    return_rm_index_set = set()
    for idx, ch in enumerate(seq):
        if 'A' <= ch <= 'Z' and ch not in ['J']:
            new_seq += ch
        else:
            invalid_char_set.add(ch)
            return_rm_index_set.add(idx)
            has_invalid_char = True
    if has_invalid_char:
        print("id: %s. Seq: %s" % (seq_id, seq))
        print("invalid char set:", invalid_char_set)
        print("return_rm_index:", return_rm_index_set)
    if return_rm_index:
        return new_seq, return_rm_index_set
    return new_seq


def clean_seq_luca(seq_id, seq):
    seq = seq.upper()
    new_seq = ""
    for idx, ch in enumerate(seq):
        if 'A' <= ch <= 'Z':
            new_seq += ch
    return new_seq


def gcd(x, y):
    '''
    最大公约数
    :param x:
    :param y:
    :return:
    '''
    m = max(x, y)
    n = min(x, y)
    while m % n:
        m, n = n, m % n
    return n


def lcm(x, y):
    '''
    最小公倍数
    :param x:
    :param y:
    :return:
    '''
    m = max(x, y)
    n = min(x, y)
    while m % n:
        m, n = n, m % n
    return x*y//n


def dict_update(raw, new):
    dict_update_iter(raw, new)
    dict_add(raw, new)


def dict_update_iter(raw, new):
    for key in raw:
        if key not in new.keys():
            continue
        if isinstance(raw[key], dict) and isinstance(new[key], dict):
            dict_update(raw[key], new[key])
        else:
            raw[key] = new[key]


def dict_add(raw, new):
    update_dict = {}
    for key in new:
        if key not in raw.keys():
            update_dict[key] = new[key]

    raw.update(update_dict)


def write_processed_sample_ids(
        dataset_type,
        time_str,
        sample_ids,
        epoch,
        local_rank
):
    """
    将已处理的样本id写入，便于恢复现场
    :param dataset_type: 数据集类型，train，validation，
    :param sample_ids: 要写入的sample_ids
    :param time_str: modeling time str
    :param epoch: 当前第几个epoch
    :param local_rank: cuda的id
    :return:
    """
    if local_rank > -1:
        dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "processed_samples",
            time_str,
            dataset_type,
            "rank-%d" % local_rank,
            "epoch-%d" % epoch
        )
    else:
        dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "processed_samples",
            time_str,
            dataset_type,
            "epoch-%d" % epoch
        )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    size = len(sample_ids)
    if size > 0:
        if local_rank > -1:
            file_path = os.path.join(
                dir_path,
                "%sed_sample_ids_epoch_%d_rank_%d.txt" % (dataset_type, epoch, local_rank)
            )
        else:
            file_path = os.path.join(
                dir_path,
                "%sed_sample_ids_epoch_%d.txt" % (dataset_type, epoch)
            )
        with open(file_path, "a+") as afp:
            for sample_id in sample_ids:
                afp.write("%s\n" % str(sample_id))
            print("Wrote %d into %s." % (size, file_path))


def calc_emb_filename_by_sample_id(sample_id, embedding_type):
    """
    根据sample_id得到emb_filename
    :param sample_id:
    :param embedding_type:
    :return:
    """
    sample_id = str(sample_id)
    if sample_id[0] == ">":
        sample_id = sample_id[1:]
    if "|" in sample_id:
        strs = sample_id.split("|")
        if len(strs) > 1:
            emb_filename = embedding_type + "_" + strs[1].strip() + ".pt"
        else:
            emb_filename = embedding_type + "_" + sample_id.replace(" ", "").replace("/", "_") + ".pt"
    else:
        emb_filename = embedding_type + "_" + sample_id.replace(" ", "").replace("/", "_") + ".pt"
    return emb_filename


def calc_emb_filename_by_seq_id(sample_id, embedding_type):
    """
    根据sample_id得到emb_filename
    :param sample_id:
    :param embedding_type:
    :return:
    """
    sample_id = str(sample_id)
    if sample_id[0] == ">":
        sample_id = sample_id[1:]
    if "|" in sample_id:
        strs = sample_id.split("|")
        if len(strs) > 1:
            emb_filename = embedding_type + "_" + strs[1].strip() + ".pt"
        else:
            emb_filename = embedding_type + "_" + sample_id.replace(" ", "").replace("/", "_") + ".pt"
    else:
        emb_filename = embedding_type + "_" + sample_id.replace(" ", "").replace("/", "_") + ".pt"
    return emb_filename


def topk_values_indices(matrix, topk):
    assert matrix.ndim == 2
    # Sort each row and get the indices
    sorted_indices = np.argsort(-matrix, axis=1)

    # Get the indices of the top 2 values
    topk_indices = sorted_indices[:, :topk]

    # Extract the top k values using advanced indexing
    rows = np.arange(matrix.shape[0])[:, None]
    topk_values = matrix[rows, topk_indices]

    return topk_values, topk_indices


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        dir_name = os.path.dirname(local_filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    # filter out keep-alive new chunks
                    f.write(chunk)
    return local_filename


def download_folder(base_url, file_names, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    for file_name in file_names:
        print(f"Downloading {file_name}...")
        file_url = f"{base_url}/{file_name}"
        local_filename = os.path.join(local_dir, file_name)
        download_file(file_url, local_filename)
        print(f"Downloaded {file_name}")


def download_trained_checkpoint_lucaone_v1(
        llm_dir,
        llm_type="lucaone_gplm",
        llm_version="v2.0",
        llm_task_level="token_level,span_level,seq_level,structure_level",
        llm_time_str="20231125113045",
        llm_step="60000000",
        base_url="http://47.93.21.181/lucaone/TrainedCheckPoint"
):
    try:
        logs_file_names = ["logs.txt"]
        models_file_names = ["config.json", "pytorch.pth", "training_args.bin", "tokenizer/alphabet.pkl"]
        logs_path = "logs/lucagplm/%s/%s/%s/%s" % (
            llm_version,
            llm_task_level,
            llm_type,
            llm_time_str
        )
        models_path = "models/lucagplm/%s/%s/%s/%s/checkpoint-step%s" % (
            llm_version,
            llm_task_level,
            llm_type,
            llm_time_str,
            llm_step
        )
        logs_local_dir = os.path.join(llm_dir, logs_path)
        print("LucaOne Dir: %s" % os.path.abspath(llm_dir))
        print("Logs Local Dir: %s" % logs_local_dir)

        exists = True
        for logs_file_name in logs_file_names:
            filepath = os.path.join(logs_local_dir, logs_file_name)
            if not os.path.exists(filepath):
                exists = False
                print(os.path.abspath(os.path.join(logs_local_dir, logs_file_name)) + ' not exists.')
                break
            else:
                print("File: %s exists: %s." % (logs_file_name, filepath))
        models_local_dir = os.path.join(llm_dir, models_path)
        print("models_local_dir: %s" % models_local_dir)

        if exists:
            for models_file_name in models_file_names:
                filepath = os.path.join(models_local_dir, models_file_name)
                if not os.path.exists(filepath):
                    exists = False
                    print(os.path.abspath(os.path.join(models_local_dir, models_file_name)) + ' not exists.')
                    break
                else:
                    print("File: %s exists: %s." % (models_file_name, filepath))
        if not exists:
            print("*" * 20 + "Downloading" + "*" * 20)
            print("Downloading LucaOne TrainedCheckPoint: LucaOne-%s-%s-%s ..." % (llm_version, llm_time_str, llm_step))
            print("Wait a moment(total 8GB), please.")
            # download logs
            if not os.path.exists(logs_local_dir):
                os.makedirs(logs_local_dir)
            logs_base_url = os.path.join(base_url, logs_path)
            download_folder(logs_base_url, logs_file_names, logs_local_dir)
            # download models
            if not os.path.exists(models_local_dir):
                os.makedirs(models_local_dir)
            models_base_url = os.path.join(base_url, models_path)
            download_folder(models_base_url, models_file_names, models_local_dir)
            print("LucaOne Download Completed.")
            print("*" * 50)
    except Exception as e:
        print(e)
        print("Download automatically LucaOne Trained CheckPoint failed!")
        print("You can manually download 'logs/' and 'models/' into local directory: %s/ from %s" % (
            os.path.abspath(llm_dir),
            base_url
        ))
        raise Exception(e)


def download_trained_checkpoint_lucaone_v2(
        llm_dir,
        llm_type,
        llm_version,
        llm_step,
        base_url="http://47.93.21.181/lucaone/TrainedCheckPoint/latest/"
):
    if llm_type not in ["lucaone"]:
        llm_type = "lucaone"
    if llm_version not in ["lucaone", "lucaone-gene", "lucaone-prot"]:
        llm_version = "lucaone"
    if llm_step is None:
        if llm_version == "lucaone":
            llm_step = "60000000"
        elif llm_version == "lucaone-gene":
            llm_step = "36800000"
        elif llm_version == "lucaone-prot":
            llm_step = "30000000"
        else:
            llm_version = "lucaone"
            llm_step = "60000000"
    try:
        logs_file_names = ["logs.txt"]
        models_file_names = ["config.json", "pytorch.pth", "training_args.bin", "tokenizer/alphabet.pkl"]
        logs_path = "logs/%s/%s/" % (llm_type, llm_version)
        models_path = "models/%s/%s/checkpoint-step%s" % (llm_type, llm_version, llm_step)
        logs_local_dir = os.path.join(llm_dir, logs_path)
        print("LucaOne Dir: %s" % os.path.abspath(llm_dir))
        print("Logs Local Dir: %s" % logs_local_dir)

        exists = True
        for logs_file_name in logs_file_names:
            filepath = os.path.join(logs_local_dir, logs_file_name)
            if not os.path.exists(filepath):
                exists = False
                print(os.path.abspath(os.path.join(logs_local_dir, logs_file_name)) + ' not exists.')
                break
            else:
                print("File: %s exists: %s." % (logs_file_name, filepath))
        models_local_dir = os.path.join(llm_dir, models_path)
        print("Model Local Dir: %s" % models_local_dir)

        if exists:
            for models_file_name in models_file_names:
                filepath = os.path.join(models_local_dir, models_file_name)
                if not os.path.exists(filepath):
                    exists = False
                    print(os.path.abspath(os.path.join(models_local_dir, models_file_name)) + ' not exists.')
                    break
                else:
                    print("File: %s exists: %s." % (models_file_name, filepath))
        if not exists:
            print("*" * 20 + "Downloading" + "*" * 20)
            print("Downloading LucaOne TrainedCheckPoint: LucaOne-%s-%s-%s ..." % (llm_type, llm_version, llm_step))
            print("Wait a moment(total 8GB), please.")
            # download logs
            if not os.path.exists(logs_local_dir):
                os.makedirs(logs_local_dir)
            logs_base_url = os.path.join(base_url, logs_path)
            download_folder(logs_base_url, logs_file_names, logs_local_dir)
            # download models
            if not os.path.exists(models_local_dir):
                os.makedirs(models_local_dir)
            models_base_url = os.path.join(base_url, models_path)
            download_folder(models_base_url, models_file_names, models_local_dir)
            print("LucaOne Download Completed.")
            print("*" * 50)
    except Exception as e:
        print(e)
        print("Download automatically LucaOne Trained CheckPoint failed!")
        print("You can manually download 'logs/' and 'models/' into local directory: %s/ from %s" % (
            os.path.abspath(llm_dir),
            base_url
        ))
        raise Exception(e)
    return models_local_dir


def download_trained_checkpoint_lucacell(
        llm_dir,
        llm_type,
        llm_version,
        llm_task_name,
        llm_time_str,
        llm_step,
        base_url="http://47.93.21.181/lucaone/TrainedCheckPoint/tmp/"
):
    if llm_type not in ["lucacell"]:
        llm_type = "lucacell"
    if llm_version not in ["lucacell-2048pos/v1.0"]:
        llm_version = "lucacell-2048pos/v1.0"
    if llm_task_name is None:
        llm_task_name = "express_token_mask"
    if llm_time_str is None:
        llm_time_str = "20250726103509"
    if llm_step is None:
        llm_step = "400000"
    try:
        logs_file_names = ["logs.txt"]
        models_file_names = ["config.json", "pytorch.pth", "training_args.bin", "tokenizer/alphabet.pkl", "tokenizer/express_bin_list.meta"]
        logs_path = "logs/%s/%s/%s/%s" % (llm_type, llm_version, llm_task_name, llm_time_str)
        models_path = "models/%s/%s/%s/%s/checkpoint-step%s" % (llm_type, llm_version, llm_task_name, llm_time_str, llm_step)
        logs_local_dir = os.path.join(llm_dir, logs_path)
        print("LucaCell Dir: %s" % os.path.abspath(llm_dir))
        print("Logs Local Dir: %s" % logs_local_dir)

        exists = True
        for logs_file_name in logs_file_names:
            filepath = os.path.join(logs_local_dir, logs_file_name)
            if not os.path.exists(filepath):
                exists = False
                print(os.path.abspath(os.path.join(logs_local_dir, logs_file_name)) + ' not exists.')
                break
            else:
                print("File: %s exists: %s." % (logs_file_name, filepath))
        models_local_dir = os.path.join(llm_dir, models_path)
        print("Model Local Dir: %s" % models_local_dir)

        if exists:
            for models_file_name in models_file_names:
                filepath = os.path.join(models_local_dir, models_file_name)
                if not os.path.exists(filepath):
                    exists = False
                    print(os.path.abspath(os.path.join(models_local_dir, models_file_name)) + ' not exists.')
                    break
                else:
                    print("File: %s exists: %s." % (models_file_name, filepath))
        if not exists:
            print("*" * 20 + "Downloading" + "*" * 20)
            print("Downloading LucaCell TrainedCheckPoint: LucaCell-%s-%s-%s ..." % (llm_type, llm_version, llm_step))
            print("Wait a moment(total 7GB), please.")
            # download logs
            if not os.path.exists(logs_local_dir):
                os.makedirs(logs_local_dir)
            logs_base_url = os.path.join(base_url, logs_path)
            download_folder(logs_base_url, logs_file_names, logs_local_dir)
            # download models
            if not os.path.exists(models_local_dir):
                os.makedirs(models_local_dir)
            models_base_url = os.path.join(base_url, models_path)
            download_folder(models_base_url, models_file_names, models_local_dir)
            print("LucaCell Download Completed.")
            print("*" * 50)
    except Exception as e:
        print(e)
        print("Download automatically LucaCell Trained CheckPoint failed!")
        print("You can manually download 'logs/' and 'models/' into local directory: %s/ from %s" % (
            os.path.abspath(llm_dir),
            base_url
        ))
        raise Exception(e)
