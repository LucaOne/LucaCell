#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/11/26 21:05
@project: LucaOnePlusTasks
@file: multi_label_metrics
@desc: metrics for multi-label classification
'''
import csv
import torch
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
import warnings


def multi_label_acc(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    acc_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            acc_list.append(1.0)
        else:
            # acc
            acc = 1.0 - (union_len - intersection_len) / targets.shape[1]
            acc_list.append(acc)
    return round(sum(acc_list)/len(acc_list), 6) if len(acc_list) > 0 else 0


def multi_label_precision(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    prec_list = []

    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        target_len = len(target_relevant)
        predict_len = len(pred_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            prec_list.append(1.0)
        else:
            # precision
            prec = 0.0
            if predict_len > 0:
                prec = intersection_len / predict_len
            prec_list.append(prec)

    round(sum(prec_list)/len(prec_list), 6) if len(prec_list) > 0 else 0


def multi_label_recall(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    recall_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        target_len = len(target_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            recall_list.append(1.0)
        else:
            # recall
            if target_len > 0:
                recall = intersection_len / target_len
            else:
                recall = 1.0
            recall_list.append(recall)
    return round(sum(recall_list)/len(recall_list), 6) if len(recall_list) > 0 else 0


def multi_label_jaccard(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    jaccard_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            jaccard_list.append(1.0)
        else:
            # jaccard sim
            jac = intersection_len / union_len
            jaccard_list.append(jac)
    return round(sum(jaccard_list)/len(jaccard_list), 6) if len(jaccard_list) > 0 else 0


def multi_label_f1(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    f1_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        target_len = len(target_relevant)
        predict_len = len(pred_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            f1_list.append(1.0)
        else:
            # precision
            prec = 0.0
            if predict_len > 0:
                prec = intersection_len / predict_len
            # recall
            if target_len > 0:
                recall = intersection_len / target_len
            else:
                recall = 1.0
            # f1
            if prec + recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * prec * recall / (prec + recall)
            f1_list.append(f1)
    return round(sum(f1_list)/len(f1_list), 6) if len(f1_list) > 0 else 0


def multi_label_roc_auc(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    roc_auc_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        if union_len == 0:
            roc_auc_list.append(1.0)
        else:
            # roc_auc
            if len(np.unique(targets[idx, :])) > 1:
                roc_auc = roc_auc_macro(targets[idx, :], probs[idx, :])
                roc_auc_list.append(roc_auc)
    return round(sum(roc_auc_list)/len(roc_auc_list), 6) if len(roc_auc_list) > 0 else 0


def multi_label_pr_auc(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    pr_auc_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        if union_len == 0:
            pr_auc_list.append(1.0)
        else:
            # roc_auc
            if len(np.unique(targets[idx, :])) > 1:

                pr_auc = pr_auc_macro(targets[idx, :], probs[idx, :])
                pr_auc_list.append(pr_auc)

    return round(sum(pr_auc_list)/len(pr_auc_list), 6) if len(pr_auc_list) > 0 else 0


def metrics_multi_label_v1(targets,  probs, threshold=0.5):
    '''
    metrics of multi-label classification
    cal metrics for true matrix to predict probability matrix
    :param targets: true 0-1 indicator matrix (n_samples, n_labels)
    :param probs: probs 0~1 probability matrix (n_samples, n_labels)
    :param threshold: negative-positive threshold
    :return: some metrics
    '''
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    acc_list = []
    prec_list = []
    recall_list = []
    jaccard_list = []
    f1_list = []
    roc_auc_list = []
    pr_auc_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        target_len = len(target_relevant)
        predict_len = len(pred_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            acc_list.append(1.0)
            prec_list.append(1.0)
            recall_list.append(1.0)
            roc_auc_list.append(1.0)
            jaccard_list.append(1.0)
            f1_list.append(1.0)
            pr_auc_list.append(1.0)
        else:
            # acc
            acc = 1.0 - (union_len - intersection_len) / targets.shape[1]
            acc_list.append(acc)

            # precision
            prec = 0.0
            if predict_len > 0:
                prec = intersection_len / predict_len
            prec_list.append(prec)

            # recall
            if target_len > 0:
                recall = intersection_len / target_len
            else:
                recall = 1.0
            recall_list.append(recall)

            # jaccard sim
            jac = intersection_len / union_len
            jaccard_list.append(jac)

            # f1
            if prec + recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * prec * recall / (prec + recall)
            f1_list.append(f1)

            # roc_auc
            if len(np.unique(targets[idx, :])) > 1:
                roc_auc = roc_auc_macro(targets[idx, :], probs[idx, :])
                roc_auc_list.append(roc_auc)
                pr_auc = pr_auc_macro(targets[idx, :], probs[idx, :])
                pr_auc_list.append(pr_auc)

    f_max_value, p_max_value, r_max_value, t_max_value, preds_max_value = f_max(targets, probs)
    return {
        "acc_v1": round(float(sum(acc_list)/len(acc_list)), 6) if len(acc_list) > 0 else 0,
        "jaccard_v1": round(float(sum(jaccard_list)/len(jaccard_list)), 6) if len(jaccard_list) > 0 else 0,
        "prec_v1": round(float(sum(prec_list)/len(prec_list)), 6) if len(prec_list) > 0 else 0,
        "recall_v1": round(float(sum(recall_list)/len(recall_list)), 6) if len(recall_list) > 0 else 0,
        "f1_v1": round(float(sum(f1_list)/len(f1_list)), 6) if len(f1_list) > 0 else 0,
        "pr_auc_v1": round(float(sum(pr_auc_list)/len(pr_auc_list)), 6) if len(pr_auc_list) > 0 else 0,
        "roc_auc_v1": round(float(sum(roc_auc_list)/len(roc_auc_list)), 6) if len(roc_auc_list) > 0 else 0,
        "fmax_v1": round(float(f_max_value), 6),
        "pmax_v1": round(float(p_max_value), 6) ,
        "rmax_v1": round(float(r_max_value), 6),
        "tmax_v1": round(float(t_max_value), 6)
    }


def f_max(targets, probs, gos=None):
    '''
    f-max for multi-label classification
    :param targets: true 0-1 indicator matrix (n_samples, n_labels)
    :param probs: probs 0~1 probability matrix (n_samples, n_labels)
    :param gos:
    :return: fmax, p_max(precision max）, r_max（recall max）, t_max（classificaton threshold）, preds_max（0-1 indicator matrix)
    '''
    preds_max = None
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    # from 0.01 to 1 (100 thresholds)
    for tt in range(1, 101):
        threshold = tt / 100.0
        preds = (probs > threshold).astype(np.int32)
        p = 0.0
        r = 0.0
        total = 0
        p_total = 0
        for i in range(preds.shape[0]):
            tp = np.sum(preds[i, :] * targets[i, :])
            fp = np.sum(preds[i, :]) - tp
            fn = np.sum(targets[i, :]) - tp
            if gos:
                fn += gos[i]

            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall

        if total > 0 and p_total > 0:
            r /= total
            p /= p_total
            if p + r > 0:
                f = 2 * p * r / (p + r)
                if f_max < f:
                    f_max = f
                    p_max = p
                    r_max = r
                    t_max = threshold
                    preds_max = preds

    return f_max, p_max, r_max, t_max, preds_max


def metrics_multi_label_for_pred(targets,  preds, savepath=None):
    '''
    metrics for multi-label classification
    cal metrics for true matrix to predict
    :param targets: true 0-1 indicator matrix (n_samples, n_labels)
    :param preds: preds 0~1 indicator matrix  (n_samples, n_labels)
    :return: some metrics
    '''
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes(preds)
    acc_list = []
    prec_list = []
    recall_list = []
    jaccard_list = []
    f1_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]

        target_len = len(target_relevant)
        predict_len = len(pred_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        acc = 1.0 - (union_len - intersection_len) / targets.shape[1]
        prec = 0.0
        if predict_len > 0:
            prec = intersection_len / predict_len
        recall = 0
        if target_len > 0:
            recall = intersection_len / target_len
        else:
            print(targets[idx])
        jac = intersection_len / union_len
        if prec + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * prec * recall / (prec + recall)

        acc_list.append(acc)
        prec_list.append(prec)
        recall_list.append(recall)
        jaccard_list.append(jac)
        f1_list.append(f1)

    return {
        "acc": round(sum(acc_list)/targets.shape[0], 6),
        "jaccard": round(sum(jaccard_list)/targets.shape[0], 6),
        "prec": round(sum(prec_list)/targets.shape[0], 6),
        "recall": round(sum(recall_list)/targets.shape[0], 6),
        "f1": round(sum(f1_list)/targets.shape[0], 6)
    }


def label_id_2_array(label_ids, label_size):
    '''
    building 0-1 indicator array for multi-label classification
    :param label_ids:
    :param label_size:
    :return:
    '''
    arr = np.zeros(label_size)
    arr[label_ids] = 1
    return arr


def relevant_indexes(matrix):
    '''
    Which positions in the multi-label are labeled as 1
    :param matrix:
    :return:
    '''
    if torch.is_tensor(matrix):
        matrix = matrix.detach().cpu().numpy()
    relevants = []
    shape = matrix.shape
    if matrix.ndim == 3:

        for x in range(shape[0]):
            relevant_x = []
            for y in range(shape[1]):
                relevant_y = []
                for z in range(shape[2]):
                    if matrix[x, y, z] == 1:
                        relevant_y.append(int(z))
                relevant_x.append(relevant_y)
            relevants.append(relevant_x)
    elif matrix.ndim == 2:
        for row in range(shape[0]):
            relevant = []
            for col in range(shape[1]):
                if matrix[row, col] == 1:
                    relevant.append(int(col))
            relevants.append(relevant)
    else:
        for idx in range(matrix.shape[0]):
            if matrix[idx] == 1:
                relevants.append(int(idx))
    return relevants


def irrelevant_indexes(matrix):
    '''
    Which positions in the multi-label label are 0
    :param matrix:
    :return:
    '''
    if torch.is_tensor(matrix):
        matrix = matrix.detach().cpu().numpy()

    irrelevants = []
    if matrix.ndim == 3:
        for x in range(matrix.shape[0]):
            irrelevant_x = []
            for y in range(matrix.shape[1]):
                irrelevant_y = []
                for z in range(matrix.shape[2]):
                    if matrix[x, y, z] == 0:
                        irrelevant_y.append(int(z))
                irrelevant_x.append(irrelevant_y)
            irrelevants.append(irrelevant_x)
    elif matrix.ndim == 2:
        for row in range(matrix.shape[0]):
            irrelevant = []
            for col in range(matrix.shape[1]):
                if matrix[row, col] == 1:
                    irrelevant.append(int(col))
            irrelevants.append(irrelevant)
    else:
        for idx in range(matrix.shape[0]):
            if matrix[idx] == 1:
                irrelevants.append(int(idx))

    return irrelevants


def prob_2_pred(prob, threshold):
    '''
    Probabilities converted to 0-1 predicted labels
    :param prob:
    :param threshold:
    :return:
    '''
    if torch.is_tensor(prob):
        prob = prob.detach().cpu().numpy()

    if isinstance(prob, (np.ndarray, np.generic)):
        return (prob >= threshold).astype(int)


def roc_auc_macro(target, prob):
    '''
    macro roc auc
    :param target:
    :param prob:
    :return:
    '''
    return roc_auc_score(target, prob, average="macro")


def pr_auc_macro(target, prob):
    '''
    macro pr-auc
    :param target:
    :param prob:
    :return:
    '''
    return average_precision_score(target, prob, average="macro")


def write_error_samples_multi_label(
        filepath,
        samples,
        input_indexs,
        input_id_2_names,
        output_id_2_name,
        targets,
        probs,
        threshold=0.5,
        use_other_diags=False,
        use_other_operas=False,
        use_checkin_department=False
):
    '''
    writer bad cases for multi-label classification
    :param filepath:
    :param samples:
    :param input_indexs:
    :param input_id_2_names:
    :param output_id_2_name:
    :param targets:
    :param probs:
    :param threshold:
    :param use_other_diags:
    :param use_other_operas:
    :param use_checkin_department:
    :return:
    '''
    preds = prob_2_pred(probs, threshold=threshold)
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes(preds)
    with open(filepath, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["score", "y_true", "y_pred", "inputs"])
        for i in range(len(targets_relevant)):
            target = set(targets_relevant[i])
            pred = set(preds_relevant[i])
            jacc = len(target.intersection(pred))/(len(target.union(pred)))
            if output_id_2_name:
                target_labels = [output_id_2_name[v] for v in target]
                pred_labels = [output_id_2_name[v] for v in pred]
            else:
                target_labels = target
                pred_labels = pred
            sample = samples[i]
            if input_id_2_names:
                new_sample = []
                for idx, input_index in enumerate(input_indexs):
                    if input_index == 3 and not use_checkin_department:
                        input_index = 12
                    new_sample.append([input_id_2_names[idx][v] for v in sample[input_index]])
                    if input_index == 6 and use_other_diags or input_index == 8 and use_other_operas or input_index == 10 and use_other_diags:
                        new_sample.append([input_id_2_names[idx][v] for v in sample[input_index + 1]])
            else:
                new_sample = sample
            row = [jacc, target_labels, pred_labels, new_sample]
            writer.writerow(row)


def metrics_multi_label(targets, probs, threshold=0.5):
    """
    Calculates a comprehensive set of multi-label classification metrics.

    Args:
        targets (np.array): A 2D numpy array of shape (n_samples, n_classes) with true binary labels (0 or 1).
        probs (np.array): A 2D numpy array of shape (n_samples, n_classes) with predicted probabilities.
        threshold (float): The threshold to convert probabilities to binary predictions.

    Returns:
        dict: A dictionary containing all the calculated metrics.
    """
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    elif isinstance(probs, list):
        probs = np.array(probs)

    # --- Part 1: Threshold-Dependent Metrics ---
    # Convert probabilities to binary predictions
    preds = (probs >= threshold).astype(int)
    metrics = {
        "acc": accuracy_score(targets, preds)
    }

    # Calculate Micro-averaged metrics
    metrics["prec_micro"] = precision_score(targets, preds, average="micro", zero_division=0)
    metrics["recall_micro"] = recall_score(targets, preds, average="micro", zero_division=0)
    metrics["f1_micro"] = f1_score(targets, preds, average="micro", zero_division=0)
    metrics["jaccard_micro"] = jaccard_score(targets, preds, average="micro", zero_division=0)

    # Calculate Macro-averaged metrics
    metrics["prec"] = precision_score(targets, preds, average="macro", zero_division=0)
    metrics["recall"] = recall_score(targets, preds, average="macro", zero_division=0)
    metrics["f1"] = f1_score(targets, preds, average="macro", zero_division=0)
    metrics["jaccard"] = jaccard_score(targets, preds, average="macro", zero_division=0)

    # --- Part 2: Threshold-Independent Metrics ---
    # ROC AUC and PR AUC
    # Use a try-except block for roc_auc_score as it can fail if a class has only one label
    try:
        metrics["roc_auc"] = roc_auc_score(targets, probs, average="macro")
        metrics["roc_auc_micro"] = roc_auc_score(targets, probs, average="micro")
    except ValueError as e:
        warnings.warn(f"ROC AUC calculation failed: {e}. Returning NaN.")
        metrics["roc_auc"] = np.nan
        metrics["roc_auc_micro"] = np.nan

    metrics["pr_auc"] = average_precision_score(targets, probs, average="macro")
    metrics["pr_auc_micro"] = average_precision_score(targets, probs, average="micro")

    # --- Part 3: Optimal Threshold Metrics (F-max, etc.) ---

    # --- Micro-averaged F-max ---
    # Flatten all predictions and targets to compute a single global PR curve
    precision_micro, recall_micro, thresholds_micro = precision_recall_curve(targets.ravel(), probs.ravel())
    # Calculate F1 scores for micro-curve, handling division by zero
    f1_scores_micro = np.divide(
        2 * precision_micro * recall_micro,
        precision_micro + recall_micro,
        out=np.zeros_like(precision_micro),
        where=(precision_micro + recall_micro) != 0
    )
    fmax_micro_idx = np.argmax(f1_scores_micro)
    metrics["fmax_micro"] = f1_scores_micro[fmax_micro_idx]
    metrics["pmax_micro"] = precision_micro[fmax_micro_idx]
    metrics["rmax_micro"] = recall_micro[fmax_micro_idx]
    # Note: thresholds_micro is one element shorter than precision/recall
    metrics["tmax_micro"] = thresholds_micro[fmax_micro_idx] if fmax_micro_idx < len(thresholds_micro) else 1.0

    # --- Macro-averaged F-max ---
    n_classes = targets.shape[1]
    fmax_per_class = []
    pmax_per_class = []
    rmax_per_class = []
    tmax_per_class = []

    for i in range(n_classes):
        # Check if a class has positive samples
        if np.sum(targets[:, i]) == 0:
            continue  # Skip classes with no true positive labels

        precision, recall, thresholds = precision_recall_curve(targets[:, i], probs[:, i])
        f1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0
        )
        fmax_idx = np.argmax(f1_scores)
        fmax_per_class.append(f1_scores[fmax_idx])
        pmax_per_class.append(precision[fmax_idx])
        rmax_per_class.append(recall[fmax_idx])
        tmax_per_class.append(thresholds[fmax_idx] if fmax_idx < len(thresholds) else 1.0)

    # Average the results across classes that had positive samples
    metrics["fmax"] = np.mean(fmax_per_class) if fmax_per_class else np.nan
    metrics["pmax"] = np.mean(pmax_per_class) if pmax_per_class else np.nan
    metrics["rmax"] = np.mean(rmax_per_class) if rmax_per_class else np.nan
    metrics["tmax"] = np.mean(tmax_per_class) if tmax_per_class else np.nan
    metrics = {item[0]: round(float(item[1]), 6) if item[1] != np.nan else np.nan for item in metrics.items()}

    metrics.update(**metrics_multi_label_v1(targets, probs, threshold=0.5))
    return metrics


# --- Example Usage ---
if __name__ == "__main__":
    # Sample data: 4 samples, 3 classes
    # This data includes an imbalanced class (class 0)
    true_labels = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])

    pred_probs = np.array([
        [0.1, 0.2, 0.8],
        [0.1, 0.7, 0.6],
        [0.8, 0.6, 0.3],
        [0.2, 0.3, 0.9],
        [0.1, 0.9, 0.2]
    ])

    # Calculate all metrics using a standard 0.5 threshold
    all_metrics = metrics_multi_label(true_labels, pred_probs, threshold=0.5)

    import json
    print("--- Comprehensive Multi-label Metrics ---")
    print(json.dumps(all_metrics, indent=4))

    print("\n--- Key Takeaways ---")
    print(f"Macro F1 at threshold 0.5: {all_metrics['f1']:.4f}")
    print(f"The best possible Macro F1 (F-max) is: {all_metrics['fmax']:.4f}")
    print(f"This is achieved at an average threshold of: {all_metrics['tmax']:.4f}")

