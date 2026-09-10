#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/5 09:55
@project: LucaOne
@file: tester
@desc: tester for LucaOne
'''
import os
import sys, torch
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import to_device, concat_output, calc_avg_loss, calc_eval_test_loss, \
        process_outputs, eval_metrics, print_shape, metrics_merge, print_batch, writer_info_tb
    from multi_files_stream_dataloader import MultiFilesStreamLoader
    from common.metrics import metrics_multi_class, metrics_binary
except ImportError:
    from src.utils import to_device, concat_output, calc_avg_loss, calc_eval_test_loss,\
        process_outputs, eval_metrics, print_shape, metrics_merge, print_batch, writer_info_tb
    from src.multi_files_stream_dataloader import MultiFilesStreamLoader
    from src.common.metrics import metrics_multi_class, metrics_binary


def test(args, model, parse_row_func, batch_data_func, global_step, prefix="", tb_writer=None, log_fp=None):
    """
    evaluation on test set
    :param args:
    :param model:
    :param parse_row_func:
    :param batch_data_func:
    :param global_step
    :param prefix:
    :param tb_writer
    :param log_fp:
    :return:
    """
    if hasattr(model, "module"):
        model = model.module
    save_output_dir = os.path.join(args.output_dir, prefix)
    print("\nTesting information dir: ", save_output_dir)
    if not os.path.exists(save_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(save_output_dir)
    test_dataloader = MultiFilesStreamLoader(
        args.test_data_dir,
        args.per_gpu_eval_batch_size,
        args.buffer_size,
        parse_row_func=parse_row_func,
        batch_data_func=batch_data_func,
        dataset_type="test",
        header=True,
        shuffle=False
    )
    # Testing
    if log_fp:
        log_fp.write("***** Running testing {} *****\n".format(prefix))
        log_fp.write("Test Dataset Instantaneous batch size per GPU = %d\n" % args.per_gpu_eval_batch_size)
        log_fp.write("#" * 50 + "\n")
        log_fp.flush()

    dataset_name = "test"

    nb_steps = 0
    # loss
    total_losses = {}

    # truth
    truths = {}
    # predicted prob
    preds = {}
    # truth
    truths_b = {}
    # predicted prob
    preds_b = {}
    # truth
    pair_truths = {}
    # predicted prob
    pair_preds = {}

    total_loss = 0

    model.eval()

    done_sample_num = 0

    for step, batch in enumerate(test_dataloader):
        if "sample_ids" in batch:
            del batch["sample_ids"]
        # testing
        with torch.no_grad():
            batch, cur_sample_num = to_device(
                args.device,
                batch,
                exclude=[
                    "nucleotide_input_ids",
                    "nucleotide_attention_mask",
                    "nucleotide_position_ids",
                    "nucleotide_input_embeds"
                ]
            )
            done_sample_num += cur_sample_num
            try:
                if args.use_bf16:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output = model(
                            **batch,
                            need_head_weights=False,
                            need_weights=False,
                            return_dict=True
                        )
                else:
                    output = model(
                        **batch,
                        need_head_weights=False,
                        need_weights=False,
                        return_dict=True
                    )
            except Exception as e:
                exception_path = "../exception/%s" % args.time_str
                if not os.path.exists(exception_path):
                    os.makedirs(exception_path)
                with open(os.path.join(exception_path, "test_exception_info_%d" % args.local_rank), "a+") as afp:
                    afp.write(str(e) + "\n")
                    afp.flush()
                with open(os.path.join(exception_path, "test_exception_input_%d" % args.local_rank), "a+") as afp:
                    afp.write(str(batch) + "\n")
                    afp.flush()
                debug_path = "../debug/%s/test/local_rank%s/%d/" % (args.time_str, "_" + str(args.local_rank) if args.local_rank >= 0 else "-1", step)
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)
                with open(os.path.join(debug_path, "test_exception_input_details.txt"), "a+") as afp:
                    print_batch(batch, key=None, debug_path=debug_path, wfp=afp, local_rank=args.local_rank)
                    afp.flush()
                continue
            if isinstance(output, dict):
                losses = output.losses
                logits = output.outputs
            else:
                losses = output[0]
                logits = output[1]
            loss = losses[0]
            outputs = {}
            # gene token mask loss
            if len(losses) > 1 and losses[1]:
                mask_gene_token_loss = losses[1].item()
                if "mask_gene_token_loss" not in total_losses:
                    total_losses["mask_gene_token_loss"] = 0
                total_losses["mask_gene_token_loss"] += mask_gene_token_loss
                outputs["gene_mask"] = logits[0]

            # express token mask loss
            if len(losses) > 2 and losses[2]:
                mask_express_token_loss = losses[2].item()
                if "mask_express_token_loss" not in total_losses:
                    total_losses["mask_express_token_loss"] = 0
                total_losses["mask_express_token_loss"] += mask_express_token_loss
                outputs["express_value_mask"] = logits[1]

            # express sorted mask loss
            if len(losses) > 3 and losses[3]:
                mask_express_sorted_loss = losses[3].item()
                if "mask_express_sorted_loss" not in total_losses:
                    total_losses["mask_express_sorted_loss"] = 0
                total_losses["mask_express_sorted_loss"] += mask_express_sorted_loss
                outputs["express_sorted_mask"] = logits[2]

            cur_loss = loss.item()
            total_loss += cur_loss
            print("\rTest, Batch: %06d, Sample Num: %d, Cur Loss: %0.6f, Avg Loss: %0.6f" %
                  (step + 1, done_sample_num, cur_loss, total_loss/(nb_steps + 1)),
                  end="", flush=True)
            nb_steps += 1
            if args.do_metrics:
                if "labels" in batch:
                    truths, preds = process_outputs(
                        args.output_mode,
                        batch["labels"],
                        outputs,
                        truths,
                        preds,
                        ignore_index=args.ignore_index,
                        keep_seq=False
                    )

    all_result, merged_loss, loss_detail = calc_avg_loss(
        total_losses,
        nb_steps,
        total_steps=None
    )
    if args.do_metrics:
        if truths is not None and len(truths) > 0:
            results = eval_metrics(args.output_mode, truths, preds, threshold=0.5)
            all_result = metrics_merge(results, all_result)
        if truths_b is not None and len(truths_b) > 0:
            results_b = eval_metrics(args.output_mode, truths_b, preds_b, threshold=0.5)
            all_result = metrics_merge(results_b, all_result)
        if pair_truths is not None and len(pair_truths) > 0:
            pair_results = eval_metrics(args.output_mode, pair_truths, pair_preds, threshold=0.5)
            all_result = metrics_merge(pair_results, all_result)

    writer_info_tb(
        tb_writer, {
            "done_sample_num": done_sample_num
        },
        global_step,
        prefix=dataset_name
    )
    writer_info_tb(
        tb_writer, {
            "merged_loss": merged_loss
        },
        global_step,
        prefix=dataset_name
    )
    writer_info_tb(
        tb_writer,
        loss_detail,
        global_step,
        prefix=dataset_name + "_avg_loss"
    )

    with open(os.path.join(save_output_dir, "eval_%s_checkpoints-step%d_metrics.txt" % (dataset_name, global_step)), "w") as writer:
        writer.write("***** Eval results %s on Checkpoints %d *****\n" % (dataset_name, global_step))
        writer.write("%s average merged_loss = %0.6f\n" % (dataset_name, merged_loss))
        writer.write("%s detail loss = %s\n" % (dataset_name, str(loss_detail)))
        for key in sorted(all_result.keys()):
            writer.write("%s = %s\n" % (key, str(all_result[key])))

    return all_result
