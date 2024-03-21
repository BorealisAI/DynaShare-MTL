# Copyright (c) 2022-present, Royal Bank of Canada.
# Copyright (c) 2022-present, Ximeng Sun
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################
# Code is based on the AdaShare (https://arxiv.org/pdf/1911.12423.pdf) implementation 
# from https://github.com/sunxm2357/AdaShare by Ximeng Sun
##################################################################################################



import sys

sys.path.insert(0, "..")

import os
import time

os.environ["NVIDIA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


from dataloaders.mimic_dataloader import *
from dataloaders.pcba_dataloader import *
from envs.blockdrop_env import BlockDropEnv
import torch
import progressbar
from utils.util import (
    print_separator,
    read_yaml,
    create_path,
    print_yaml,
    should,
    fix_random_seed,
)
import numpy as np
from tqdm import tqdm
from models.util import *


def eval(
    opt,
    environ,
    dataloader,
    tasks,
    policy=False,
    num_train_layers=None,
    hard_sampling=False,
    eval_iter=400,
):
    batch_size = []
    records = {}
    val_metrics = {}
    if "pheno" in tasks:
        records["pheno"] = {"AUC": [], "loss": []}
    if "los" in tasks:
        records["los"] = {"kappa": [], "AUC": [], "loss": []}
    if "decomp" in tasks:
        records["decomp"] = {"AUC": [], "loss": []}
    if "ihm" in tasks:
        records["ihm"] = {"AUC": [], "loss": []}
    if "PCBA" in tasks[0]:
        for t_id, task in enumerate(tasks):
            records[task] = {"AUC": [], "loss": []}

    print_separator(" MID WAY VALIDATION ")
    if opt["dataload"]["dataset"] == "pcba":
        """Note: Combine several batches is important to avoid problems to
        calculate the AUC. When we calculate the metrics using only one
        batch we frequently have errors because only one class is present.
        """
        n = 0
        GT, prediction_sig, prediction, w = [], [], [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                if eval_iter != -1:
                    if batch_idx > eval_iter:
                        break

                if ((batch_idx + 1) % len(dataloader)) == 0:
                    auc, loss = environ.pcba_metric(prediction_sig, GT, w)
                    for t_id, task in enumerate(tasks):
                        records[task]["AUC"].append(auc[t_id])
                        records[task]["loss"].append(loss[t_id])

                else:

                    environ.set_inputs(batch)
                    prediction_temp, gt_temp = environ.val2_pcba(
                        policy, num_train_layers, hard_sampling
                    )

                    n0 = batch[1].shape[0]

                    gt, pred_sig, pred, w0 = [], [], [], []
                    for t_id in range(len(tasks)):
                        w0.append(np.array(batch[2][:, t_id].long().detach().numpy()))
                        gt.append(np.array(gt_temp[t_id].cpu().detach().numpy()))
                        pred.append(
                            np.array(prediction_temp[t_id].cpu().detach().numpy())
                        )
                        pred_sig.append(
                            np.array(
                                torch.sigmoid(prediction_temp[t_id])
                                .cpu()
                                .detach()
                                .numpy()
                            )
                        )
                    if batch_idx == 0:

                        w = np.array(w0).reshape(len(tasks), n0)
                        GT = np.array(gt).reshape(len(tasks), n0)
                        prediction = np.array(pred).reshape(len(tasks), n0)
                        prediction_sig = np.array(pred_sig).reshape(len(tasks), n0)
                    else:

                        w = np.concatenate(
                            (w, np.array(w0).reshape(len(tasks), n0)), axis=1
                        )
                        GT = np.concatenate(
                            (GT, np.array(gt).reshape(len(tasks), n0)), axis=1
                        )
                        prediction = np.concatenate(
                            (prediction, np.array(pred).reshape(len(tasks), n0)), axis=1
                        )
                        prediction_sig = np.concatenate(
                            (
                                prediction_sig,
                                np.array(pred_sig).reshape(len(tasks), n0),
                            ),
                            axis=1,
                        )
    else:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                if eval_iter != -1:
                    if batch_idx > eval_iter:
                        break

                environ.set_inputs(batch)
                metrics = environ.val2(policy, num_train_layers, hard_sampling)

                if "pheno" in tasks:
                    records["pheno"]["AUC"].append(metrics["pheno"]["AUC"])
                    records["pheno"]["loss"].append(metrics["pheno"]["loss"])
                if "los" in tasks:
                    records["los"]["kappa"].append(metrics["los"]["kappa"])
                    records["los"]["AUC"].append(metrics["los"]["AUC"])
                    records["los"]["loss"].append(metrics["los"]["loss"])
                if "decomp" in tasks:
                    records["decomp"]["AUC"].append(metrics["decomp"]["AUC"])
                    records["decomp"]["loss"].append(metrics["decomp"]["loss"])
                if "ihm" in tasks:
                    records["ihm"]["AUC"].append(metrics["ihm"]["AUC"])
                    records["ihm"]["loss"].append(metrics["ihm"]["loss"])
                batch_size.append(len(batch[0]))

    if "pheno" in tasks:
        val_metrics["pheno"] = {}
        val_metrics["pheno"]["AUC"] = (
            np.array(records["pheno"]["AUC"]) * np.array(batch_size)
        ).sum() / sum(batch_size)
        val_metrics["pheno"]["loss"] = (
            np.array(records["pheno"]["loss"]) * np.array(batch_size)
        ).sum() / sum(batch_size)

    if "los" in tasks:
        val_metrics["los"] = {}
        val_metrics["los"]["kappa"] = (
            np.array(records["los"]["kappa"]) * np.array(batch_size)
        ).sum() / sum(batch_size)
        val_metrics["los"]["AUC"] = (
            np.array(records["los"]["AUC"]) * np.array(batch_size)
        ).sum() / sum(batch_size)
        val_metrics["los"]["loss"] = (
            np.array(records["los"]["loss"]) * np.array(batch_size)
        ).sum() / sum(batch_size)

    if "decomp" in tasks:
        val_metrics["decomp"] = {}
        val_metrics["decomp"]["AUC"] = (
            np.array(records["decomp"]["AUC"]) * np.array(batch_size)
        ).sum() / sum(batch_size)
        val_metrics["decomp"]["loss"] = (
            np.array(records["decomp"]["loss"]) * np.array(batch_size)
        ).sum() / sum(batch_size)

    if "ihm" in tasks:
        val_metrics["ihm"] = {}
        val_metrics["ihm"]["AUC"] = (
            np.array(records["ihm"]["AUC"]) * np.array(batch_size)
        ).sum() / sum(batch_size)
        val_metrics["ihm"]["loss"] = (
            np.array(records["ihm"]["loss"]) * np.array(batch_size)
        ).sum() / sum(batch_size)

    if "PCBA" in tasks[0]:
        avg = []
        sd = 0
        for t_id, task in enumerate(tasks):
            val_metrics[task] = {}
            val_metrics[task]["AUC"] = np.array(records[task]["AUC"])
            val_metrics[task]["loss"] = np.array(records[task]["loss"])
            avg.append(records[task]["AUC"])
        avg = np.average(avg)
        sd = np.std(avg)
        print("avrage AUC: %.2f, std AUC: %.2f" % (avg, sd))

    return val_metrics


def train():
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    # Read the yaml
    print_separator(" READ YAML ")
    opt, gpu_ids, _ = read_yaml()
    fix_random_seed(opt["seed"][0])
    create_path(opt)
    # Print yaml on the screen
    lines = print_yaml(opt)
    for line in lines:
        print(line)
    # Print to file
    with open(
        os.path.join(opt["paths"]["log_dir"], opt["exp_name"], "opt.txt"), "w+"
    ) as f:
        f.writelines(lines)

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # Load the dataloader
    print_separator(" CREATE DATALOADERS ")
    if opt["dataload"]["dataset"] == "mimic":
        # To warm up
        (
            train_loader,
            val_loader,
            test_loader,
            task_info,
            task_number,
        ) = data_preparation_mimic3(
            opt["dataload"]["dataroot"],
            opt["train"]["batch_size"],
            opt["train"]["seqlen"],
            opt["train"]["prop"],
        )
        X, pheno, los, decomp, ihm = next(iter(train_loader))
        # To update the network parameters
        (train_loader1, _, _, _, _) = data_preparation_mimic3(
            opt["dataload"]["dataroot"],
            opt["train"]["batch_size"],
            opt["train"]["seqlen"],
            0.8 * opt["train"]["prop"],
        )
        # To update the policy weights
        (train_loader2, _, _, _, _) = data_preparation_mimic3(
            opt["dataload"]["dataroot"],
            opt["train"]["batch_size"],
            opt["train"]["seqlen"],
            0.2 * opt["train"]["prop"],
        )

    elif opt["dataload"]["dataset"] == "pcba":
        (
            train_loader,
            val_loader,
            test_loader,
            num_features,
            task_number,
            opt["tasks"],
        ) = data_preparation_pcba(opt)
        X, y, w = next(iter(train_loader))
        # To update the network parameters
        opt["train"]["prop"] = 0.8 * opt["train"]["prop"]
        (train_loader1, _, _, _, _, _) = data_preparation_pcba(opt)
        # To update the policy weights
        opt["train"]["prop"] = 0.2 * opt["train"]["prop"]
        (train_loader2, _, _, _, _, _) = data_preparation_pcba(opt)
        opt["train"]["prop"] = 1
        opt["lambdas"] = np.repeat(1, task_number)
        opt["tasks_num_class"] = np.repeat(1, task_number)
    else:
        raise NotImplementedError(
            "Dataset %s is not implemented" % opt["dataload"]["dataset"]
        )

    print("size of training set: ", len(train_loader))
    print("size of training set 1: ", len(train_loader1))
    print("size of training set 2: ", len(train_loader2))
    print("size of test set: ", len(val_loader))

    opt["train"]["weight_iter_alternate"] = opt["train"].get(
        "weight_iter_alternate", len(train_loader1)
    )
    opt["train"]["alpha_iter_alternate"] = opt["train"].get(
        "alpha_iter_alternate", len(train_loader2)
    )

    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************

    # Create the model and the pretrain model
    print_separator(" CREATE THE ENVIRONMENT ")

    environ = BlockDropEnv(
        opt["paths"]["log_dir"],
        opt["paths"]["checkpoint_dir"],
        opt["exp_name"],
        opt["tasks_num_class"],
        opt["init_neg_logits"],
        gpu_ids,
        opt["train"]["init_temp"],
        opt["train"]["decay_temp"],
        is_train=True,
        opt=opt,
    )
    current_iter = 0
    current_iter_w, current_iter_a = 0, 0
    if opt["train"]["resume"]:
        current_iter = environ.load(opt["train"]["which_iter"])
        if isinstance(environ.networks["mtl-net"], nn.DataParallel):
            environ.networks["mtl-net"].module.reset_logits()
        else:
            environ.networks["mtl-net"].reset_logits()

    environ.define_optimizer(False)
    environ.define_scheduler(False)
    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    batch_enumerator = enumerate(train_loader)
    batch_enumerator1 = enumerate(train_loader1)
    batch_enumerator2 = enumerate(train_loader2)
    flag = "update_w"

    environ.fix_alpha()
    environ.free_w(opt["fix_BN"])
    best_value, best_iter = 0, 0

    if opt["dataload"]["dataset"] == "mimic":
        refer_metrics = {
            "pheno": {"AUC": 50.00},
            "los": {"AUC": 50.00},
            "decomp": {"AUC": 50.00},
            "ihm": {"AUC": 50.00},
        }
    if opt["dataload"]["dataset"] == "pcba":
        refer_metrics = {}
    else:
        raise NotImplementedError(
            "Dataset %s is not implemented" % opt["dataload"]["dataset"]
        )

    best_metrics = None
    p_epoch = 0
    flag_warmup = True
    if opt["backbone"] == "ResNet18":
        num_blocks = 8
    elif opt["backbone"] in ["ResNet34", "ResNet50"]:
        num_blocks = 18
    elif opt["backbone"] == "ResNet101":
        num_blocks = 33
    elif opt["backbone"] == "WRN":
        num_blocks = 15
    else:
        raise ValueError("Backbone %s is invalid" % opt["backbone"])

    print_separator(" START TRAINING ")
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    while current_iter < opt["train"]["total_iters"]:
        start_time = time.time()
        environ.train()
        current_iter += 1
        # Warm up
        if current_iter < opt["train"]["warm_up_iters"]:
            batch_idx, batch = next(batch_enumerator)
            environ.set_inputs(batch)  # Set the inputs to cuda
            environ.optimize(opt["lambdas"], is_policy=False, flag="update_w")
            if batch_idx == len(train_loader) - 1:
                batch_enumerator = enumerate(train_loader)

            if should(current_iter, opt["train"]["print_freq"]):
                environ.print_loss(current_iter, start_time)

            # Validation
            if should(current_iter, opt["train"]["val_freq"]):
                environ.eval()
                val_metrics = eval(
                    opt,
                    environ,
                    val_loader,
                    opt["tasks"],
                    policy=False,
                    num_train_layers=None,
                )
                environ.print_loss(current_iter, start_time, val_metrics)
                environ.save("latest", current_iter)
                environ.train()
        else:
            if flag_warmup:
                environ.define_optimizer(policy_learning=True)
                environ.define_scheduler(True)

                flag_warmup = False

            if current_iter == opt["train"]["warm_up_iters"]:
                environ.save("warmup", current_iter)
                environ.fix_alpha()

            # Update the network weights
            if flag == "update_w":
                current_iter_w += 1
                batch_idx_w, batch = next(batch_enumerator1)
                environ.set_inputs(batch)

                if opt["is_curriculum"]:
                    num_train_layers = p_epoch // opt["curriculum_speed"] + 1
                else:
                    num_train_layers = None

                environ.optimize(
                    opt["lambdas"],
                    is_policy=opt["policy"],
                    flag=flag,
                    num_train_layers=num_train_layers,
                    hard_sampling=opt["train"]["hard_sampling"],
                )

                if should(current_iter, opt["train"]["print_freq"]):
                    environ.print_loss(current_iter, start_time)

                if should(current_iter_w, opt["train"]["weight_iter_alternate"]):
                    flag = "update_alpha"
                    environ.fix_w()
                    environ.free_alpha()

                    environ.eval()
                    print("Evaluating...")
                    val_metrics = eval(
                        opt,
                        environ,
                        val_loader,
                        opt["tasks"],
                        policy=opt["policy"],
                        num_train_layers=num_train_layers,
                        hard_sampling=opt["train"]["hard_sampling"],
                    )
                    environ.print_loss(current_iter, start_time, val_metrics)
                    environ.save("latest", current_iter)

                    if current_iter - opt["train"]["warm_up_iters"] >= num_blocks * opt[
                        "curriculum_speed"
                    ] * (
                        opt["train"]["weight_iter_alternate"]
                        + opt["train"]["alpha_iter_alternate"]
                    ):
                        new_value = 0

                        for k in refer_metrics.keys():
                            if k in val_metrics.keys():
                                for kk in val_metrics[k].keys():
                                    if not kk in refer_metrics[k].keys():
                                        continue

                                    value = val_metrics[k][kk] / refer_metrics[k][kk]

                                    value = value / len(
                                        list(
                                            set(val_metrics[k].keys())
                                            & set(refer_metrics[k].keys())
                                        )
                                    )
                                    new_value += value

                        if new_value > best_value:

                            best_value = new_value
                            best_metrics = val_metrics
                            best_iter = current_iter
                            environ.save("best", current_iter)
                        print("new value: %.3f" % new_value)
                        print(
                            "best iter: %d, best_value: %.3f" % (best_iter, best_value),
                            best_metrics,
                        )
                    environ.train()

                if batch_idx_w == len(train_loader1) - 1:
                    batch_enumerator1 = enumerate(train_loader1)

            # Update the policy network
            elif flag == "update_alpha":
                current_iter_a += 1
                batch_idx_a, batch = next(batch_enumerator2)
                environ.set_inputs(batch)
                if opt["is_curriculum"]:
                    num_train_layers = p_epoch // opt["curriculum_speed"] + 1
                else:
                    num_train_layers = None

                environ.optimize(
                    opt["lambdas"],
                    is_policy=opt["policy"],
                    flag=flag,
                    num_train_layers=num_train_layers,
                    hard_sampling=opt["train"]["hard_sampling"],
                )

                if should(current_iter, opt["train"]["print_freq"]):
                    environ.print_loss(current_iter, start_time)

                if should(current_iter_a, opt["train"]["alpha_iter_alternate"]):
                    flag = "update_w"
                    environ.fix_alpha()
                    environ.free_w(opt["fix_BN"])
                    environ.decay_temperature()
                    dists = environ.get_policy_prob()
                    print(np.concatenate(dists, axis=-1))
                    p_epoch += 1

                if batch_idx_a == len(train_loader2) - 1:
                    batch_enumerator2 = enumerate(train_loader2)

            else:
                raise ValueError("flag %s is not recognized" % flag)
        bar.update(current_iter)


if __name__ == "__main__":
    train()
