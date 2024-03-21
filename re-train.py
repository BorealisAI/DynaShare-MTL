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

os.environ["NVIDIA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time
import numpy as np


from torch.utils.data import DataLoader
import progressbar
from dataloaders.mimic_dataloader import *
from dataloaders.pcba_dataloader import *
from dataloaders.census_dataloader import *
from envs.blockdrop_env import BlockDropEnv
import torch
from utils.util import (
    print_separator,
    read_yaml,
    create_path,
    print_yaml,
    should,
    fix_random_seed,
)
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def eval_fix_policy(opt, environ, dataloader, tasks, eval_iter=400):
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
    if "education" in tasks:
        records["education"] = {"AUC": [], "loss": []}
    if "income" in tasks:
        records["income"] = {"AUC": [], "loss": []}
    if "marital" in tasks:
        records["marital"] = {"AUC": [], "loss": []}
    if "PCBA" in tasks[0]:
        for t_id, task in enumerate(tasks):
            records[task] = {"AUC": [], "loss": []}

    if opt["dataload"]["dataset"] == "pcba":
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
                    prediction_temp, gt_temp = environ.val_fix_policy_pcba()

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
                metrics = environ.val_fix_policy()

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
                if "education" in tasks:
                    records["education"]["AUC"].append(metrics["education"]["AUC"])
                    records["education"]["loss"].append(metrics["education"]["loss"])
                if "income" in tasks:
                    records["income"]["AUC"].append(metrics["income"]["AUC"])
                    records["income"]["loss"].append(metrics["income"]["loss"])
                if "marital" in tasks:
                    records["marital"]["AUC"].append(metrics["marital"]["AUC"])
                    records["marital"]["loss"].append(metrics["marital"]["loss"])
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

    if "education" in tasks:
        val_metrics["education"] = {}
        val_metrics["education"]["AUC"] = (
            np.array(records["education"]["AUC"]) * np.array(batch_size)
        ).sum() / sum(batch_size)
        val_metrics["education"]["loss"] = (
            np.array(records["education"]["loss"]) * np.array(batch_size)
        ).sum() / sum(batch_size)

    if "income" in tasks:
        val_metrics["income"] = {}
        val_metrics["income"]["AUC"] = (
            np.array(records["income"]["AUC"]) * np.array(batch_size)
        ).sum() / sum(batch_size)
        val_metrics["income"]["loss"] = (
            np.array(records["income"]["loss"]) * np.array(batch_size)
        ).sum() / sum(batch_size)

    if "marital" in tasks:
        val_metrics["marital"] = {}
        val_metrics["marital"]["AUC"] = (
            np.array(records["marital"]["AUC"]) * np.array(batch_size)
        ).sum() / sum(batch_size)
        val_metrics["marital"]["loss"] = (
            np.array(records["marital"]["loss"]) * np.array(batch_size)
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


def _train(exp_id, opt, gpu_ids):

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # load the dataloader
    print_separator("CREATE DATALOADERS")
    opt_tmp = opt if opt["policy_model"] == "instance-specific" else None
    if opt["dataload"]["dataset"] == "census":
        # To warm up
        (
            train_loader,
            val_loader,
            _,
            num_features,
            task_number,
            _,
        ) = data_preparation_census(opt)
        X, y = next(iter(train_loader))
    elif opt["dataload"]["dataset"] == "mimic":
        # To warm up
        (
            train_loader,
            val_loader,
            _,
            task_info,
            task_number,
        ) = data_preparation_mimic3(
            opt["dataload"]["dataroot"],
            opt["train"]["batch_size"],
            opt["train"]["seqlen"],
            opt["train"]["prop"],
        )
        X, pheno, los, decomp, ihm = next(iter(train_loader))
    elif opt["dataload"]["dataset"] == "census":
        # To warm up
        (
            train_loader,
            val_loader,
            test_loader,
            num_features,
            task_number,
            _,
        ) = data_preparation_census(opt)
        X, y = next(iter(train_loader))
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
        opt["train"]["prop"] = 1
        opt["lambdas"] = np.repeat(1, task_number)
        opt["tasks_num_class"] = np.repeat(1, task_number)
    else:
        raise NotImplementedError(
            "Dataset %s is not implemented" % opt["dataload"]["dataset"]
        )

    print("size of training set: ", len(train_loader))
    print("size of validation set: ", len(val_loader))

    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************

    # Create the model and the pretrain model
    print_separator("CREATE THE ENVIRONMENT")
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
    policy_label = "Iter%s_rs%04d" % (opt["train"]["policy_iter"], opt["seed"][exp_id])
    if opt["train"]["retrain_resume"]:
        current_iter = environ.load(opt["train"]["which_iter"])
        environ.load_policy(policy_label)
    else:

        init_state = deepcopy(environ.get_current_state(0))
        if environ.check_exist_policy(policy_label):
            environ.load_policy(policy_label)
        else:
            environ.load(opt["train"]["policy_iter"])
            dists = environ.get_policy_prob()
            overall_dist = np.concatenate(dists, axis=-1)
            print(overall_dist)
            environ.sample_policy(opt["train"]["hard_sampling"])
            environ.save_policy(policy_label)

        if opt["retrain_from_pl"]:
            environ.load(opt["train"]["policy_iter"])
        else:
            environ.load_snapshot(init_state)

    policys = environ.get_current_policy()
    overall_policy = np.concatenate(policys, axis=-1)
    print(overall_policy)

    environ.define_optimizer(False)
    environ.define_scheduler(False)
    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    environ.fix_alpha()
    environ.free_w(fix_BN=opt["fix_BN"])
    batch_enumerator = enumerate(train_loader)

    if opt["dataload"]["dataset"] == "mimic":
        refer_metrics = {
            "pheno": {"AUC": 77.4},
            "los": {"AUC": 63.45},
            "decomp": {"AUC": 97.03},
            "ihm": {"AUC": 91.03},
        }
    elif opt["dataload"]["dataset"] == "census":
        refer_metrics = {
            "education": {"AUC": 10.00},
            "income": {"AUC": 10.00},
            "marital": {"AUC": 10.00},
        }

    elif opt["dataload"]["dataset"] == "pcba":
        refer_metrics = {}
    else:
        raise NotImplementedError(
            "Dataset %s is not implemented" % opt["dataload"]["dataset"]
        )

    best_value, best_iter = 0, 0
    best_metrics = None
    opt["train"]["retrain_total_iters"] = opt["train"].get(
        "retrain_total_iters", opt["train"]["total_iters"]
    )

    print_separator(" START RE-TRAINING ")
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    while current_iter < opt["train"]["retrain_total_iters"]:
        start_time = time.time()
        environ.train()
        current_iter += 1
        batch_idx, batch = next(batch_enumerator)

        environ.set_inputs(batch)
        environ.optimize_fix_policy(opt["lambdas"])

        if should(current_iter, opt["train"]["print_freq"]):
            environ.print_loss(current_iter, start_time)

        if should(current_iter, opt["train"]["val_freq"]):
            environ.eval()

            val_metrics = eval_fix_policy(opt, environ, val_loader, opt["tasks"])
            environ.print_loss(current_iter, start_time, val_metrics)
            environ.save(
                "retrain%03d_policyIter%s_latest"
                % (exp_id, opt["train"]["policy_iter"]),
                current_iter,
            )
            environ.train()
            new_value = 0

            for k in refer_metrics.keys():
                if k in val_metrics.keys():
                    for kk in val_metrics[k].keys():
                        if not kk in refer_metrics[k].keys():
                            continue
                        if (
                            (k == "sn" and kk in ["Angle Mean", "Angle Median"])
                            or (k == "depth" and not kk.startswith("sigma"))
                            or (kk == "err")
                        ):
                            value = refer_metrics[k][kk] / val_metrics[k][kk]
                        else:
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
                environ.save(
                    "retrain%03d_policyIter%s_best"
                    % (exp_id, opt["train"]["policy_iter"]),
                    current_iter,
                )
            print("new value: %.3f" % new_value)
            print(
                "best iter: %d, best value: %.3f" % (best_iter, best_value),
                best_metrics,
            )

        if batch_idx == len(train_loader) - 1:
            batch_enumerator = enumerate(train_loader)
        bar.update(current_iter)

    return best_metrics


def train():
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    print_separator(" READ YAML ")
    opt, gpu_ids, exp_ids = read_yaml()

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

    best_results = {}
    for exp_id in exp_ids:
        fix_random_seed(opt["seed"][exp_id])
        best_results = _train(exp_id, opt, gpu_ids)


if __name__ == "__main__":
    train()
