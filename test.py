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
import numpy as np

from dataloaders.mimic_dataloader import *
from dataloaders.pcba_dataloader import *
from dataloaders.census_dataloader import *
from envs.blockdrop_env import BlockDropEnv
import torch
from utils.util import print_separator, read_yaml, create_path, print_yaml
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


def test():
    # # ********************************************************************
    # # ****************** create folders and print options ****************
    # # ********************************************************************
    print_separator(" READ YAML ")
    opt, gpu_ids, exp_ids = read_yaml()
    create_path(opt)
    lines = print_yaml(opt)
    for line in lines:
        print(line)
    with open(
        os.path.join(opt["paths"]["log_dir"], opt["exp_name"], "opt.txt"), "w+"
    ) as f:
        f.writelines(lines)

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # Load the dataloader
    print_separator(" CREATE DATALOADERS ")
    opt_tmp = opt if opt["policy_model"] == "instance-specific" else None
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
        valset = test_loader

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
    else:
        raise NotImplementedError(
            "Dataset %s is not implemented" % opt["dataload"]["dataset"]
        )

    print("size of validation set: ", len(test_loader))

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
        device=gpu_ids,
        is_train=False,
        opt=opt,
    )

    current_iter = environ.load(
        "retrain%03d_policyIter%s_best" % (exp_ids[0], opt["train"]["policy_iter"])
    )

    print("Evaluating the snapshot saved at %d iter" % current_iter)

    policy_label = "Iter%s_rs%04d" % (
        opt["train"]["policy_iter"],
        opt["seed"][exp_ids[0]],
    )

    if environ.check_exist_policy(policy_label):
        environ.load_policy(policy_label)

    policys = environ.get_current_policy()
    overall_policy = np.concatenate(policys, axis=-1)
    print(overall_policy)

    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    environ.eval()
    val_metrics = eval_fix_policy(opt, environ, test_loader, opt["tasks"], eval_iter=-1)
    print(val_metrics)


if __name__ == "__main__":
    test()
