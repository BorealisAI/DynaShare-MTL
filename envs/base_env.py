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


import os
import time
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from utils.util import print_current_errors
from sklearn.metrics import (
    roc_auc_score,
    cohen_kappa_score,
)


class BaseEnv:
    """
    The environment to train a simple classification model
    """

    def __init__(
        self,
        log_dir,
        checkpoint_dir,
        exp_name,
        tasks_num_class,
        device=0,
        is_train=True,
        opt=None,
    ):
        """
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param is_train: bool, specify during the training
        """
        print(self.name())
        self.checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
        self.log_dir = os.path.join(log_dir, exp_name)
        self.is_train = is_train
        self.tasks_num_class = tasks_num_class
        self.device_id = device
        self.opt = opt
        self.dataset = self.opt["dataload"]["dataset"]
        self.tasks = self.opt["tasks"]
        if torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % device[0])

        self.networks = {}
        self.define_networks(tasks_num_class)
        self.define_loss()
        self.losses = {}

        self.optimizers = {}
        self.schedulers = {}
        if is_train:
            # define optimizer
            self.define_optimizer()
            self.define_scheduler()
            # define summary writer
            self.writer = SummaryWriter(log_dir=self.log_dir)

    # ##################### define networks / optimizers / losses ####################################

    def define_loss(self):
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
        self.l1_loss2 = nn.L1Loss(reduction="none")
        if self.dataset == "mimic":
            self.cross_entropy_sparsity = nn.CrossEntropyLoss(ignore_index=-1)
            self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
            # pheno
            self.Bcross_entropy_pheno = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.opt["train"]["cw_pheno"]).to(self.device)
            )
            # los
            self.cross_entropy_los = nn.CrossEntropyLoss(
                weight=torch.tensor(self.opt["train"]["cw_los"]).to(self.device)
            )
            # ihm
            self.Bcross_entropy_ihm = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.opt["train"]["cw_ihm"]).to(self.device)
            )
            # decomp
            self.Bcross_entropy_decomp = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.opt["train"]["cw_decomp"]).to(self.device)
            )

        elif self.dataset == "census":
            self.Bcross_entropy_census = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.opt["train"]["cw_census"]).to(self.device)
            )

        elif self.dataset == "pcba":
            self.Bcross_entropy_pcba = [
                nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(self.opt["train"]["cw_pcba"]).to(
                        self.device
                    )
                )
                for i in range(len(self.tasks))
            ]

        elif self.dataset == "Taskonomy":
            dataroot = self.opt["dataload"]["dataroot"]
            weight = (
                torch.from_numpy(
                    np.load(os.path.join(dataroot, "semseg_prior_factor.npy"))
                )
                .to(self.device)
                .float()
            )
            self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
            self.cross_entropy2 = nn.CrossEntropyLoss(
                ignore_index=255, reduction="none"
            )
            self.cross_entropy_sparisty = nn.CrossEntropyLoss(ignore_index=255)

        elif self.dataset == "CityScapes":
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
            self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
            self.cross_entropy_sparsity = nn.CrossEntropyLoss(ignore_index=-1)

        else:
            raise NotImplementedError("Dataset %s is not implemented" % self.dataset)

    def define_networks(self, tasks_num_class):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    # ##################### train / test ####################################
    def set_inputs(self, batch):
        """
        :param batch: {'images': a tensor [batch_size, slen, feature], 'categories': np.ndarray [batch_size,]}
        """
        if self.dataset == "mimic":
            self.img = batch[0].reshape(
                batch[0].shape[0], batch[0].shape[2], batch[0].shape[1]
            )
        elif self.dataset == "census":
            self.img = np.expand_dims(batch[0], axis=1)
        elif self.dataset == "pcba":
            self.img = np.expand_dims(batch[0], axis=1)
        else:
            self.img = batch[0]

        if torch.cuda.is_available():
            if self.dataset == "pcba" or self.dataset == "census":
                self.img = (
                    torch.from_numpy(self.img)
                    .float()
                    .to(self.device, dtype=torch.float)
                )
            else:
                self.img = self.img.to(self.device, dtype=torch.float)

        if "pheno" in self.tasks:
            self.pheno = batch[1]
            if torch.cuda.is_available():
                self.pheno = self.pheno.to(self.device, dtype=torch.float)

        if "los" in self.tasks:
            self.los = batch[2]
            if torch.cuda.is_available():
                self.los = self.los.to(self.device, dtype=torch.long)

        if "decomp" in self.tasks:
            self.decomp = batch[3]
            if torch.cuda.is_available():
                self.decomp = self.decomp.to(self.device, dtype=torch.float)

        if "ihm" in self.tasks:
            self.ihm = batch[4]
            if torch.cuda.is_available():
                self.ihm = self.ihm.to(self.device, dtype=torch.float)

        if "PCBA" in self.tasks[0]:
            self.pcba = batch[1]  # this include label of all tasks
            self.pcba_w = batch[2]

        if "education" in self.tasks:
            self.education = batch[1][:, 0]
            if torch.cuda.is_available():
                self.education = self.education.to(self.device, dtype=torch.float)
        if "income" in self.tasks:
            self.income = batch[1][:, 1]
            if torch.cuda.is_available():
                self.income = self.income.to(self.device, dtype=torch.float)
        if "marital" in self.tasks:
            self.marital = batch[1][:, 2]
            if torch.cuda.is_available():
                self.marital = self.marital.to(self.device, dtype=torch.float)

    def extract_features(self):
        pass

    def get_education_loss(self, instance=False):
        self.losses["education"] = {}
        prediction = self.education_pred
        s0, s1, s3 = prediction.shape
        prediction = prediction.reshape(s0, s1)
        gt = self.education.reshape(s0, s1)
        loss = (
            self.Bcross_entropy_census(prediction, gt).float() * self.opt["lambdas"][0]
        )

        self.losses["education"]["total"] = loss

    def get_income_loss(self, instance=False):
        self.losses["income"] = {}
        prediction = self.income_pred
        s0, s1, s3 = prediction.shape
        prediction = prediction.reshape(s0, s1)
        gt = self.income.reshape(s0, s1)
        loss = (
            self.Bcross_entropy_census(prediction, gt).float() * self.opt["lambdas"][1]
        )
        self.losses["income"]["total"] = loss

    def get_marital_loss(self, instance=False):
        self.losses["marital"] = {}
        prediction = self.marital_pred
        s0, s1, s3 = prediction.shape
        prediction = prediction.reshape(s0, s1)
        gt = self.marital.reshape(s0, s1)
        loss = (
            self.Bcross_entropy_census(prediction, gt).float() * self.opt["lambdas"][2]
        )
        self.losses["marital"]["total"] = loss

    def get_pheno_loss(self, instance=False):
        self.losses["pheno"] = {}
        pred = self.pheno_pred
        s1, s2 = self.pheno_pred.shape
        prediction = pred.reshape(s1 * s2)
        gt = self.pheno.reshape(s1 * s2)
        loss = (
            self.Bcross_entropy_pheno(prediction, gt).float() * self.opt["lambdas"][0]
        )
        self.losses["pheno"]["total"] = loss

    def get_los_loss(self, instance=False):
        self.losses["los"] = {}
        s0, s1, s2 = self.los_pred.float().shape
        prediction = self.los_pred.reshape(s0 * s2, s1)
        gt = self.los.reshape(s0 * s2)
        loss = self.cross_entropy_los(prediction, gt).float() * self.opt["lambdas"][1]
        self.losses["los"]["total"] = loss

    def get_decomp_loss(self, instance=False):
        self.losses["decomp"] = {}
        s0, s1, s2 = self.decomp_pred.float().shape
        prediction = self.decomp_pred.reshape(s0 * s2, s1).float()
        gt = self.decomp.reshape(s0 * s2, s1)
        loss = (
            self.Bcross_entropy_decomp(prediction, gt).float() * self.opt["lambdas"][2]
        )
        self.losses["decomp"]["total"] = loss

    def get_ihm_loss(self, instance=False):
        self.losses["ihm"] = {}
        prediction = self.ihm_pred.float()
        s0, s1 = self.ihm_pred.float().shape
        gt = self.ihm.reshape(s0, s1)
        loss = self.Bcross_entropy_ihm(prediction, gt).float() * self.opt["lambdas"][3]
        self.losses["ihm"]["total"] = loss

    def get_pcba_loss(self, instance=False):

        for t_id, task in enumerate(self.opt["tasks"]):
            self.losses[task] = {}
            pred_temp = getattr(self, "%s_pred" % task)[
                self.pcba_w[:, t_id] > 0
            ].float()
            s0, s1, s3 = pred_temp.shape
            prediction = pred_temp.reshape(s0, s1)
            label = self.pcba[:, t_id].long().to(self.device).reshape(-1, 1)
            gt = label.float()[self.pcba_w[:, t_id] > 0]
            loss = (
                self.Bcross_entropy_pcba[t_id](prediction, gt).float()
                * self.opt["lambdas"][t_id]
            )
            self.losses[task]["total"] = loss

    def pcba_error(self, task_id):
        # For single task in pcba, the same function should be called for all the tasks
        pred_temp = getattr(self, "%s_pred" % self.opt["tasks"][task_id])
        s0, s1, s3 = pred_temp.shape
        prediction = pred_temp.reshape(s0, s1)
        label = self.pcba[:, task_id].long().to(self.device).reshape(-1, 1)
        gt = label.float()
        return prediction, gt

    def pcba_metric(self, prediction, gt, w):
        auc = []
        error = []
        for task_id, task in enumerate(self.tasks):
            auc_temp = roc_auc_score(
                gt[task_id][w[task_id] > 0], prediction[task_id][w[task_id] > 0]
            )
            pred_tensor = (
                torch.tensor(prediction[task_id][w[task_id] > 0])
                .float()
                .to(self.device)
                .reshape(-1, 1)
            )
            gt_tensor = (
                torch.tensor(gt[task_id][w[task_id] > 0])
                .float()
                .to(self.device)
                .reshape(-1, 1)
            )
            error_temp = (
                self.Bcross_entropy_pcba[task_id](pred_tensor, gt_tensor).float()
                * self.opt["lambdas"][task_id]
            )
            auc.append(auc_temp)
            error.append(error_temp.cpu().numpy())
        return auc, error

    def education_error(self):
        prediction = self.education_pred
        s0, s1, s3 = prediction.shape
        prediction = prediction.reshape(s0, s1)
        gt = self.education.reshape(s0, s1)

        error = self.Bcross_entropy_census(prediction, gt).float()
        auc = roc_auc_score(gt.cpu(), prediction.cpu())
        return auc, error.cpu().numpy()

    def income_error(self):
        prediction = self.income_pred
        s0, s1, s3 = prediction.shape
        prediction = prediction.reshape(s0, s1)
        gt = self.income.reshape(s0, s1)

        error = self.Bcross_entropy_census(prediction, gt).float()
        auc = roc_auc_score(gt.cpu(), prediction.cpu())
        return auc, error.cpu().numpy()

    def marital_error(self):
        prediction = self.marital_pred
        s0, s1, s3 = prediction.shape
        prediction = prediction.reshape(s0, s1)
        gt = self.marital.reshape(s0, s1)

        error = self.Bcross_entropy_census(prediction, gt).float()
        auc = roc_auc_score(gt.cpu(), prediction.cpu())
        return auc, error.cpu().numpy()

    def pheno_error(self):
        pred = self.pheno_pred
        s1, s2 = self.pheno_pred.shape
        prediction = pred.reshape(s1 * s2)
        gt = self.pheno.reshape(s1 * s2)

        error = self.Bcross_entropy_pheno(prediction, gt).float()
        auc = roc_auc_score(gt.cpu(), prediction.cpu())
        return auc, error.cpu().numpy()

    def los_error(self):
        s0, s1, s2 = self.los_pred.float().shape
        prediction = self.los_pred.reshape(s0 * s2, s1)
        gt = self.los.reshape(s0 * s2)

        y_pred = torch.log_softmax(prediction, dim=1)
        y_pred1 = torch.softmax(prediction, dim=1)
        prediction_max = (
            y_pred.max(dim=1)[1].cpu().detach().numpy()
        )  # Getting the max value

        error = self.cross_entropy_los(prediction, gt).float()
        kappa = cohen_kappa_score(
            gt.cpu(), prediction_max, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        auc = roc_auc_score(gt.cpu(), y_pred1.cpu(), multi_class="ovo")
        return kappa, auc, error.cpu().numpy()

    def decomp_error(self):
        s0, s1, s2 = self.decomp_pred.float().shape
        prediction = self.decomp_pred.reshape(s0 * s2, s1).float()
        gt = self.decomp.reshape(s0 * s2, s1)

        error = self.Bcross_entropy_decomp(prediction, gt).float()
        auc = roc_auc_score(gt.cpu(), prediction.cpu())
        return auc, error.cpu().numpy()

    def ihm_error(self):
        prediction = self.ihm_pred.float()
        s0, s1 = self.ihm_pred.float().shape
        gt = self.ihm.reshape(s0, s1)

        error = self.Bcross_entropy_ihm(prediction, gt).float()
        auc = roc_auc_score(gt.cpu(), prediction.cpu())
        return auc, error.cpu().numpy()

    # ##################### print loss ####################################
    def get_loss_dict(self):
        loss = {}
        for key in self.losses.keys():
            loss[key] = {}
            for subkey, v in self.losses[key].items():
                loss[key][subkey] = v.data
        return loss

    def print_loss(self, current_iter, start_time, metrics=None):
        if metrics is None:
            loss = self.get_loss_dict()
        else:
            loss = metrics

        print("-------------------------------------------------------------")
        for key in loss.keys():
            print(key + ":")
            for subkey in loss[key].keys():
                self.writer.add_scalar(
                    "%s/%s" % (key, subkey), loss[key][subkey], current_iter
                )
            print_current_errors(
                os.path.join(self.log_dir, "loss.txt"),
                current_iter,
                loss[key],
                time.time() - start_time,
            )

    # ##################### change the state of each module ####################################
    def get_current_state(self, current_iter):
        current_state = {}
        for k, v in self.networks.items():
            if isinstance(v, nn.DataParallel):
                current_state[k] = v.module.state_dict()
            else:
                current_state[k] = v.state_dict()
        for k, v in self.optimizers.items():
            current_state[k] = v.state_dict()
        current_state["iter"] = current_iter
        return current_state

    def save(self, label, current_iter):
        """
        Save the current checkpoint
        :param label: str, the label for the loading checkpoint
        :param current_iter: int, the current iteration
        """
        current_state = self.get_current_state(current_iter)
        save_filename = "%s_model.pth.tar" % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(current_state, save_path)

    def load_snapshot(self, snapshot):
        for k, v in self.networks.items():
            if k in snapshot.keys():
                # loading values for the existed keys
                model_dict = v.state_dict()
                pretrained_dict = {}
                for kk, vv in snapshot[k].items():
                    if kk in model_dict.keys() and model_dict[kk].shape == vv.shape:
                        pretrained_dict[kk] = vv
                    else:
                        print("skipping %s" % kk)
                model_dict.update(pretrained_dict)
                self.networks[k].load_state_dict(model_dict)
        if self.is_train:
            for k, v in self.optimizers.items():
                if k in snapshot.keys():
                    self.optimizers[k].load_state_dict(snapshot[k])
        return snapshot["iter"]

    def load(self, label, path=None):
        """
        load the checkpoint
        :param label: str, the label for the loading checkpoint
        :param path: str, specify if knowing the checkpoint path
        """
        if path is None:
            save_filename = "%s_model.pth.tar" % label
            save_path = os.path.join(self.checkpoint_dir, save_filename)
        else:
            save_path = path
        if os.path.isfile(save_path):
            print("=> loading snapshot from {}".format(save_path))
            snapshot = torch.load(save_path, map_location="cuda:%d" % self.device_id[0])
            return self.load_snapshot(snapshot)
        else:
            raise ValueError("snapshot %s does not exist" % save_path)

    def train(self):
        """
        Change to the training mode
        """
        for k, v in self.networks.items():
            v.train()

    def eval(self):
        """
        Change to the eval mode
        """
        for k, v in self.networks.items():
            v.eval()

    def cuda(self, gpu_ids):
        if len(gpu_ids) == 1:
            for k, v in self.networks.items():
                v.to(self.device)
        else:
            for k, v in self.networks.items():
                self.networks[k] = nn.DataParallel(v, device_ids=gpu_ids)
                self.networks[k].to(self.device)

    def cpu(self):
        for k, v in self.networks.items():
            v.cpu()

    def name(self):
        return "BaseEnv"
