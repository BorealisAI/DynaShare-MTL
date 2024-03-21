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
import json
import pickle
import platform
import pandas as pd
import numpy as np
import torch
from torch.utils.data import  DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    """
    Deepchem is required only if working with PCBA
    reference: https://github.com/deepchem/deepchem
    """
    import gzip
    import deepchem
except ImportError:
    print("Deepchem not installed")


class Reader(object):
    """
    DATASET 2: MIMIC-III
    For Download and pre-processing, follow:
    # https://github.com/YerevaNN/mimic3-benchmarks
    """

    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        self._data = pd.read_csv(listfile_path).values
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class MultitaskReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """Reader for multitask learning.
        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)

        def process_ihm(x):
            return list(map(int, x.split(";")))

        def process_los(x):
            x = x.split(";")
            if x[0] == "":
                return ([], [])
            return (
                list(map(int, x[: len(x) // 2])),
                list(map(float, x[len(x) // 2 :])),
            )

        def process_ph(x):
            return list(map(int, x.split(";")))

        def process_decomp(x):
            x = x.split(";")
            if x[0] == "":
                return ([], [])
            return (list(map(int, x[: len(x) // 2])), list(map(int, x[len(x) // 2 :])))

        self._data = [
            (
                fname,
                float(t),
                process_ihm(ihm),
                process_los(los),
                process_ph(pheno),
                process_decomp(decomp),
            )
            for fname, t, ihm, los, pheno, decomp in self._data
        ]

    def _read_timeseries(self, ts_filename):
        ret = []
        tsfile = pd.read_csv(self._dataset_dir + "/" + ts_filename)
        tsfile = tsfile.fillna("")
        header = tsfile.columns
        assert header[0] == "Hours"
        return (tsfile.values, header)

    def read_example(self, index):
        """Reads the example with given index.
        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Return dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            ihm : array
                Array of 3 integers: [pos, mask, label].
            los : array
                Array of 2 arrays: [masks, labels].
            pheno : array
                Array of 25 binary integers (phenotype labels).
            decomp : array
                Array of 2 arrays: [masks, labels].
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError(
                "Index must be from 0 (inclusive) to number of lines (exclusive)."
            )

        name = self._data[index][0]
        (X, header) = self._read_timeseries(name)

        return {
            "X": X,
            "t": self._data[index][1],
            "ihm": self._data[index][2],
            "los": self._data[index][3],
            "pheno": self._data[index][4],
            "decomp": self._data[index][5],
            "header": header,
            "name": name,
        }


class Discretizer:
    """
    Transform raw data (in text file, with categories) into a matrix
    Original Data: 18 columns
    New Data: 76 columns (due to cagetorical variables)
    """

    def __init__(
        self,
        timestep=0.8,
        store_masks=True,
        impute_strategy="zero",
        start_time="zero",
        # TODO parameterize the address
        config_path="/home/MTL/data/mimiciii/multitask/discretizer_config.json",
    ):
        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config["id_to_channel"]
            self._channel_to_id = dict(
                zip(self._id_to_channel, range(len(self._id_to_channel)))
            )
            self._is_categorical_channel = config["is_categorical_channel"]
            self._possible_values = config["possible_values"]
            self._normal_values = config["normal_values"]

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # For statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps
        if self._start_time == "relative":
            first_time = ts[0]
        elif self._start_time == "zero":
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - 0
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]

        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:
                if channel == "Glascow coma scale total":
                    value = str(int(value))
                elif channel == "Capillary refill rate":
                    value = str(value)
                category_id = self._possible_values[channel].index(value)
                category_id = int(category_id)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)

        for row in X:
            t = float(row[0]) - 0
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1
                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        if self._impute_strategy not in ["zero", "normal_value", "previous", "next"]:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ["normal_value", "previous"]:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(
                            original_value[bin_id][channel_id]
                        )
                        continue
                    if self._impute_strategy == "normal_value":
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == "previous":
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == "next":
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins - 1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(
                            original_value[bin_id][channel_id]
                        )
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print(
            "\taverage unused data = {:.2f} percent".format(
                100.0 * self._unused_data_sum / self._done_count
            )
        )
        print(
            "\taverage empty  bins = {:.2f} percent".format(
                100.0 * self._empty_bins_sum / self._done_count
            )
        )


class Normalizer:
    """
    Normalize continuos variables in the dataset based on their names and a resource file
    """

    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x ** 2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x ** 2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(
                1.0
                / (N - 1)
                * (
                    self._sum_sq_x
                    - 2.0 * self._sum_x * self._means
                    + N * self._means ** 2
                )
            )
            self._stds[self._stds < eps] = eps
            pickle.dump(
                obj={"means": self._means, "stds": self._stds},
                file=save_file,
                protocol=2,
            )

    def load_params(self, load_file_path):
        # Load params from resources folder
        path = __file__.replace("data_preprocessing.py", "")
        load_file_path = load_file_path

        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == "2":
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding="latin1")
        self._means = dct["means"]
        self._stds = dct["stds"]

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret


class Data(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        reader,
        discretizer,
        normalizer,
        ihm_pos,
        seqlen=100,
        shuffle=True,
        return_names=False,
        small_part=None,
    ):
        "Initialization"
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.ihm_pos = ihm_pos
        self.shuffle = shuffle
        self.return_names = return_names
        self.reader = reader
        N = reader.get_number_of_examples()
        if small_part is not None:
            self.N = int(N * small_part)
        else:
            self.N = N

        self.seqlen = seqlen

    def _preprocess_single(self, X, max_time, ihm, decomp, los, pheno):

        timestep = self.discretizer._timestep
        eps = 1e-6

        def get_bin(t):
            return int(t / timestep - eps)

        n_steps = get_bin(max_time) + 1
        X = self.discretizer.transform(X, end=n_steps)[0]
        if self.normalizer is not None:
            X = self.normalizer.transform(X)

        assert len(X) == n_steps

        # ihm
        # NOTE: when mask is 0, we set y to be 0. This is important
        #       because in the multitask networks when ihm_M = 0 we set
        #       our prediction thus the loss will be 0.
        if np.equal(ihm[1], 0):
            ihm[2] = 0
        ihm = (np.int32(ihm[1]), np.int32(ihm[2]))  # mask, label

        # decomp
        decomp_M = [0] * self.seqlen
        decomp_y = [0] * self.seqlen
        # los
        los_M = [0] * self.seqlen
        los_y = [0] * self.seqlen  # n_steps

        for i in range(self.seqlen):
            pos = get_bin(i)
            if i < len(decomp[0]):
                decomp_M[pos] = decomp[0][i]
                decomp_y[pos] = decomp[1][i]
                los_M[pos] = los[0][i]
                partition = int(los[1][i] / 24)
                if partition > 8 and partition < 15:
                    partition = 8
                elif partition >= 15:
                    partition = 9
                los_y[pos] = partition
            else:
                decomp_M[pos] = 0
                decomp_y[pos] = 0
                los_M[pos] = 0
                los_y[pos] = 0

        decomp = (
            np.array(decomp_M, dtype=np.int32),
            np.array(decomp_y, dtype=np.int32),
        )

        los = (np.array(los_M, dtype=np.int32), np.array(los_y, dtype=np.float32))

        pheno = np.array(pheno, dtype=np.int32)

        return (X, ihm, decomp, los, pheno)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.reader.read_example(idx)

    def pad_collate(self, batch):
        """
        Input: batch from loader
        Do: Data Organization and add it to torch
        Return: pad sequence
        """
        data0 = {}
        for i in range(len(batch)):
            for k, v in batch[i].items():
                if k not in data0:
                    data0[k] = []
                data0[k].append(v)
        data0["header"] = data0["header"][0]

        Xs = data0["X"]
        ts = data0["t"]
        ihms = data0["ihm"]
        loss = data0["los"]
        phenos = data0["pheno"]
        decomps = data0["decomp"]

        data = dict()
        data["pheno_ts"] = ts
        data["names"] = data0["name"]
        data["decomp_ts"] = []
        data["los_ts"] = []

        for i in range(len(batch)):
            data["decomp_ts"].append(
                [pos for pos, m in enumerate(decomps[i][0]) if m == 1]
            )
            data["los_ts"].append([pos for pos, m in enumerate(loss[i][0]) if m == 1])
            (Xs[i], ihms[i], decomps[i], loss[i], phenos[i]) = self._preprocess_single(
                Xs[i], ts[i], ihms[i], decomps[i], loss[i], phenos[i]
            )

        data["X"] = []
        for i in range(len(Xs)):
            if Xs[i].shape[0] >= self.seqlen:
                X = Xs[i][0 : self.seqlen, :]
            else:
                X = np.concatenate(
                    (Xs[i], np.zeros((self.seqlen - Xs[i].shape[0], Xs[i].shape[1]))),
                    axis=0,
                )
            data["X"].append(torch.tensor(X))

        # _y is the observed data, _M is the mask (0 if imputed value, 1 if observed value)
        data["ihm_y"] = [torch.tensor(x[1]) for x in ihms]
        data["decomp_y"] = [torch.tensor(x[1]) for x in decomps]
        data["los_y"] = [torch.tensor(x[1]) for x in loss]
        data["pheno_y"] = torch.tensor(phenos)

        # Putting on the LSTM input format
        X = pad_sequence(data["X"], batch_first=True, padding_value=0)
        los = pad_sequence(data["los_y"], batch_first=True, padding_value=0)
        decomp = pad_sequence(data["decomp_y"], batch_first=True, padding_value=0)
        ihm = torch.tensor(data["ihm_y"])
        pheno = data["pheno_y"]
        return X, pheno, los, decomp, ihm


def data_preparation_mimic3(dataroot, bach_size, seqlen=100, small_part=None):
    path = dataroot
    args = {"timestep": 1, "mode": "train", "batch_size": bach_size}

    # Create functions to read bachs
    train_reader = MultitaskReader(
        dataset_dir=os.path.join(path + "/multitask/train/"),
        listfile=os.path.join(path + "/multitask/train_listfile.csv"),
    )
    validation_reader = MultitaskReader(
        dataset_dir=os.path.join(path + "/multitask/train/"),
        listfile=os.path.join(path + "/multitask/val_listfile.csv"),
    )
    test_reader = MultitaskReader(
        dataset_dir=os.path.join(path + "/multitask/test/"),
        listfile=os.path.join(path + "/multitask/test_listfile.csv"),
    )

    # The data from Multitaskreader is row (text mixed with continuos and categorical data)
    # The discretizer and normalizer organize the data
    # The data changes from 18 features to 76 columns (new columns due to cat data)
    discretizer = Discretizer(
        timestep=1.0, store_masks=True, impute_strategy="previous", start_time="zero"
    )
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[
        1
    ].split(",")
    cont_channels = [
        i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
    ]
    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = None
    if normalizer_state is None:
        normalizer_state = "mult_ts{}.input_str_{}.start_time_zero.normalizer".format(
            1.0, "previous"
        )
        # TODO parameterize the address
        normalizer_state = os.path.join(
            "/home/MTL/data/mimiciii/multitask/", normalizer_state
        )
    normalizer.load_params(normalizer_state)

    args_dict = {}
    args_dict["header"] = discretizer_header
    args_dict["ihm_pos"] = int(48.0 / args["timestep"] - 1e-6)

    dataset_train = Data(
        reader=train_reader,
        discretizer=discretizer,
        normalizer=normalizer,
        ihm_pos=args_dict["ihm_pos"],
        seqlen=seqlen,
        small_part=small_part,
    )
    dataset_val = Data(
        reader=validation_reader,
        discretizer=discretizer,
        normalizer=normalizer,
        ihm_pos=args_dict["ihm_pos"],
        seqlen=seqlen,
        small_part=small_part,
    )
    dataset_test = Data(
        reader=test_reader,
        discretizer=discretizer,
        normalizer=normalizer,
        ihm_pos=args_dict["ihm_pos"],
        seqlen=seqlen,
        small_part=small_part,
    )

    print(
        "Train - {}, Validation - {}, Test - {}".format(
            dataset_train.N, dataset_val.N, dataset_test.N
        )
    )

    """ Required: Create DataLoader for training the models """
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=dataset_train.pad_collate,
    )
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=dataset_test.pad_collate,
    )
    val_loader = DataLoader(
        dataset=dataset_val,
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=dataset_val.pad_collate,
    )

    # All possible tasks info
    task_info = {"pheno": 25, "los": 10, "decomp": 1, "ihm": 1}
    task_number = {"pheno": 0, "los": 1, "decomp": 2, "ihm": 3}
    return train_loader, val_loader, test_loader, task_info, task_number


def data_preparation_newdataset(params, data_directory="/home/"):
    """
    DATASET X: Template
    """
    dataset_train = pd.read_csv(data_directory + "train_features.csv")
    dataset_validation = pd.read_csv(data_directory + "val_features.csv")
    dataset_test = pd.read_csv(data_directory + "test_features.csv")

    train_loader = DataLoader(
        dataset_train, shuffle=params["shuffle"], batch_size=params["batch_size"]
    )
    validation_loader = DataLoader(
        dataset_validation, shuffle=params["shuffle"], batch_size=params["batch_size"]
    )
    test_loader = DataLoader(
        dataset_test, shuffle=False, batch_size=params["batch_size"]
    )

    num_features = dataset_train.shape[1]
    num_tasks = params["num_tasks"]
    output_info = None

    return (
        train_loader,
        validation_loader,
        test_loader,
        num_features,
        num_tasks,
        output_info,
    )
