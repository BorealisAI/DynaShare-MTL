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


import pandas as pd
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

PATH = "/home/mtl_datasets/census_dataset/"

SEED = 1


def data_preparation_census(params):
    # The column names are from
    # https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    column_names = [
        "age",
        "class_worker",
        "det_ind_code",
        "det_occ_code",
        "education",
        "wage_per_hour",
        "hs_college",
        "marital_stat",
        "major_ind_code",
        "major_occ_code",
        "race",
        "hisp_origin",
        "sex",
        "union_member",
        "unemp_reason",
        "full_or_part_emp",
        "capital_gains",
        "capital_losses",
        "stock_dividends",
        "tax_filer_stat",
        "region_prev_res",
        "state_prev_res",
        "det_hh_fam_stat",
        "det_hh_summ",
        "instance_weight",
        "mig_chg_msa",
        "mig_chg_reg",
        "mig_move_reg",
        "mig_same",
        "mig_prev_sunbelt",
        "num_emp",
        "fam_under_18",
        "country_father",
        "country_mother",
        "country_self",
        "citizenship",
        "own_or_self",
        "vet_question",
        "vet_benefits",
        "weeks_worked",
        "year",
        "income_50k",
    ]

    # Load the dataset in Pandas
    train_df = pd.read_csv(
        "/home/mtl_datasets/census_dataset/census-income.data.gz",
        delimiter=",",
        header=None,
        index_col=None,
        names=column_names,
    )
    other_df = pd.read_csv(
        "/home/mtl_datasets/census_dataset/census-income.test.gz",
        delimiter=",",
        header=None,
        index_col=None,
        names=column_names,
    )
    tasks_dict = {
        "income": "income_50k",
        "marital": "marital_stat",
        "education": "education",
    }

    # First group of tasks according to the paper
    label_columns = [tasks_dict[t] for t in params["tasks"]]

    # One-hot encoding categorical columns
    categorical_columns = [
        "class_worker",
        "det_ind_code",
        "det_occ_code",
        "hs_college",
        "major_ind_code",
        "major_occ_code",
        "race",
        "hisp_origin",
        "sex",
        "union_member",
        "unemp_reason",
        "full_or_part_emp",
        "tax_filer_stat",
        "region_prev_res",
        "state_prev_res",
        "det_hh_fam_stat",
        "det_hh_summ",
        "mig_chg_msa",
        "mig_chg_reg",
        "mig_move_reg",
        "mig_same",
        "mig_prev_sunbelt",
        "fam_under_18",
        "country_father",
        "country_mother",
        "country_self",
        "citizenship",
        "vet_question",
    ]

    for i in list(tasks_dict.keys()):
        if tasks_dict[i] not in label_columns:
            categorical_columns.append(tasks_dict[i])

    train_raw_labels = train_df[label_columns]
    other_raw_labels = other_df[label_columns]
    transformed_train = pd.get_dummies(
        train_df.drop(label_columns, axis=1), columns=categorical_columns
    )
    transformed_other = pd.get_dummies(
        other_df.drop(label_columns, axis=1), columns=categorical_columns
    )

    # Scaling continus columns
    continuous_columns = [
        "age",
        "wage_per_hour",
        "capital_gains",
        "capital_losses",
        "stock_dividends",
        "instance_weight",
        "num_emp",
        "weeks_worked",
        "year",
    ]
    scaler = MinMaxScaler()
    transformed_train[continuous_columns] = scaler.fit_transform(
        transformed_train[continuous_columns]
    )
    transformed_other[continuous_columns] = scaler.fit_transform(
        transformed_other[continuous_columns]
    )

    # Filling the missing column in the other set
    transformed_other["det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily"] = 0

    dict_outputs = {}
    dict_train_labels = {}
    dict_other_labels = {}

    # One-hot encoding categorical labels
    if "income_50k" in label_columns:
        train_income = train_raw_labels.income_50k == " 50000+."
        other_income = other_raw_labels.income_50k == " 50000+."
        train_income = train_income * 1  # turning into binary
        other_income = other_income * 1  # turning into binary
        dict_outputs["income"] = len(set(train_income))
        dict_train_labels["income"] = train_income.values
        dict_other_labels["income"] = other_income.values

    if "marital_stat" in label_columns:
        train_marital = train_raw_labels.marital_stat == " Never married"
        other_marital = other_raw_labels.marital_stat == " Never married"
        train_marital = train_marital * 1  # turning into binary
        other_marital = other_marital * 1  # turning into binary
        dict_outputs["marital"] = len(set(train_marital))
        dict_train_labels["marital"] = train_marital.values
        dict_other_labels["marital"] = other_marital.values

    if "education" in label_columns:
        print(".... CENSUS + EDUCATION TASK")
        edu = [
            " Masters degree(MA MS MEng MEd MSW MBA)",
            "Prof school degree (MD DDS DVM LLB JD)",
            " Bachelors degree(BA AB BS)",
            " Doctorate degree(PhD EdD)",
        ]
        train_education = [1 if e in edu else 0 for e in train_raw_labels.education]
        other_education = [1 if e in edu else 0 for e in other_raw_labels.education]
        other_education = other_education * 1
        train_education = train_education * 1  # turning into binary
        dict_outputs["education"] = len(set(train_education))
        dict_train_labels["education"] = np.array(train_education)
        dict_other_labels["education"] = np.array(other_education)

    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # Split the other dataset into 1:1 validation to test according to the paper
    validation_indices = transformed_other.sample(
        frac=0.5, replace=False, random_state=SEED
    ).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices]
    validation_label = [
        dict_other_labels[key][validation_indices]
        for key in sorted(dict_other_labels.keys())
    ]
    test_data = transformed_other.iloc[test_indices]
    test_label = [
        dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())
    ]
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]
    print("Training data shape = {}".format(train_data.values.shape))
    print("Validation data shape = {}".format(validation_data.values.shape))
    print("Test data shape = {}".format(test_data.values.shape))
    num_features = train_data.values.shape[1]
    num_tasks = np.asmatrix(train_label).transpose().shape[1]
    dataset_train = TensorDataset(
        Tensor(train_data.values), Tensor(np.asmatrix(train_label).transpose())
    )
    dataset_validation = TensorDataset(
        Tensor(validation_data.values),
        Tensor(np.asmatrix(validation_label).transpose()),
    )
    dataset_test = TensorDataset(
        Tensor(test_data.values), Tensor(np.asmatrix(test_label).transpose())
    )

    train_loader = DataLoader(
        dataset_train,
        shuffle=params["shuffle"],
        batch_size=params["train"]["batch_size"],
    )
    validation_loader = DataLoader(
        dataset_validation,
        shuffle=params["shuffle"],
        batch_size=params["train"]["batch_size"],
    )
    test_loader = DataLoader(
        dataset_test, shuffle=False, batch_size=params["train"]["batch_size"]
    )

    print("tasks order", output_info)

    return (
        train_loader,
        validation_loader,
        test_loader,
        num_features,
        num_tasks,
        output_info,
    )
