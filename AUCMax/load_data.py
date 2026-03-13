#### This file is downloaded from https://github.com/RobinVogel/Learning-Fair-Scoring-Functions/blob/master/load_data.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as tfs
import os
import cv2

DEF_TEST_SIZE = 0.20
BIG_TEST_SIZE = 0.40
TEST_SIZE_COMPAS = DEF_TEST_SIZE

N_TRAIN_TOY = 10000

def load_adult_dataset():
    # The continuous variable fnlwgt represents final weight, which is the
    # number of units in the target population that the responding unit
    # represents.
    df_train = pd.read_csv("data/adult/adult.data", header=None)
    columns = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week",
               "native-country", "salary"]
    df_train.columns = columns
    df_test = pd.read_csv("data/adult/adult.test", header=None, comment="|")
    df_test.columns = columns

    def proc_z(Z):
        return np.array([1 if "Male" in z else 0 for z in Z])

    def proc_y(Y):
        return np.array([1 if ">50K" in y else 0 for y in Y])  ### new: 0 vs 1, by Gang Li

    Z_train, Z_test = [proc_z(s["sex"]) for s in [df_train, df_test]]
    Y_train, Y_test = [proc_y(s["salary"]) for s in [df_train, df_test]]

    col_quanti = ["age", "education-num", "capital-gain",
                  "capital-loss", "hours-per-week"]  # "fnlwgt",
    col_quali = ["workclass", "education", "marital-status", "occupation",
                 "relationship", "race", "sex", "native-country"]

    X_train_quali = df_train[col_quali].values
    X_test_quali = df_test[col_quali].values

    X_train_quanti = df_train[col_quanti]
    X_test_quanti = df_test[col_quanti]

    quali_encoder = OneHotEncoder(categories="auto")  # drop="first")
    quali_encoder.fit(X_train_quali)

    X_train_quali_enc = quali_encoder.transform(X_train_quali).toarray()
    X_test_quali_enc = quali_encoder.transform(X_test_quali).toarray()

    X_train = np.concatenate([X_train_quali_enc, X_train_quanti], axis=1)
    X_test = np.concatenate([X_test_quali_enc, X_test_quanti], axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, Y_train, Z_train), (X_test, Y_test, Z_test)


def load_compas_data():
    # See https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
    # Load the two-year data
    df = pd.read_csv('data/compas/compas-scores.csv')

    # vr = violent recidivism
    # r = recidivism
    # Types of crimes in the USA: felonies and misdemeanors
    interesting_cols = [  # 'compas_screening_date',
        'sex',  # 'dob',
        'age', 'race',
        'juv_fel_count', 'decile_score', 'juv_misd_count',
        'juv_other_count', 'priors_count',
        'days_b_screening_arrest',
        'c_jail_in', 'c_jail_out',
        # 'c_offense_date', 'c_arrest_date',
        # 'c_days_from_compas',
        'c_charge_degree',
        # 'c_charge_desc',
        'is_recid',
        # 'r_charge_degree',
        # 'r_days_from_arrest', 'r_offense_date',  # 'r_charge_desc',
        # 'r_jail_in', 'r_jail_out',
        # 'is_violent_recid', 'num_vr_cases',  # 'vr_case_number',
        # 'vr_charge_degree', 'vr_offense_date',
        # 'vr_charge_desc', 'v_type_of_assessment',
        'v_decile_score',  # 'v_score_text',
        # 'v_screening_date',
        # 'type_of_assessment',
        'decile_score.1',  # 'score_text',
        # 'screening_date'
        ]
    df = df[interesting_cols]
    df = df[np.logical_and(df["days_b_screening_arrest"] >= -30,
                           df["days_b_screening_arrest"] <= 30)]
    df["days_in_jail"] = [a.days for a in (pd.to_datetime(df["c_jail_out"]) -
                                           pd.to_datetime(df["c_jail_in"]))]
    df = df[df["is_recid"] >= 0]
    df = df[df["c_charge_degree"] != "O"]
    # df = df[[x in {"Caucasian", "African-American"} for x in df["race"]]]
    Z = np.array([int(x == "African-American") for x in df["race"]])
    # Y = 2*df["is_recid"].values - 1 ### original: -1 vs 1
    Y = df["is_recid"].values         ### new: 0 vs 1, by Gang Li

    cols_to_delete = ["c_jail_out", "c_jail_in", "days_b_screening_arrest"]
    df = df[[a for a in df.columns if a not in cols_to_delete]]

    col_quanti = ["age", "juv_fel_count", "decile_score", "juv_misd_count",
                  "priors_count", "v_decile_score", "decile_score.1",
                  "days_in_jail"]
    col_quali = ["race", "c_charge_degree"]

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    quali_encoder = OneHotEncoder(categories="auto")
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=TEST_SIZE_COMPAS, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, Y_train, Z_train), (X_test, Y_test, Z_test)


def load_db_by_name(db_name):
    if db_name == "adult":
        return load_adult_dataset()
    elif db_name == "compas":
        return load_compas_data()
    raise ValueError("Wrong db name...")
