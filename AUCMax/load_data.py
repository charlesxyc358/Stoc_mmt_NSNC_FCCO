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
TEST_SIZE_GERMAN = BIG_TEST_SIZE
TEST_SIZE_COMPAS = DEF_TEST_SIZE
TEST_SIZE_BANK = DEF_TEST_SIZE
TEST_SIZE_YOW = BIG_TEST_SIZE
TEST_SIZE_ARRHYTHMIA = BIG_TEST_SIZE

N_TRAIN_TOY = 10000


def load_german_data():
    def preprocess_z(x):
        assert x in {"A91", "A92", "A93", "A94", "A95"}
        if x in {"A91", "A93", "A94"}:
            return 1
        else:
            return 0

    def preprocess_y(x):
        assert x in {1, 2}
        return 2*int(x == 1) - 1

    # Generates a dataset with 48 covariates with
    df = pd.read_csv("data/german_credit_data/german.data", sep=" ",
                     header=None)
    df.columns = ["check account", "duration", "credit history", "purpose",
                  "credit amount", "savings/bonds", "employed since",
                  "installment rate", "status and sex",
                  "other debtor/guarantor", "residence since", "property",
                  "age", "other plans", "housing", "existing credits",
                  "job", "number liable people", "telephone",
                  "foreign worker", "credit decision"]
    ind_sex = 8
    Z = np.array([preprocess_z(x) for x in df[df.columns[ind_sex]]])
    Y = np.array([preprocess_y(x) for x in df[df.columns[-1]]])

    cols_X = df.columns[:-1]
    ind_quali = {0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19}
    ind_all = set(range(len(cols_X)))

    X_quanti = df[df.columns[list(ind_all.difference(ind_quali))]].values

    X_quali = df[df.columns[list(ind_quali)]].values

    quali_encoder = OneHotEncoder(categories="auto")
    quali_encoder.fit(X_quali)
    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    X_train, X_test, Z_train, Z_test, Y_train, Y_test = train_test_split(
        X, Z, Y, test_size=TEST_SIZE_GERMAN, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, Y_train, Z_train), (X_test, Y_test, Z_test)



def load_adult_dataset():
    # The continuous variable fnlwgt represents final weight, which is the
    # number of units in the target population that the responding unit
    # represents.
    df_train = pd.read_csv("data/adult_dataset/adult.data", header=None)
    columns = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week",
               "native-country", "salary"]
    df_train.columns = columns
    df_test = pd.read_csv("data/adult_dataset/adult.test", header=None, comment="|")
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
    df = pd.read_csv('data/compas_data/compas-analysis-master/compas-scores.csv')

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


def load_bank_data():
    # It is bank marketing data.
    # bank.csv 462K lines 450 Ko
    # bank-full 4M 614K lines 4.4 Mo
    # https://archive.ics.uci.edu/ml/datasets/bank+marketing
    df = pd.read_csv("data/bank_marketing_data/bank-additional"
                     + "/bank-additional-full.csv", sep=";")

    # Y = np.array([2*int(y == "yes") - 1 for y in df["y"]])
    Y = np.array([int(y == "yes") for y in df["y"]])  ### new: 0 vs 1, by Gang Li
    Z = np.logical_and(df["age"].values <= 60,
                       df["age"].values >= 25).astype(int)

    col_quanti = ["age", "duration", "campaign", "pdays", "previous",
                  "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                  "euribor3m", "nr.employed"]
    col_quali = ["job", "education", "default", "housing", "loan", "contact",
                 "month", "day_of_week", "poutcome"]

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    quali_encoder = OneHotEncoder(categories="auto")
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=TEST_SIZE_BANK, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, Y_train, Z_train), (X_test, Y_test, Z_test)


def load_toy1(n=4000):
    n_tr = n
    n_te = n*2
    n_tot = n_tr + n_te

    # q0 = 1/10
    q1 = 17 / 20

    X = np.random.uniform(0, 1, (n_tot, 2))
    Z = np.random.binomial(1, q1, n_tot)
    Y = np.zeros_like(Z)

    Y[Z == 0] = 2*np.random.binomial(1, X[Z == 0, 0]) - 1
    Y[Z == 1] = 2*np.random.binomial(1, X[Z == 1, 1]) - 1

    return (X[:n_tr], Y[:n_tr], Z[:n_tr]), (X[n_tr:], Y[n_tr:], Z[n_tr:])


def load_toy2(n=4000, q1=1/2):
    n_tr = n
    n_te = n*2
    n_tot = n_tr + n_te

    Z = np.random.binomial(1, q1, n_tot)
    thetas = np.random.uniform(0, np.pi/2, n_tot)
    rs = np.random.uniform(0, 0.5, n_tot) + 0.5*Z
    X = np.array([rs*np.cos(thetas), rs*np.sin(thetas)]).transpose()
    Y = 2*np.random.binomial(1, 2*thetas/np.pi) - 1

    return (X[:n_tr], Y[:n_tr], Z[:n_tr]), (X[n_tr:], Y[n_tr:], Z[n_tr:])


def load_db_by_name(db_name):
    if db_name == "german":
        return load_german_data()
    elif db_name == "adult":
        return load_adult_dataset()
    elif db_name == "compas":
        return load_compas_data()
    elif db_name == "bank":
        return load_bank_data()
    elif db_name == "toy1":
        return load_toy1(n=N_TRAIN_TOY)
    elif db_name == "toy2":
        return load_toy2(n=N_TRAIN_TOY)
    raise ValueError("Wrong db name...")


### adapted from https://github.com/Optimization-AI/LibAUC/blob/1.4.0/libauc/datasets/chexpert.py
class CheXpert(Dataset):
    r"""
        Reference:
            .. [1] Yuan, Zhuoning, Yan, Yan, Sonka, Milan, and Yang, Tianbao.
               "Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification."
               Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
               https://arxiv.org/abs/2012.03173
    """
    def __init__(self, 
                 csv_path, 
                 image_root_path='',
                 image_size=224,
                 use_frontal=True,
                 verbose=False,
                 transforms=None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion', 'Cardiomegaly',],
                 return_index=False,
                 mode='train',
                 set_type = 'train'):
        
    
        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=True)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '', regex=True)
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']  
            
        # # upsample selected cols
        # if use_upsampling:
        #     assert isinstance(upsampling_cols, list), 'Input should be list!'
        #     sampled_df_list = []
        #     for col in upsampling_cols:
        #         print ('Upsampling %s...'%col)
        #         sampled_df_list.append(self.df[self.df[col] == 1])
        #     self.df = pd.concat([self.df] + sampled_df_list, axis=0)


        # impute missing values 
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                # self.df[col].replace(-1, 1, inplace=True)  
                self.df = self.df[self.df[col]!=-1]
                self.df[col].fillna(0, inplace=True) 
            elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other','Fracture','Support Devices']: # other labels
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)
        
        ## train, val ,test split
        np.random.seed(42)
        shffled_ids = np.random.permutation(range(len(self.df)))
        line_1, line_2 = int(len(self.df)*0.8), int(len(self.df)*0.9)
        if set_type =='train':
            self.df = self.df.iloc[shffled_ids[:line_1]]
        elif set_type == 'val':
            self.df = self.df.iloc[shffled_ids[line_1:line_2]]
        elif set_type == 'test':
            self.df = self.df.iloc[shffled_ids[line_2:]]
        
        self._num_images = len(self.df)
        
            
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if len(train_cols)>0: 
            if verbose:
                print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
                print ('-'*30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict

        
        self.mode = mode
        self.image_size = image_size
        self.transforms = transforms
        self.return_index = return_index
        
        self._images_list =  [image_root_path+path for path in self.df['Path'].tolist()]
        self.targets = self.df[train_cols].values.tolist()
        self.sensitive_attr = (self.df['Sex'].values =='Male').astype(int).tolist()
    
        if verbose:
            imratio_list = []
            for class_key, select_col in enumerate(train_cols):
                try:
                    imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                except:
                    if len(self.value_counts_dict[class_key]) == 1 :
                        only_key = list(self.value_counts_dict[class_key].keys())[0]
                        if only_key == 0:
                            self.value_counts_dict[class_key][1] = 0
                            imratio = 0 # no postive samples
                        else:    
                            self.value_counts_dict[class_key][1] = 0
                            imratio = 1 # no negative samples
                        
                imratio_list.append(imratio)
                
                if verbose:
                    #print ('-'*30)
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                    print ()
                    #print ('-'*30)
                
    
    def image_augmentation(self, image):
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train' :
            if self.transforms is None:
                image = self.image_augmentation(image)
            else:
                image = self.transforms(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]  ]]) 
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)

        label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        attr = self.sensitive_attr[idx]

        # if self.return_index:
        #     return image, label, idx   
        return image, label, attr

