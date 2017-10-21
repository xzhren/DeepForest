"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import os.path as osp
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .ds_base import ds_base
from .ds_base import get_dataset_base


def load_data():
    # data_path = osp.join(get_dataset_base(), "driver", "letter-recognition.data")
    # with open(data_path) as f:
    #     rows = [row.strip().split(',') for row in f.readlines()]
    # n_datas = len(rows)
    # X = np.zeros((n_datas, 16), dtype=np.float32)
    # y = np.zeros(n_datas, dtype=np.int32)
    # for i, row in enumerate(rows):
    #     X[i, :] = list(map(float, row[1:]))
    #     y[i] = ord(row[0]) - ord('A')
    # X_train, y_train = X[:16000], y[:16000]
    # X_test, y_test = X[16000:], y[16000:]

    df_train = pd.read_csv(osp.join(get_dataset_base(), "driver", 'train.csv'))
    df_test = pd.read_csv(osp.join(get_dataset_base(), "driver", 'test.csv'))
    target_train = df_train['target'].values
    id_test = df_test['id'].values
    # id_train = df_train['id'].values
    df_train=df_train.drop(['target','id'],axis=1)
    df_test=df_test.drop(['id'], axis = 1)
    print ("The train shape is:",df_train.shape)
    print ('The test shape is:',df_test.shape)
    X = df_train.values
    y = target_train
    X_result = df_test.values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=9487)
    train_index, test_index = list(sss.split(X, y))[0]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print ("The train shape is:",X_train.shape, y_train.shape)
    print ("The test shape is:",X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test, X_result, id_test

class Driver(ds_base):
    def __init__(self, **kwargs):
        super(Driver, self).__init__(**kwargs)
        X_train, y_train, X_test, y_test, X_result, id_test = load_data()
        X, y = self.get_data_by_imageset(X_train, y_train, X_test, y_test)

        X = X[:,np.newaxis,:,np.newaxis]
        X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        print ("The X shape is:",X.shape)
        print ('The y shape is:',y.shape)
        self.X = X
        self.y = y
        self.test = X_result
        self.test_id = id_test
        print ('The test shape is:',self.test.shape)
