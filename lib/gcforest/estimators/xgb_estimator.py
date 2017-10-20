# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import os, os.path as osp
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold

from ..utils.log_utils import get_logger
from ..utils.cache_utils import name2path

LOGGER = get_logger("gcforest.estimators.xgb_estimator")

# def check_dir(path):
#     d = osp.abspath(osp.join(path, osp.pardir))
#     if not osp.exists(d):
#         os.makedirs(d)

class GCXGBClassifier(object):
    """
    K-Fold Wrapper
    """
    def __init__(self, name, n_folds, est_args, random_state=None):
        """
        Parameters
        ----------
        n_folds (int): 
            Number of folds.
            If n_folds=1, means no K-Fold
        est_args (dict):
            Arguments of estimator
        random_state (int):
            random_state used for KFolds split and Estimator
        """
        self.name = name
        self.n_folds = n_folds
        self.est_args = est_args
        self.random_state = random_state
        self.estimator1d = [None for k in range(self.n_folds)]
        # self.cache_suffix = ".pkl"

    # def __init__(self, name, est_class, est_args):
    #     """
    #     name: str)
    #         Used for debug and as the filename this model may be saved in the disk
    #     """
    #     self.name = name
    #     self.est_class = xgb
    #     self.est_args = est_args
    #     self.cache_suffix = ".pkl"
    #     self.est = None

    # def _init_estimator(self, k):
    #     est_args = self.est_args.copy()
    #     est_name = "{}/{}".format(self.name, k)
    #     est_args["random_state"] = self.random_state
    #     return self.est_class(est_name, est_args)

    def fit_transform(self, X, y, y_stratify, cache_dir=None, test_sets=None, eval_metrics=None, keep_model_in_mem=True):
        """
        X (ndarray):
            n x k or n1 x n2 x k
            to support windows_layer, X could have dim >2 
        y (ndarray):
            n or n1 x n2
        y_stratify (list):
            used for StratifiedKFold or None means no stratify
        test_sets (list): optional
            A list of (prefix, X_test, y_test) pairs.
            predict_proba for X_test will be returned 
            use with keep_model_in_mem=False to save mem useage
            y_test could be None, otherwise use eval_metrics for debugging
        eval_metrics (list): optional
            A list of (str, callable functions)
        keep_model_in_mem (bool):
        """
        if cache_dir is not None:
            cache_dir = osp.join(cache_dir, name2path(self.name))
        assert 2 <= len(X.shape) <= 3, "X.shape should be n x k or n x n2 x k"
        assert len(X.shape) == len(y.shape) + 1
        assert X.shape[0] == len(y_stratify)
        test_sets = test_sets if test_sets is not None else []
        eval_metrics = eval_metrics if eval_metrics is not None else []
        # K-Fold split
        n_stratify = X.shape[0]
        if self.n_folds == 1:
            cv = [(range(len(X)), range(len(X)))]
        else:
            if y_stratify is None:
                skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                cv = [(t, v) for (t, v) in skf.split(len(n_stratify))]
            else:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_stratify)]

        # Fit
        y_probas = []
        n_dims = X.shape[-1]
        n_datas = X.size // n_dims
        inverse = False
        for k in range(self.n_folds):
            # est = self._init_estimator(k)
            if not inverse:
                train_idx, val_idx = cv[k]
            else:
                val_idx, train_idx = cv[k]
            # fit on k-fold train

            xg_train = xgb.DMatrix(X[train_idx].reshape((-1, n_dims)), label=y[train_idx].reshape(-1))
            xg_test = xgb.DMatrix(X[val_idx].reshape((-1, n_dims)), label=y[val_idx].reshape(-1))
            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            LOGGER.info("X_train.shape={}, y_train.shape={}".format(X.shape, y.shape))
            num_round = 100
            est = xgb.train(self.est_args, dtrain=xg_train, num_boost_round=num_round, 
                evals=watchlist, verbose_eval=5)
            y_pred = est.predict(xg_test)
            y_proba = []
            for item in y_pred:
                tmp = []
                tmp.append(1-item)
                tmp.append(item)
                y_proba.append(tmp)
            y_proba = np.array(y_proba)
            LOGGER.info("y_proba.shape={}".format(y_proba.shape))
            # est.fit(X[train_idx].reshape((-1, n_dims)), y[train_idx].reshape(-1), cache_dir=cache_dir)

            # predict on k-fold validation
            # y_proba = est.predict_proba(X[val_idx].reshape((-1, n_dims)), cache_dir=cache_dir)
            if len(X.shape) == 3:
                y_proba = y_proba.reshape((len(val_idx), -1, y_proba.shape[-1]))
            self.log_eval_metrics(self.name, y[val_idx], y_proba, eval_metrics, "train_{}".format(k))

            # merging result
            if k == 0:
                if len(X.shape) == 2:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]), dtype=np.float32)
                else:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1], y_proba.shape[2]), dtype=np.float32)
                y_probas.append(y_proba_cv)
            y_probas[0][val_idx, :] += y_proba
            if keep_model_in_mem:
                self.estimator1d[k] = est

            # test
            for vi, (prefix, X_test, y_test) in enumerate(test_sets):
                # y_pred = est.predict(X_test.reshape((-1, n_dims)), cache_dir=cache_dir)
                xg_test = xgb.DMatrix(X_test.reshape((-1, n_dims)), label=y_test.reshape(-1))
                y_pred = est.predict(xg_test)
                y_proba = []
                for item in y_pred:
                    tmp = []
                    tmp.append(1-item)
                    tmp.append(item)
                    y_proba.append(tmp)
                y_proba = np.array(y_proba)
                LOGGER.info("y_proba.shape={}".format(y_proba.shape))

                if len(X.shape) == 3:
                    y_proba = y_proba.reshape((X_test.shape[0], X_test.shape[1], y_proba.shape[-1]))
                if k == 0:
                    y_probas.append(y_proba)
                else:
                    y_probas[vi + 1] += y_proba
        if inverse and self.n_folds > 1:
            y_probas[0] /= (self.n_folds - 1)
        for y_proba in y_probas[1:]:
            y_proba /= self.n_folds
        # log
        self.log_eval_metrics(self.name, y, y_probas[0], eval_metrics, "train")
        for vi, (test_name, X_test, y_test) in enumerate(test_sets):
            if y_test is not None:
                self.log_eval_metrics(self.name, y_test, y_probas[vi + 1], eval_metrics, test_name)
        return y_probas

    def log_eval_metrics(self, est_name, y_true, y_proba, eval_metrics, y_name):
        """
        y_true (ndarray): n or n1 x n2
        y_proba (ndarray): n x n_classes or n1 x n2 x n_classes
        """
        if eval_metrics is None:
            return
        for (eval_name, eval_metric) in eval_metrics:
            accuracy = eval_metric(y_true, y_proba)
            LOGGER.info("Accuracy({}.{}.{})={:.2f}%".format(est_name, y_name, eval_name, accuracy * 100.))

    # def fit(self, X, y, cache_dir=None):
    #     """
    #     cache_dir(str): 
    #         if not None
    #             then if there is something in cache_dir, dont have fit the thing all over again
    #             otherwise, fit it and save to model cache 
    #     """
    #     LOGGER.debug("X_train.shape={}, y_train.shape={}".format(X.shape, y.shape))
    #     cache_path = self._cache_path(cache_dir)
    #     # cache
    #     if self._is_cache_exists(cache_path):
    #         LOGGER.info("Find estimator from {} . skip process".format(cache_path))
    #         return
    #     est = self._init_estimator()
    #     self._fit(est, X, y)
    #     if cache_path is not None:
    #         # saved in disk
    #         LOGGER.info("Save estimator to {} ...".format(cache_path))
    #         check_dir(cache_path); 
    #         self._save_model_to_disk(self.est, cache_path)
    #         self.est = None
    #     else:
    #         # keep in memory
    #         self.est = est

    # def predict_proba(self, X, cache_dir=None, batch_size=None):
    #     LOGGER.debug("X.shape={}".format(X.shape))
    #     cache_path = self._cache_path(cache_dir)
    #     # cache
    #     if cache_path is not None:
    #         LOGGER.info("Load estimator from {} ...".format(cache_path))
    #         est = self._load_model_from_disk(cache_path)
    #         LOGGER.info("done ...")
    #     else:
    #         est = self.est
    #     batch_size = batch_size or self._default_predict_batch_size(est, X)
    #     if batch_size > 0:
    #         y_proba = self._batch_predict_proba(est, X, batch_size)
    #     else:
    #         y_proba = self._predict_proba(est, X)
    #     LOGGER.debug("y_proba.shape={}".format(y_proba.shape))
    #     return y_proba

    # def _cache_path(self, cache_dir):
    #     if cache_dir is None:
    #         return None
    #     return osp.join(cache_dir, name2path(self.name) + self.cache_suffix)

    # def _is_cache_exists(self, cache_path):
    #     return cache_path is not None and osp.exists(cache_path)

    # def _batch_predict_proba(self, est, X, batch_size):
    #     LOGGER.debug("X.shape={}, batch_size={}".format(X.shape, batch_size))
    #     if hasattr(est, "verbose"):
    #         verbose_backup = est.verbose
    #         est.verbose = 0
    #     n_datas = X.shape[0]
    #     y_pred_proba = None
    #     for j in range(0, n_datas, batch_size):
    #         LOGGER.info("[progress][batch_size={}] ({}/{})".format(batch_size, j, n_datas))
    #         y_cur = self._predict_proba(est, X[j:j+batch_size])
    #         if j == 0:
    #             n_classes = y_cur.shape[1]
    #             y_pred_proba = np.empty((n_datas, n_classes), dtype=np.float32)
    #         y_pred_proba[j:j+batch_size,:] = y_cur
    #     if hasattr(est, "verbose"):
    #         est.verbose = verbose_backup
    #     return y_pred_proba

    # def _load_model_from_disk(self, cache_path):
    #     raise NotImplementedError()

    # def _save_model_to_disk(self, est, cache_path):
    #     raise NotImplementedError()

    # def _default_predict_batch_size(self, est, X):
    #     """
    #     You can re-implement this function when inherient this class 

    #     Return
    #     ------
    #     predict_batch_size (int): default=0
    #         if = 0,  predict_proba without batches
    #         if > 0, then predict_proba without baches
    #         sklearn predict_proba is not so inefficient, has to do this
    #     """
    #     return 0

    # def _fit(self, est, X, y, eval_set):
    #     est.fit(X, y, eval_set=eval_set, eval_metric="auc", early_stopping_rounds=10)

    # def _predict_proba(self, est, X):
    #     return est.predict_proba(X)
