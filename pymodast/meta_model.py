from __future__ import print_function
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn.multioutput
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel, ConstantKernel)


def multi_xgb(x,y):
    single_model = xgb.XGBRegressor(booster='gbtree',
                                    objective='reg:squarederror',
                                    eval_metric='rmse',
                                    gamma=0.01,
                                    min_child_weight=0,
                                    max_depth=1,
                                    subsample=0.6,
                                    colsample_bytree=1,
                                    tree_method='exact',
                                    learning_rate=0.14,
                                    n_estimators=300,
                                    nthread=4,
                                    scale_pos_weight=1,
                                    reg_lambda=0.9,
                                    reg_alpha=0,
                                    seed=27)

    model = sklearn.multioutput.MultiOutputRegressor(single_model)
    model.fit(x, y)
    return model


def kriging(x, y):
    kernel = Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, alpha=1e-4)
    gp.fit(x, y)
    print(gp.kernel_)

    return gp
