import pandas as pd
pd.options.display.width = 0
import numpy as np
import random
import sqlite3
import os
from copy import deepcopy
###########################################
################ I P S W ##################
###########################################

def log_reg(data, T):
    from sklearn.linear_model import LogisticRegression
    LogModel = LogisticRegression(max_iter=10000)
    # data = dpp.scale(data)
    LogModel.fit(data, T)
    print(f"Logistic model accuracy :{LogModel.score(data, T)}")
    return LogModel

def rand_forest(data, T):
    from sklearn.ensemble import RandomForestClassifier
    RandForest = RandomForestClassifier()
    RandForest.fit(data, T)
    print(f"RandForest model accuracy :{RandForest.score(data, T)}")
    return RandForest

def svm(data, T):
    from sklearn.svm import SVR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SVM = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    SVM.fit(data, T)
    print(f"SVM model accuracy :{SVM.score(data, T)}")
    return SVM

def propensity_score(data, T):
    """
    :return: e(x) = p( T = 1 | X = x)
    """
    # LogModel = log_reg(data, T)
    # propensity_scores = [prob_t_is_[1] for prob_t_is_ in LogModel.predict_proba(data)]

    RandomForest = rand_forest(data, T)
    propensity_scores = [prob_t_is_[1] for prob_t_is_ in RandomForest.predict_proba(data)]
    return RandomForest.predict_proba(data)[:, RandomForest.classes_.tolist().index(1)]
    # SVM = svm(data, T)
    # propensity_scores = [prob_t_is_[1] for prob_t_is_ in SVM.predict_proba(data)]

    # print(propensity_scores)
    # return propensity_scores

def IPW(data, T, Y):
    """
    Calculate Inverse Propensity Score Weighting
    :param data:
    :param T:
    :param Y:
    :param propensity_scores:
    :return:
    """
    propensity_scores = propensity_score(data, T)
    arg1 = 0
    arg2 = 0
    arg3 = 0
    for i, y in enumerate(Y):
        arg1 += y * T[i]
        arg2 += (1 - T[i]) * y * propensity_scores[i] / (1 - propensity_scores[i])
        arg3 += (1 - T[i]) * propensity_scores[i] / (1 - propensity_scores[i])
    IPW_ATT = arg1 / sum(T) - arg2 / arg3
    return IPW_ATT, propensity_scores

###########################################
############ S - l e a r n e r ############
###########################################

def _2d_1(data, T):
    data_2 = data.multiply(T, axis="index")
    data_2.columns = [col_name + 'T' for col_name in data_2.columns]
    return data.join(data_2)

def lin_reg(X, y):
    """
    Linear regression model
    :param X:
    :param y:
    :return:
    """
    from sklearn import linear_model
    LinModel = linear_model.LinearRegression()
    LinModel.fit(X, y)
    return LinModel

def S_learner(data, T, Y):
    """
    :param data:
    :param T:
    :param Y:
    :return:
    """
    X = _2d_1(data, T)
    X = X.join(T)
    model = lin_reg(X, Y)
    treated_X_1 = pd.DataFrame(X[X['T'] == 1])
    treated_X_0 = pd.DataFrame(X[X['T'] == 1])
    treated_X_0['T'] = 0
    pred_0 = model.predict(treated_X_0)
    pred_1 = model.predict(treated_X_1)
    S_learner_ATT = 1 / len(pred_1) * (sum(pred_1) - sum(pred_0))
    return S_learner_ATT

###########################################
############ T - l e a r n e r ############
###########################################

def T_learner(data, T, Y):
    """
    :param data:
    :param T:
    :param Y:
    :return:
    """
    T_learner_ATT = 0
    all = data.join(T).join(Y)

    treated = pd.DataFrame(all[all['T'] == 1]).drop('T', axis=1)
    tr_Y = treated['Y']
    tr_X = treated.drop('Y', axis=1)

    control = pd.DataFrame(all[all['T'] == 0]).drop('T', axis=1)
    ctrl_Y = control['Y']
    ctrl_X = control.drop('Y', axis=1)

    model_tr = lin_reg(tr_X, tr_Y)
    model_ctrl = lin_reg(ctrl_X, ctrl_Y)

    pred_0 = model_ctrl.predict(tr_X)
    pred_1 = model_tr.predict(tr_X)

    T_learner_ATT = 1 / len(pred_1) * (sum(pred_1) - sum(pred_0))
    return T_learner_ATT

###########################################
############# M a t c h i n g #############
###########################################

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

def Matching(X, t, y, k=1):
    scaler = StandardScaler()
    X_normed = scaler.fit_transform(X)

    knn0 = KNeighborsRegressor(k, weights='distance')
    knn0.fit(X_normed[t == 0], y[t == 0])
    y0_hat = knn0.predict(X_normed[t == 1])

    Matching_ATT = np.mean(y[t == 1] - y0_hat)
    return Matching_ATT
