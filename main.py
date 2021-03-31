import pandas as pd
import numpy as np
import random
import sqlite3
import os
from copy import deepcopy

import data_preprocesing as dpp
from visualisation import visualise_all
import algorithms as algs

pd.options.display.width = 0
random.seed(179)
np.random.seed(179)

#### MODELS ###
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import best_model_selection as bms

def fit_best_models(X, t, y, condition, flags : dict, models_fold):
    if (not os.path.exists(f'{models_fold}/best_classifier_{condition}.pkl') or flags['trainNewNodelsFlag']):
        print(f'Starting to train best_classifier_{condition}.pkl')
        bmodel = bms.BestModel('classifier', init_points=10, n_iter=10, **{'max_iter': 20, 'n_neighbors': -1})
        bmodel.fit(X, t, 5, scoring='balanced_accuracy')
        bmodel.save_model(f'{models_fold}/best_classifier_{condition}.pkl')
        print(bmodel.__dict__)
    else:
        print(f'best_classifier_{condition}.pkl already trained')

    if (not os.path.exists(f'{models_fold}/best_regressor_s_{condition}.pkl') or flags['trainNewNodelsFlag']):
        print(f'Starting to train best_regressor_s_{condition}.pkl')
        Xt = pd.concat([X, t], axis=1)
        rmodel = bms.BestModel('regressor', init_points=10, n_iter=10, **{'max_iter': 20, 'n_neighbors': -1})
        rmodel.fit(Xt, y, 5, scoring='neg_mean_squared_error')
        rmodel.save_model(f'{models_fold}/best_regressor_s_{condition}.pkl')
        print(rmodel.__dict__)
    else:
        print(f'best_regressor_s_{condition}.pkl already trained')

    if (not os.path.exists(f'{models_fold}/best_regressor_t_0_{condition}.pkl') or flags['trainNewNodelsFlag']):
        print(f'Starting to train best_regressor_t_0_{condition}.pkl')
        rmodel = bms.BestModel('regressor', init_points=3, n_iter=3, **{'max_iter': 6, 'n_neighbors': -1})
        rmodel.fit(X[t == 0], y[t == 0], 5, scoring='neg_mean_squared_error')
        rmodel.save_model(f'{models_fold}/best_regressor_t_0_{condition}.pkl')
        print(rmodel.__dict__)
        print(f'Starting to train best_regressor_t_1_{condition}.pkl')
        rmodel = bms.BestModel('regressor', init_points=3, n_iter=3, **{'max_iter': 6, 'n_neighbors': -1})
        rmodel.fit(X[t == 1], y[t == 1], 5, scoring='neg_mean_squared_error')
        rmodel.save_model(f'{models_fold}/best_regressor_t_1_{condition}.pkl')
        print(rmodel.__dict__)
    else:
        print(f'best_regressor_t_{condition}.pkl already trained')

    if (not os.path.exists(f'{models_fold}/best_regressor_x_0_{condition}.pkl') or flags['trainNewNodelsFlag']):
        t_learner_0_model = bms.BestModel()
        t_learner_0_model.load_model(f'{models_fold}/best_regressor_t_0_{condition}.pkl')

        t_learner_1_model = bms.BestModel()
        t_learner_1_model.load_model(f'{models_fold}/best_regressor_t_1_{condition}.pkl')

        D0_hat = t_learner_1_model.predict(X[t==0]) - y[t==0]
        D1_hat = y[t == 1] - t_learner_0_model.predict(X[t == 1])

        print(f'Starting to train best_regressor_x_{condition}.pkl')
        x0_model = bms.BestModel('regressor', init_points=3, n_iter=3, **{'max_iter': 6, 'n_neighbors': -1})
        x0_model.fit(X[t == 0], D0_hat, 5, scoring='neg_mean_squared_error')
        x0_model.save_model(f'{models_fold}/best_regressor_x_0_{condition}.pkl')
        print(x0_model.__dict__)

        print(f'Starting to train best_regressor_x_1_{condition}.pkl')

        x1_model = bms.BestModel('regressor', init_points=3, n_iter=3, **{'max_iter': 6, 'n_neighbors': -1})
        x1_model.fit(X[t == 1], D1_hat, 5, scoring='neg_mean_squared_error')
        x1_model.save_model(f'{models_fold}/best_regressor_x_1_{condition}.pkl')
        print(x1_model.__dict__)
    else:
        print(f'best_regressor_x_{condition}.pkl already trained')

def propensity(X, t, fitted_best_model=None):
    if fitted_best_model is not None:
        acc = np.sum(fitted_best_model.predict(X) == t)/len(t)
        print(f"Model for propensity score accuracy: {acc}")
        return fitted_best_model.predict_proba(X)[:, fitted_best_model._fitted_model.classes_.tolist().index(1)]
    # classifier = LogisticRegression(max_iter=10000)
    classifier = RandomForestClassifier()
    classifier.fit(X, t)
    acc = np.sum(classifier.predict(X) == t)/len(t)
    print(f"Model for propensity score accuracy: {acc}")
    return classifier.predict_proba(X)[:, classifier.classes_.tolist().index(1)]

def ipw(t, y, e):
    # t = t[e != 0]
    # y = y[e != 0]
    # e = e[e != 0]
    #
    # t = t[e != 1]
    # y = y[e != 1]
    # e = e[e != 1]
    return 1/len(e)*np.sum(t*y/e) - 1/len(e)*np.sum((1-t)*y/(1-e))

def s_learner(X, t, y, fitted_best_model=None):
    Xt = pd.concat([X, t], axis=1)
    # Xt0 = pd.concat([X, t * 0], axis=1)[t == 1]
    Xt0 = pd.concat([X, t * 0], axis=1)
    Xt1 = pd.concat([X, t * 0 + 1], axis=1)
    if fitted_best_model is not None:
        y0_hat = fitted_best_model.predict(Xt0)
        y1_hat = fitted_best_model.predict(Xt1)
    else:
        regressor = LinearRegression()
        regressor.fit(Xt, y)
        y0_hat = regressor.predict(Xt0)
        y1_hat = regressor.predict(Xt1)
    return np.mean(y1_hat - y0_hat)

def t_learner(X, t, y, t_learner_0_model=None, t_learner_1_model=None):
    if t_learner_0_model is not None:
        y0_hat = t_learner_0_model.predict(X)
    else:
        regressor0 = RandomForestRegressor()
        regressor0.fit(X[t == 0], y[t == 0])
        y0_hat = regressor0.predict(X)
    if t_learner_1_model is not None:
        y1_hat = t_learner_1_model.predict(X)
    else:
        regressor1 = RandomForestRegressor()
        regressor1.fit(X[t == 1], y[t == 1])
        y1_hat = regressor1.predict(X)
    return np.mean(y1_hat - y0_hat)

def x_learner(X, t, y, e, x_learner_0_model, x_learner_1_model):
    tau0_hat = x_learner_0_model.predict(X)
    tau1_hat = x_learner_1_model.predict(X)
    return np.mean(e*tau0_hat + (1-e)*tau1_hat)

#### END OF MODELS ###


def get_match_df(datafile, flags : dict):
    """

    :param datafile: sqlite database
    :param flags:  flags description defined on higher level
    :return: preprocessed match_df with some blanks in odds columns
    """
    team_df_path = 'preprocessed_data/team_df.csv'
    team_att_df_path = 'preprocessed_data/team_attributes_df.csv'
    match_df_path = 'preprocessed_data/match_df.csv'

    con = sqlite3.connect(datafile) # sqlite cursor

    if (os.path.exists(team_df_path) and flags['loadFlag']):
        team_df = pd.read_csv(team_df_path)
    else:
        db_tables = pd.read_sql_query("SELECT * from sqlite_sequence", con).set_index('name').to_dict()[
            'seq']  # matches_df
        print(f"Database tables: {db_tables}")
        team_df = pd.read_sql_query("SELECT * from Team", con)
        team_df = dpp.team_table_update(team_df)

    if (os.path.exists(team_att_df_path) and flags['loadFlag']):
        team_att_df = pd.read_csv(team_att_df_path, parse_dates=['date'], index_col=0)
    else:
        team_att_df = pd.read_sql_query("SELECT * from Team_Attributes", con)
        team_df, team_att_df = dpp.team_att_table_update(team_df, team_att_df)

        team_df.to_csv(team_df_path)
        team_att_df.to_csv(team_att_df_path)

    if (os.path.exists(match_df_path) and flags['loadFlag']):
        match_df = pd.read_csv(match_df_path, parse_dates=['date'], index_col=0)
    else:
        match_df = pd.read_sql_query("SELECT * from Match", con)  # matches_df
        country_df = pd.read_sql_query("SELECT * from Country", con)  # countries df
        match_df = dpp.match_table_update(team_df, team_att_df, country_df, match_df)
        match_df.to_csv(match_df_path)

    return match_df

def get_data(match_df, condition, flags : dict):
    data_path = f"preprocessed_data/Conditions_data/{condition}_algorithm_data.csv"
    if (os.path.exists(data_path) and flags['loadFlag']):
        data = pd.read_csv(data_path, index_col=0)
    else:
        data = dpp.match_data_preprocessing(match_df, condition, flags)
        data.to_csv(data_path)

    T = data['T']
    Y = data['Y']

    data = data.drop(['T', 'Y',], axis=1)

    print(f'DATA COLUMNS : ')
    cols = list(data.columns)
    for i, el in enumerate(cols):
        print(f"{cols[i]},")
    return data, T, Y

def get_data_multiple_conditions(match_df, conditions, flags : dict):
    conditions.sort()
    condition_str = '_'.join(conditions)
    data_path = f"preprocessed_data/Multiple_conditions_data/{condition_str}_algorithm_data.csv"
    if (os.path.exists(data_path) and flags['loadFlag']):
        data = pd.read_csv(data_path, index_col=0)
    else:
        data = dpp.match_data_preprocessing_multiple_conditions(match_df, conditions, flags)
        data.to_csv(data_path)

    T = data['T']
    Y = data['Y']

    data = data.drop(['T', 'Y',], axis=1)

    print(f'DATA COLUMNS : ')
    cols = list(data.columns)
    for i, el in enumerate(cols):
        print(f"{cols[i]},")
    return data, T, Y


def ATTs_calculation(data, T, Y , flags : dict):
    results = {}

    IPW_ATT, propensity_scores = algs.IPW(data, T, Y)
    results['1'] = IPW_ATT
    print(f"IPW ATT: {IPW_ATT}")

    S_learner_ATT = algs.S_learner(data, T, Y)
    results['2'] = S_learner_ATT
    print(f"S-learner ATT: {S_learner_ATT}")

    T_learner_ATT = algs.T_learner(data, T, Y)
    results['3'] = T_learner_ATT
    print(f"T-learner ATT: {T_learner_ATT}")

    Matching_ATT = algs.Matching(data, T, Y)
    results['4'] = Matching_ATT
    print(f"Matching ATT: {Matching_ATT}")

    Competition_ATT = max([IPW_ATT, S_learner_ATT, T_learner_ATT, Matching_ATT])
    results['5'] = Competition_ATT
    print(f"Competition ATT: {Competition_ATT}")

    return results, propensity_scores

def main_mp():
    import time
    start = time.time()

    # Program parameters:
    # if loadFlag is false, then get_data creates match_df from the scratch.
    # O.W. loads it from the preprocessed_data folder
    loadFlag = False
    # if GraphsFlag is true, program shows graphs in PART 2
    GraphsFlag = False
    # if loadHelperTables is false, then:
    #   Tournament positions for each stage calculates once again from the scratch.
    #   Rage data calculated fron the scratch
    # O.W. loads Tournament positions and Rage data from the preprocessed_data folder
    loadHelperTablesFlag = False
    # if conditionGraphFlag is true, then we build graphs for data with conditions
    conditionGraphFlag = True
    # if trainNewNodelsFlag is true, then we train new models for scores
    trainNewNodelsFlag = False

    models_fold = "models_no_odds"
    # models_fold = "models_backup"

    flags = {
        'loadFlag': loadFlag,
        'GraphsFlag': GraphsFlag,
        'loadHelperTablesFlag': loadHelperTablesFlag,
        'conditionGraphFlag': conditionGraphFlag,
        'trainNewNodelsFlag': trainNewNodelsFlag
    }

    # PART 1: [DONE] DATA PREPROCESSING

    datafile = "data/database.sqlite"
    # match_graph_df is file to build graphs from
    match_graph_df = get_match_df(datafile, flags)  # odds are currently empty sometimes
    print(f"Preprocessed data for graphs: \n{match_graph_df} ")
    match_df = deepcopy(dpp.match_table_fill_odds(match_graph_df))  # now all null odds are equal to 1.0
    print(f"Preprocessed data: \n{match_df} ")
    # PART 2: DATA VISUALISATION + SIMPLE STATISTICS
    visualise_all(match_graph_df, flags)

    # PART 3.0 : CONDITIONS DATA PREPROCESSING
    # 0) No_conditions (Full data)
    # 1) LowStage condition (Stage < 10)
    # 1.5) HighStage condition (Stage > 20)
    # 2) Winter condition (month in [11,12,1])
    # 3) Similarity condition (norm of teams params diffs < threshold)
    # 4) Rivalry condition (Close position in the tournament table)
    # 5) Rage condition (Desire to recoup on the home field)
    # applied_conditions = ['Winter', 'Similarity', 'Rivalry', ]
    applied_conditions = ['Winter', 'Similarity', ]
    applied_conditions_str = "_".join(applied_conditions)
    scores = []
    columns = []
    match_c_df = deepcopy(match_df)
    # data_c, T_c, Y_c = get_data(match_c_df, condition, flags)
    data_c, T_c, Y_c = get_data_multiple_conditions(match_c_df, applied_conditions, flags)
    print(f"Ready for algorithm data with {applied_conditions_str} : \n{data_c}")
    # PART 4: CATE IMPLEMENTATION (Condition already applied)

    ### MODEL:

    X = data_c
    t = T_c
    y = Y_c

    fit_best_models(X, t, y, applied_conditions_str, flags, models_fold)
    propensity_model = bms.BestModel()
    propensity_model.load_model(f'{models_fold}/best_classifier_{applied_conditions_str}.pkl')

    res = {'Type': f'{applied_conditions_str}'}
    e = propensity(data_c, T_c, propensity_model)

    ipw_score = ipw(t, y, e)
    res['ipw_score'] = np.round(ipw_score, 4)
    columns.append('ipw_score')

    s_learner_model = bms.BestModel()
    s_learner_model.load_model(f'{models_fold}/best_regressor_s_{applied_conditions_str}.pkl')
    s_learner_score = s_learner(X, t, y, s_learner_model)
    res['s_learner'] = np.round(s_learner_score, 4)
    columns.append('s_learner')

    t_learner_0_model = bms.BestModel()
    t_learner_0_model.load_model(f'{models_fold}/best_regressor_t_0_{applied_conditions_str}.pkl')
    t_learner_1_model = bms.BestModel()
    t_learner_1_model.load_model(f'{models_fold}/best_regressor_t_1_{applied_conditions_str}.pkl')
    t_learner_score = t_learner(X, t, y, t_learner_0_model, t_learner_1_model)
    res['t_learner'] = np.round(t_learner_score, 4)
    columns.append('t_learner')

    x_learner_0_model = bms.BestModel()
    x_learner_0_model.load_model(f'{models_fold}/best_regressor_x_0_{applied_conditions_str}.pkl')
    x_learner_1_model = bms.BestModel()
    x_learner_1_model.load_model(f'{models_fold}/best_regressor_x_1_{applied_conditions_str}.pkl')
    x_learner_score = x_learner(X, t, y, e, x_learner_0_model, x_learner_1_model)
    res['x_learner'] = np.round(x_learner_score, 4)
    columns.append('x_learner')

    print(f'Result :{res}')
    scores.append(res)
    print(f"scores: {scores}")
    res_df = pd.DataFrame(columns=['Type'] + columns)
    for el in scores:
        # row = [el['Type']]
        # for col in columns:
        #     row.append(el[col])
        res_df = res_df.append(el, ignore_index=True)
    print(res_df.set_index('Type'))
    res_df.to_csv('results.csv')

    runtime = time.time() - start
    print(f"Runtime: {np.round(runtime, 2)} sec")

def main_1c():
    import time
    start = time.time()

    # Program parameters:
    # if loadFlag is false, then get_data creates match_df from the scratch.
    # O.W. loads it from the preprocessed_data folder
    loadFlag = True
    # if GraphsFlag is true, program shows graphs in PART 2
    GraphsFlag = False
    # if loadHelperTables is false, then:
    #   Tournament positions for each stage calculates once again from the scratch.
    #   Rage data calculated fron the scratch
    # O.W. loads Tournament positions and Rage data from the preprocessed_data folder
    loadHelperTablesFlag = True
    # if conditionGraphFlag is true, then we build graphs for data with conditions
    conditionGraphFlag = False
    # if trainNewNodelsFlag is true, then we train new models for scores
    trainNewNodelsFlag = False

    models_fold = "models_no_odds"
    # models_fold = "models_backup"

    flags = {
        'loadFlag': loadFlag,
        'GraphsFlag': GraphsFlag,
        'loadHelperTablesFlag': loadHelperTablesFlag,
        'conditionGraphFlag': conditionGraphFlag,
        'trainNewNodelsFlag': trainNewNodelsFlag
    }

    # PART 1: [DONE] DATA PREPROCESSING

    datafile = "data/database.sqlite"
    # match_graph_df is file to build graphs from
    match_graph_df = get_match_df(datafile, flags)  # odds are currently empty sometimes
    print(f"Preprocessed data for graphs: \n{match_graph_df} ")
    match_df = deepcopy(dpp.match_table_fill_odds(match_graph_df))  # now all null odds are equal to 1.0
    print(f"Preprocessed data: \n{match_df} ")
    # PART 2: DATA VISUALISATION + SIMPLE STATISTICS
    visualise_all(match_graph_df, flags)

    # PART 3.0 : CONDITIONS DATA PREPROCESSING
    # 0) No_conditions (Full data)
    # 1) LowStage condition (Stage < 10)
    # 1.5) HighStage condition (Stage > 20)
    # 2) Winter condition (month in [11,12,1])
    # 3) Similarity condition (norm of teams params diffs < threshold)
    # 4) Rivalry condition (Close position in the tournament table)
    # 5) Rage condition (Desire to recoup on the home field)
    Conditions = [
        'No_conditions',
        'LowStage',
        'HighStage',
        'Winter',
        'Similarity',
        'Rivalry',
        'Rage'
    ]
    # Conditions = ['Similarity']
    scores = []
    columns = []
    for condition in Conditions:
        columns = []
        match_c_df = deepcopy(match_df)
        data_c, T_c, Y_c = get_data(match_c_df, condition, flags)
        print(f"Ready for algorithm data with {condition} : \n{data_c}")
        # PART 4: CATE IMPLEMENTATION (Condition already applied)

        ### MODEL:

        X = data_c
        t = T_c
        y = Y_c

        fit_best_models(X, t, y, condition, flags, models_fold)
        propensity_model = bms.BestModel()
        propensity_model.load_model(f'{models_fold}/best_classifier_{condition}.pkl')

        res = {'Type': f'{condition}'}
        e = propensity(data_c, T_c, propensity_model)

        ipw_score = ipw(t, y, e)
        res['ipw_score'] = np.round(ipw_score, 4)
        columns.append('ipw_score')

        s_learner_model = bms.BestModel()
        s_learner_model.load_model(f'{models_fold}/best_regressor_s_{condition}.pkl')
        s_learner_score = s_learner(X, t, y, s_learner_model)
        res['s_learner'] = np.round(s_learner_score, 4)
        columns.append('s_learner')

        t_learner_0_model = bms.BestModel()
        t_learner_0_model.load_model(f'{models_fold}/best_regressor_t_0_{condition}.pkl')
        t_learner_1_model = bms.BestModel()
        t_learner_1_model.load_model(f'{models_fold}/best_regressor_t_1_{condition}.pkl')
        t_learner_score = t_learner(X, t, y, t_learner_0_model, t_learner_1_model)
        res['t_learner'] = np.round(t_learner_score, 4)
        columns.append('t_learner')

        x_learner_0_model = bms.BestModel()
        x_learner_0_model.load_model(f'{models_fold}/best_regressor_x_0_{condition}.pkl')
        x_learner_1_model = bms.BestModel()
        x_learner_1_model.load_model(f'{models_fold}/best_regressor_x_1_{condition}.pkl')
        x_learner_score = x_learner(X, t, y, e, x_learner_0_model, x_learner_1_model)
        res['x_learner'] = np.round(x_learner_score, 4)
        columns.append('x_learner')

        print(f'Result :{res}')
        scores.append(res)
    print(f"scores: {scores}")
    res_df = pd.DataFrame(columns=['Type'] + columns)
    for el in scores:
        # row = [el['Type']]
        # for col in columns:
        #     row.append(el[col])
        res_df = res_df.append(el, ignore_index=True)
    print(res_df.set_index('Type'))
    res_df.to_csv('results.csv')

    runtime = time.time() - start
    print(f"Runtime: {np.round(runtime, 2)} sec")

if __name__ == '__main__':
    main_1c() # one condition at a time
    # main_mp() # few conditions at a time