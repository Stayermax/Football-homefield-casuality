import pandas as pd
pd.options.display.width = 0
import numpy as np
import random
import sqlite3
import os
from copy import deepcopy

import data_preprocesing as dpp
import visualisation as vis

random.seed(179)
np.random.seed(179)

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

    # SVM = svm(data, T)
    # propensity_scores = [prob_t_is_[1] for prob_t_is_ in SVM.predict_proba(data)]

    # print(propensity_scores)
    return propensity_scores

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

def ATTs_for_data(datafile):
    df = pd.read_csv(datafile)
    results = {}
    data, T, Y = dpp.data_preprocessing(df)

    IPW_ATT, propensity_scores = IPW(data, T, Y)
    results['1'] = IPW_ATT
    print(f"IPW ATT: {IPW_ATT}")

    S_learner_ATT = S_learner(data, T, Y)
    results['2'] = S_learner_ATT
    print(f"S-learner ATT: {S_learner_ATT}")

    T_learner_ATT = T_learner(data, T, Y)
    results['3'] = T_learner_ATT
    print(f"T-learner ATT: {T_learner_ATT}")

    Matching_ATT = Matching(data, T, Y)
    results['4'] = Matching_ATT
    print(f"Matching ATT: {Matching_ATT}")

    Сompetition_ATT = max([IPW_ATT, S_learner_ATT, T_learner_ATT, Matching_ATT])
    results['5'] = Сompetition_ATT
    print(f"Competition ATT: {Сompetition_ATT}")

    return results, propensity_scores

def get_match_df(datafile, load = False):
    """

    :param datafile: sqlite databese
    :param load:  if True loads preprocessed tables from preprocessed_data folder
    :return: preprocessed match_df with some blanks in odds columns
    """
    team_df_path = 'preprocessed_data/team_df.csv'
    team_att_df_path = 'preprocessed_data/team_attributes_df.csv'
    match_df_path = 'preprocessed_data/match_df.csv'

    if (os.path.exists(team_df_path) and load):
        team_df = pd.read_csv(team_df_path)
    else:
        con = sqlite3.connect(datafile)

        db_tables = pd.read_sql_query("SELECT * from sqlite_sequence", con).set_index('name').to_dict()[
            'seq']  # matches_df
        print(f"Database tables: {db_tables}")
        team_df = pd.read_sql_query("SELECT * from Team", con)
        team_df = dpp.team_table_update(team_df)

    if (os.path.exists(team_att_df_path) and load):
        team_att_df = pd.read_csv(team_att_df_path, parse_dates=['date'], index_col=0)
    else:
        team_att_df = pd.read_sql_query("SELECT * from Team_Attributes", con)
        team_df, team_att_df = dpp.team_att_table_update(team_df, team_att_df)

        team_df.to_csv(team_df_path)
        team_att_df.to_csv(team_att_df_path)

    if (os.path.exists(match_df_path) and load):
        match_df = pd.read_csv(match_df_path, parse_dates=['date'], index_col=0)
    else:
        match_df = pd.read_sql_query("SELECT * from Match", con)  # matches_df
        country_df = pd.read_sql_query("SELECT * from Country", con) # countries df
        match_df = dpp.match_table_update(team_df, team_att_df, country_df, match_df )
        match_df.to_csv(match_df_path)

    return match_df

def get_data(match_df, condition, loadFlag = False, loadHelperTables = False):
    data_path = f"preprocessed_data/Conditions_data/{condition}_algorithm_data.csv"
    if (os.path.exists(data_path) and loadFlag):
        data = pd.read_csv(data_path, index_col=0)
    else:
        data = dpp.match_data_preprocessing(match_df, condition, loadHelperTables)
        data.to_csv(data_path)

    print(f'Get data function results with {condition}:\n {data}')

    T = data['T']
    Y = data['Y']
    data = data.drop(['T', 'Y'], axis=1)

    return data, T, Y

def ATTs_for_sql_data(datafile):
    loadFlag = False
    loadHelperTables = True
    match_df = get_match_df(datafile, loadFlag)

    results = {}
    data, T, Y = get_data(match_df, "No_conditions", loadFlag, loadHelperTables)

    print("DATA FOR ALGORITHM: ")
    print(data)

    IPW_ATT, propensity_scores = IPW(data, T, Y)
    results['1'] = IPW_ATT
    print(f"IPW ATT: {IPW_ATT}")

    S_learner_ATT = S_learner(data, T, Y)
    results['2'] = S_learner_ATT
    print(f"S-learner ATT: {S_learner_ATT}")

    T_learner_ATT = T_learner(data, T, Y)
    results['3'] = T_learner_ATT
    print(f"T-learner ATT: {T_learner_ATT}")

    Matching_ATT = Matching(data, T, Y)
    results['4'] = Matching_ATT
    print(f"Matching ATT: {Matching_ATT}")

    Сompetition_ATT = max([IPW_ATT, S_learner_ATT, T_learner_ATT, Matching_ATT])
    results['5'] = Сompetition_ATT
    print(f"Competition ATT: {Сompetition_ATT}")

    return results, propensity_scores

def main():

    res1, ps1 = ATTs_for_data('data/data1.csv')
    res2, ps2 = ATTs_for_data('data/data2.csv')
    res_df = pd.DataFrame(columns=['Type', 'data1', 'data2'])

    for i in range(len(res1)):
        res_df = res_df.append({'Type': str(i + 1).split('.')[0], 'data1': res1[f"{i + 1}"], 'data2': res2[f"{i + 1}"]}, ignore_index=True)
    res_df.to_csv('results/ATT results.csv', index=False)

    scores_file = open('results/models propensity.csv', 'w')

    s_1 = 'data1'
    for el in ps1:
        s_1 += ','+str(el)
    scores_file.write(s_1+'\n')

    s_2 = 'data2'
    for el in ps2:
        s_2 += ','+str(el)
    scores_file.write(s_2)

    scores_file.close()

def sql_main():

    res, ps = ATTs_for_sql_data('data/database.sqlite')
    res_df = pd.DataFrame(columns=['Type', 'football_data'])

    for i in range(len(res)):
        res_df = res_df.append({'Type': str(i + 1).split('.')[0], 'football_data': res[f"{i + 1}"]}, ignore_index=True)
    print(res_df)
    import pickle
    pickle.dump(ps, open("p_scores.pkl","wb"))

def graph():
    import pickle
    ps = pickle.load( open( "p_scores.pkl", "rb" ))
    Y_1 = [1]*int(len(ps)/2)
    Y_0 = [0]*int(len(ps)/2)
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.histplot(data=ps[:int(len(ps)/2)])
    plt.show()


if __name__ == '__main__':
    # todo 1 : Check poor logistic model accuracy
    # todo 2 : Random forest works slightly better, but from the graph it seems that we still
    #  have some issues
    # todo 3 : [DONE] Use odds values. Uri said that it can be usefull.
    # todo 4 : CATE model implementation

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
    loadHelperTables = True

    # PART 1: [DONE] DATA PREPROCESSING
    datafile = "data/database.sqlite"
    # match_df_no_odds is file to build graphs from
    match_df_gappy_odds = get_match_df(datafile, loadFlag) # odds are currently empty sometimes
    print("PREPROCESSED DATA FOR GRAPHS: ")
    print(match_df_gappy_odds)
    match_df = deepcopy(match_df_gappy_odds)
    # PART 2: DATA VISUALISATION + SIMPLE STATISTICS
    if(GraphsFlag):
        match_df_gappy_odds = dpp.add_winner_column(match_df_gappy_odds)
        # Possible graphs:
        # 1) Odds prediction accuracies [DONE]
        vis.odds_accuracies(match_df_gappy_odds)
        # 2) Teams parameters distribution by countries [TODO]
        vis.params_by_countries(match_df_gappy_odds)
        # 3) Losses, Drawns, Wins at home [DONE]
        vis.win_lose_at_home(match_df_gappy_odds)
        # 4) Losses, Drawns, Wins by country [DONE]
        vis.win_lose_by_countries(match_df_gappy_odds)
        # 5) Goals distribution by countries (Maybe goal diffs) [DONE]
        vis.goals_distribution_by_countries(match_df_gappy_odds)
        # 6)


    # PART 3.0 : CONDITIONS DATA PREPROCESSING
    # 1) Weather condition
    match_df = deepcopy(dpp.match_table_fill_odds(match_df_gappy_odds))  # now all null odds are equal to 1.0
    data_weather, T_weather, Y_weather = get_data(match_df, "Weather", loadFlag)
    # 2) Similarity condition
    match_df = deepcopy(dpp.match_table_fill_odds(match_df_gappy_odds))  # now all null odds are equal to 1.0
    data_similarity, T_similarity, Y_similarity = get_data(match_df,"Similarity", loadFlag)
    # 3) Rivalry condition
    match_df = deepcopy(dpp.match_table_fill_odds(match_df_gappy_odds))  # now all null odds are equal to 1.0
    data_rivalry, T_rivalry, Y_rivalry = get_data(match_df,"Rivalry", loadFlag, loadHelperTables)
    # 4) Rage condition
    match_df = deepcopy(dpp.match_table_fill_odds(match_df_gappy_odds))  # now all null odds are equal to 1.0
    data_rage, T_rage, Y_rage = get_data(match_df,"Rage", loadFlag, loadHelperTables)
    #
    #
    #
    # # PART 3.5: ALGORITHM DATA PREPROCESSING
    #
    # data, T, Y = get_data(match_df, "No_conditions", loadFlag)
    #
    # print("DATA FOR ALGORITHM: ")
    # print(data)
    # Check that data of match k dublicated right

    # PART 4: CATE IMPLEMENTATION
    # I don't know what this should do

    # # TODO: delete from here
    #
    # data_T = data
    # data_T['T'] = T
    # from sklearn import linear_model as lm
    # from sklearn.metrics import r2_score as r2
    # model = lm.LinearRegression()
    # model.fit(data_T, Y)
    # Y_gag = model.predict(data_T)
    # Y_gag = np.round(Y_gag)
    # print(f"Linear regression score: {r2(Y,Y_gag)}")
    # print(model.intercept_)
    # # todo: add linear treatment effect
    # # todo: causal forest
    #
    # # TODO: delete to here
    #
    # # PART 5: ADVANCED STATISTICS
    # # Maybe some use of CATE results
    # # Check that there are some other thing that influence and correlate with homefield advantage
    #
    # # main() # runs hw4 script on hw4 data
    # # sql_main() # runs hw4 script on our data