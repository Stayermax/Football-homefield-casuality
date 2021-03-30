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
    if (os.path.exists(data_path) and loadFlag):
        data = pd.read_csv(data_path, index_col=0)
    else:
        data = dpp.match_data_preprocessing(match_df, condition, flags)
        data.to_csv(data_path)

    T = data['T']
    Y = data['Y']
    data = data.drop(['T', 'Y'], axis=1)

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

if __name__ == '__main__':
    # todo 1 : Check poor logistic model accuracy
    # todo 2 : Random forest works slightly better, but from the graph it seems that we still
    #  have some issues
    # todo 4 : CATE model implementation

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

    flags = {
        'loadFlag': loadFlag,
        'GraphsFlag': GraphsFlag,
        'loadHelperTablesFlag': loadHelperTablesFlag,
        'conditionGraphFlag': conditionGraphFlag
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
    Conditions = ['No_conditions',
                  'LowStage',
                  'HighStage',
                  'Winter',
                  'Similarity',
                  'Rivalry',
                  'Rage']
    Conditions = ['No_conditions']
    for condition in Conditions:
        match_c_df = deepcopy(match_df)
        data_c, T_c, Y_c = get_data(match_c_df, condition, flags)
        print(f"Ready for algorithm data with {condition} : \n{data_c}")
        # PART 4: CATE IMPLEMENTATION (Condition already applied)
        res, ps = ATTs_calculation(data_c, T_c, Y_c, flags)
        res_df = pd.DataFrame(columns=['Type', 'football_data'])

        for i in range(len(res)):
            res_df = res_df.append({'Type': str(i + 1).split('.')[0], 'football_data': res[f"{i + 1}"]},
                                   ignore_index=True)
        print(res_df)
        import pickle
        # pickle.dump(ps, open("p_scores.pkl", "wb"))

