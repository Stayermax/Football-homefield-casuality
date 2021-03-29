import pandas as pd
from math import sqrt
import os
from copy import deepcopy
###########################################
#### S Q L   P r e p r o c e s s i n g ####
###########################################

def Weather_condition(date):
    month = date.month
    if(month>11 or month<3):
        return 1
    return 0

def Similarity_condition(teams_params):
    data = teams_params.to_dict()
    params = []
    R_2 = 0
    for key in data.keys():
        if('away' in key):
            params.append(key.split('_')[0])
    for param in params:
        R_2 += (teams_params[param + '_away'] - teams_params[param + '_home'])**2
    return sqrt(R_2)

def Tournament_position(match_df : pd.DataFrame, loadFlag):
    data_path = f"preprocessed_data/Conditions_data/tournament_positions.csv"
    if (os.path.exists(data_path) and loadFlag):
        res_df = pd.read_csv(data_path, index_col=0)
    else:
        game_cols = ['date', 'league_id', 'season', 'stage', 'team_api_id_home', 'team_api_id_away', 'team_goal_home', 'team_goal_away']
        df = match_df[game_cols]

        pos_cols = ['pos_home', 'pos_away']
        res_df = pd.DataFrame(columns = pos_cols)

        LS_grps = df.groupby(['league_id','season'])
        for LS_name, LS_group in LS_grps:
            team_scores = {id : 0 for id in LS_group['team_api_id_home'].unique()}
            Stage_grps = LS_group.groupby('stage')
            # print(f'GROUP: {LS_name} with {len(team_scores.keys())} teams and {len(Stage_grps)} stages')
            for stage_name, stage_group in Stage_grps:
                for index, row in stage_group.iterrows():
                    if(row['team_goal_home'] > row['team_goal_away']):
                        team_scores[row['team_api_id_home']] += 3
                    elif(row['team_goal_home'] < row['team_goal_away']):
                        team_scores[row['team_api_id_away']] += 3
                    else:
                        team_scores[row['team_api_id_home']] += 1
                        team_scores[row['team_api_id_away']] += 1
                sorted_teams_ids = sorted(team_scores.items(), key=lambda item: item[1], reverse=True)
                # print(sorted_teams_ids)
                sorted_teams_ids = {el[0] : ind for ind, el in enumerate(sorted_teams_ids)}
                for index, row in stage_group.iterrows():
                    home_position = sorted_teams_ids[row['team_api_id_home']]
                    away_position = sorted_teams_ids[row['team_api_id_away']]
                    res_df.loc[index] = [home_position, away_position]
        res_df.to_csv(data_path)
    return res_df

def Rivalry_condition(game_data):
    data = game_data.to_dict()
    if(abs(data['pos_home']-data['pos_away'])<3):
        return 1
    return 0

def Rage_calculation(match_df : pd.DataFrame, loadFlag):
    data_path = f"preprocessed_data/Conditions_data/rage_data.csv"
    if (os.path.exists(data_path) and loadFlag):
        res_df = pd.read_csv(data_path, index_col=0)
    else:
        game_cols = ['date', 'league_id', 'season', 'stage', 'team_api_id_home', 'team_api_id_away', 'team_goal_home', 'team_goal_away']
        df = match_df[game_cols]

        res_df = pd.DataFrame(columns = ['Rage'])
        LS_grps = df.groupby(['league_id','season'])
        for LS_name, LS_group in LS_grps:
            # lost_at_away_field_teams dict: team_id : list of team_ids
            # For each team x list of teams y that won at their home field in match with x
            lost_at_away_field_teams= {id : [] for id in LS_group['team_api_id_home'].unique()}
            Stage_grps = LS_group.groupby('stage')
            # print(f'GROUP: {LS_name} with {len(team_scores.keys())} teams and {len(Stage_grps)} stages')
            for stage_name, stage_group in Stage_grps:
                for index, row in stage_group.iterrows():
                    if(row['team_api_id_away'] in lost_at_away_field_teams[row['team_api_id_home']]):
                        res_df.loc[index] = [1]
                    else:
                        res_df.loc[index] = [0]
                    if(row['team_goal_home'] >= row['team_goal_away']):
                        lost_at_away_field_teams[row['team_api_id_away']].append(row['team_api_id_home'])
                    else:
                        pass
        res_df.to_csv(data_path)
    return res_df

def match_data_preprocessing(match_df, condition, loadFlag, conditionGraphFlag):
    all_cols = list(match_df.columns)
    print(all_cols)
    drop_cols = ['league_id','season', 'stage',
                 'country_id', 'date', 'home_team_api_id', 'away_team_api_id', 'team_api_id_home',
                 'team_long_name_home', 'team_short_name_home', 'team_fifa_api_id_home', 'id_home', 'team_api_id_away',
                 'team_long_name_away', 'team_short_name_away', 'team_fifa_api_id_away', 'id_away']
    home_cols = [ 'buildUpPlaySpeedClass_home',
                  'buildUpPlayDribblingClass_home',
                  'buildUpPlayPassingClass_home', 'buildUpPlayPositioningClass_home',
                  'chanceCreationPassingClass_home', 'chanceCreationCrossingClass_home',
                  'chanceCreationShootingClass_home', 'chanceCreationPositioningClass_home',
                  'defencePressureClass_home',
                  'defenceAggressionClass_home', 'defenceTeamWidthClass_home',
                  'defenceDefenderLineClass_home', 'team_goal_home',
                  'B365_home','B365_draw','VC_home','VC_draw','BW_home','BW_draw']
    home_num_cols = ['buildUpPlaySpeed_home', 'buildUpPlayDribbling_home', 'buildUpPlayPassing_home',
                     'chanceCreationPassing_home', 'chanceCreationCrossing_home', 'chanceCreationShooting_home',
                     'defencePressure_home', 'defenceAggression_home', 'defenceTeamWidth_home']

    away_cols = ['buildUpPlaySpeedClass_away',
                 'buildUpPlayDribblingClass_away',
                 'buildUpPlayPassingClass_away', 'buildUpPlayPositioningClass_away',
                 'chanceCreationPassingClass_away', 'chanceCreationCrossingClass_away',
                 'chanceCreationShootingClass_away', 'chanceCreationPositioningClass_away',
                 'defencePressureClass_away',
                 'defenceAggressionClass_away', 'defenceTeamWidthClass_away',
                 'defenceDefenderLineClass_away', 'team_goal_away',
                 'B365_away','VC_away','BW_away']

    away_num_cols = ['buildUpPlaySpeed_away', 'buildUpPlayDribbling_away', 'buildUpPlayPassing_away',
                     'chanceCreationPassing_away', 'chanceCreationCrossing_away', 'chanceCreationShooting_away',
                     'defencePressure_away', 'defenceAggression_away', 'defenceTeamWidth_away']

    if(condition == "No_conditions"):
        pass
    elif(condition == 'LowStage'):
        match_df = match_df[match_df['stage'] <10]
    elif(condition == 'HighStage'):
        match_df = match_df[match_df['stage'] > 20]
    elif(condition == "Weather"):
        # match_df['date']
        match_df['Weather'] = match_df['date'].apply(Weather_condition)
        match_df = match_df[match_df['Weather'] == 1]
        match_df = match_df.drop('Weather', axis=1)
    elif(condition == "Similarity"):
        match_df['Similarity'] = match_df[home_num_cols + away_num_cols].apply(Similarity_condition, axis=1)
        # threshold = match_df['Similarity'].mean()
        threshold = 20
        print(f"Similarity df: {match_df}")
        match_df = match_df[match_df['Similarity'] <= threshold]
        match_df = match_df.drop('Similarity',axis=1)
        print(f"Reduced similarity df: {match_df}")
    elif(condition == "Rivalry"):
        # todo: check stages number
        # todo: check 'id_away' column - what is it?
        match_df.sort_values(by='date')
        tournament_df = Tournament_position(match_df, loadFlag)
        match_df = pd.merge(match_df, tournament_df, left_index=True, right_index=True)
        match_df['Rivalry'] = match_df[['pos_home','pos_away']].apply(Rivalry_condition, axis=1)
        # print(f"Rivalry df: {match_df}")
        match_df = match_df[match_df['Rivalry'] == 1]
        match_df = match_df.drop(['Rivalry','pos_home','pos_away'], axis=1)
    elif(condition == "Rage"):
        match_df.sort_values(by='date')
        rage_df = Rage_calculation(match_df, loadFlag)
        match_df = pd.merge(match_df, rage_df, left_index=True, right_index=True)
        # print(f"Rage df: {match_df}")
        match_df = match_df[match_df['Rage'] == 1]
        match_df = match_df.drop('Rage', axis=1)

    if(conditionGraphFlag):
        import visualisation as vis
        match_df_winner = deepcopy(match_df)
        match_df_winner = add_winner_column(match_df_winner)
        vis.win_lose_by_countries(match_df_winner, condition)

    match_df = match_df.drop(drop_cols, axis=1)

    home_data = match_df.drop('team_goal_away', axis=1)
    home_rename = {}
    for col in home_data.columns:
        if(col == "team_goal_home"):
            home_rename["team_goal_home"] = "Y"
        elif("draw" in col):
            pass
        else:
            name, place = col.split("_")
            if(place =="away"):
                home_rename[col] = name + "_2"
            else:
                home_rename[col] = name + "_1"
    home_data = home_data.rename(columns=home_rename)
    home_data["T"] = 1

    away_data = match_df.drop('team_goal_home', axis=1)
    away_rename = {}
    for col in away_data.columns:
        if (col == "team_goal_away"):
            away_rename["team_goal_away"] = "Y"
        elif("draw" in col):
            pass
        else:
            name, place = col.split("_")
            if (place == "home"):
                away_rename[col] = name + "_2"
            else:
                away_rename[col] = name + "_1"
    away_data = away_data.rename(columns=away_rename)
    away_data['T'] = 0
    data = home_data.append(away_data, ignore_index=True)
    data = cat_to_num(data)

    return data


def team_table_update(team_df : pd.DataFrame ):
    print(f"=== Team df update ===")
    drop_columns = ['id']
    team_df = team_df.drop(drop_columns, axis=1)
    team_df = team_df.fillna(-1)
    team_df['team_fifa_api_id'] = team_df['team_fifa_api_id'].astype(int)
    return team_df

def cat_val_filler(x, mean_val):
    print(x)
    if(x.isnull().values.all()):
        return mean_val
    else:
        return x.value_counts().idxmax()

def team_att_table_update(team_df : pd.DataFrame, team_att_df: pd.DataFrame):
    print(f"=== Team attributes df update ===")

    # Part 1: get average data of the team_att_df and save it into mean_data dictionary
    pd.options.display.max_columns = 999
    pd.set_option("expand_frame_repr", True)
    pd.set_option('display.max_columns', None)
    pd.set_option('max_columns', None)
    pd.options.display.width = 0
    # Mean data search:
    team_att_df['date'] = pd.to_datetime(team_att_df['date'], format='%Y-%m-%d %H:%M:%S')
    cat_mean_data = {'buildUpPlaySpeedClass':'Balanced',
                 'buildUpPlayDribblingClass': 'Normal',
                 'buildUpPlayPassingClass': 'Mixed',
                 'buildUpPlayPositioningClass': 'Free Form',
                 'chanceCreationPassingClass': 'Normal',
                 'chanceCreationCrossingClass': 'Normal',
                 'chanceCreationShootingClass': 'Normal',
                 'chanceCreationPositioningClass': 'Free Form',
                 'defencePressureClass': 'Medium',
                 'defenceAggressionClass': 'Press',
                 'defenceTeamWidthClass': 'Normal',
                 'defenceDefenderLineClass': 'Cover',
                 }
    cat_variables = cat_mean_data.keys()
    mean_df = team_att_df.drop(['id', 'team_fifa_api_id', 'team_api_id', 'date'], axis=1)

    mean_df = mean_df.drop(cat_mean_data.keys(), axis=1)
    num_mean_data = mean_df.mean().astype(int).to_dict()
    num_variables = num_mean_data.keys()
    cat_mean_data.update(num_mean_data)
    mean_data = cat_mean_data # mean of all teams attributes

    # Part 2: Fill table with attributes of the same team by closest date
    for num_var in num_variables:
        slice_df = team_att_df[['id', 'date','team_api_id',num_var]]
        s = (pd.merge_asof(
            slice_df.sort_values('date').reset_index(),  # Full Data Frame
            slice_df.sort_values('date').dropna(subset=[num_var]),  # Subset with valid scores
            by='team_api_id',  # Only within `'cn'` group
            on='date', direction='nearest'  # Match closest date
        )
             .set_index('index').sort_index())
        team_att_df[num_var] = team_att_df[num_var].fillna(s[f"{num_var}_y"])

    for cat_var in cat_variables:
        slice_df = team_att_df[['id', 'date', 'team_api_id', cat_var]]
        s = (pd.merge_asof(
            slice_df.sort_values('date').reset_index(),  # Full Data Frame
            slice_df.sort_values('date').dropna(subset=[cat_var]),  # Subset with valid scores
            by='team_api_id',  # Only within `'cn'` group
            on='date', direction='nearest'  # Match closest date
        )
             .set_index('index').sort_index())
        team_att_df[cat_var] = team_att_df[cat_var].fillna(s[f"{cat_var}_y"])

    # fill the rest of the empty fields with average data of the table
    team_att_df = team_att_df.fillna(mean_data)

    # Part 3: Check that all teams are in this table:
    print(team_df.head())
    all_teams = set(team_df['team_api_id'].unique())
    att_teams = set(team_att_df['team_api_id'].unique())

    lost_teams_ids = list(all_teams-att_teams)
    max_fifa_id = max(team_att_df['team_fifa_api_id'])+1
    max_id = max(team_att_df['id'])+1
    print(f'Lost teams: {lost_teams_ids}')
    for i, id in enumerate(lost_teams_ids):
        team_dict = {'id': max_id + i, 'team_fifa_api_id': max_fifa_id + i, 'team_api_id': id,
                     'date': pd.Timestamp('2000-01-01 00:00:00'), }
        team_dict.update(mean_data)
        team_att_df = team_att_df.append(team_dict, ignore_index=True)
    # for i, id in enumerate(att_teams):
    #     team_dict = {'id': id, 'team_fifa_api_id': max_fifa_id + i, 'team_api_id': id,
    #                  'date': pd.Timestamp('2000-01-01 00:00:00'), }


    team_df = team_df.drop('team_fifa_api_id',axis=1).reset_index()
    team_df = team_df.merge(team_att_df[['team_api_id','team_fifa_api_id']], on='team_api_id', how='left').drop_duplicates().set_index('index')


    team_att_df = team_df.merge(team_att_df.drop(['team_fifa_api_id'],axis=1),on = 'team_api_id',how='outer')

    # print("Final team attributes table:")
    # print(team_att_df[team_att_df['team_api_id']==188163].head(10))
    # print(team_att_df.head(10))

    return team_df, team_att_df

def match_table_fill_odds(match_df):
    odds_columns = ["B365_home", "B365_away", "B365_draw",
                    "VC_home", "VC_away", "VC_draw",
                    "BW_home", "BW_away",  "BW_draw", ]

    match_df[odds_columns] = match_df[odds_columns].fillna(1)
    return match_df

def match_table_update(team_df : pd.DataFrame, team_att_df : pd.DataFrame, country_df : pd.DataFrame, match_df : pd.DataFrame):
    print(f"=== Match df update ===")

    pd.options.display.max_columns = 999
    pd.set_option("expand_frame_repr", True)
    pd.options.display.width = 0

    cols = list(match_df.columns)
    print(f"Default match_df columns: {cols}")

    tournament_columns = ['league_id','season', 'stage']
    drop_columns = ['id','match_api_id']
    drop_odds_columns = ['IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH',
                         'WHD', 'WHA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA', 'SJH', 'SJD', 'SJA']
    odds_columns = ['B365H', 'B365D', 'B365A', 'VCH', 'VCD', 'VCA','BWH', 'BWD', 'BWA']
    # B365D
    # BWD
    # VCD
    zero_columns = ['goal', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'possession']
    player_columns = ['home_player_X1', 'home_player_X2', 'home_player_X3', 'home_player_X4', 'home_player_X5',
                      'home_player_X6', 'home_player_X7', 'home_player_X8', 'home_player_X9', 'home_player_X10',
                      'home_player_X11', 'away_player_X1', 'away_player_X2', 'away_player_X3', 'away_player_X4',
                      'away_player_X5', 'away_player_X6', 'away_player_X7', 'away_player_X8', 'away_player_X9',
                      'away_player_X10', 'away_player_X11', 'home_player_Y1', 'home_player_Y2', 'home_player_Y3',
                      'home_player_Y4', 'home_player_Y5', 'home_player_Y6', 'home_player_Y7', 'home_player_Y8',
                      'home_player_Y9', 'home_player_Y10', 'home_player_Y11', 'away_player_Y1', 'away_player_Y2',
                      'away_player_Y3', 'away_player_Y4', 'away_player_Y5', 'away_player_Y6', 'away_player_Y7',
                      'away_player_Y8', 'away_player_Y9', 'away_player_Y10', 'away_player_Y11', 'home_player_1',
                      'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6',
                      'home_player_7', 'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11',
                      'away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5',
                      'away_player_6', 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10',
                      'away_player_11']

    # draw_columns = [el for el in drop_odds_columns + odds_columns if 'D' in el]
    # print(f"EMPTY VALUES: {match_df[draw_columns].isna().sum()}")

    # unnecessary
    match_df = match_df.drop(drop_odds_columns, axis=1)
    match_df = match_df.drop(drop_columns, axis=1)
    match_df = match_df.drop(zero_columns, axis=1)
    match_df = match_df.drop(player_columns, axis=1)

    # date
    match_df['date'] = pd.to_datetime(match_df['date'], format='%Y-%m-%d %H:%M:%S')

    # Countries
    match_df = match_df.merge(country_df, left_on='country_id', right_on='id', how='left')
    match_df = match_df.drop('id', axis=1)
    match_df = match_df.rename(columns={"name": "country_name"})
    match_df = match_df[tournament_columns + ["country_id", "country_name", "date", "home_team_api_id", "away_team_api_id",  "home_team_goal",  "away_team_goal"] + odds_columns ]

    # odds:

    match_df = match_df.rename(columns={"B365H": "B365_home", "B365A": "B365_away", "B365D": "B365_draw",
                                        'VCH': 'VC_home', 'VCD': 'VC_draw', 'VCA': 'VC_away',
                                        'BWH': 'BW_home', 'BWD': 'BW_draw', 'BWA': 'BW_away',
                                        })


    # cols = list(match_df.columns)
    # print(f"Current match_df columns: {cols}")
    # print(match_df.head(10))

    home_team_df = pd.merge_asof(match_df.sort_values('date'), team_att_df.sort_values('date'),
                        left_by='home_team_api_id', right_by='team_api_id', on='date',
                        direction='backward')
    home_team_df = home_team_df.fillna(pd.merge_asof(match_df.sort_values('date'), team_att_df.sort_values('date'),
                        left_by='home_team_api_id', right_by='team_api_id', on='date',
                        direction='forward'))

    home_and_away_df = pd.merge_asof(home_team_df, team_att_df.sort_values('date'),
                        left_by='away_team_api_id', right_by='team_api_id', on='date', suffixes=('_home','_away'),
                        direction='backward')
    home_and_away_df = home_and_away_df.fillna(pd.merge_asof(home_team_df, team_att_df.sort_values('date'),
                        left_by='away_team_api_id', right_by='team_api_id', on='date', suffixes=('_home','_away'),
                        direction='forward'))

    home_and_away_df = home_and_away_df.rename(columns={"home_team_goal": "team_goal_home", "away_team_goal": "team_goal_away", "name":"country_name"})

    Lost_teams = [4064, 188163, 9765, 10213, 6601, 7947, 177361, 7992, 4049, 7896, 6367]
    print(team_df[team_df['team_api_id'].isin(Lost_teams)]['team_long_name'].unique())
    lost_df = home_and_away_df[(home_and_away_df['home_team_api_id'].isin(Lost_teams)) | (home_and_away_df['away_team_api_id'].isin(Lost_teams)) ]
    print(f'LOST DF: {lost_df}')
    home_and_away_df = home_and_away_df.drop(lost_df.index, axis=0)


    return home_and_away_df

###########################################
######## P r e p r o c e s s i n g ########
###########################################

def data_preprocessing(df):
    df = df.drop('Unnamed: 0', axis=1)
    df = cat_to_num(df)
    T = df['T']
    Y = df['Y']
    data = df.drop(['T', 'Y'], axis=1)
    return data, T, Y

def cat_to_num(df):
    cat_cols = []
    for c_name in df.columns:
        if (type(df[c_name][0]) is str):
            cat_cols.append(c_name)

    from sklearn.preprocessing import OneHotEncoder
    for cat_col in cat_cols:
        oe_style = OneHotEncoder()
        oe_results = oe_style.fit_transform(df[[cat_col]])
        columns = [cat_col + str(cat) for cat in oe_style.categories_[0]]
        df = df.join(pd.DataFrame(oe_results.toarray(), columns=columns))
    df = df.drop(cat_cols, axis=1)
    return df

def scale(df):
    from sklearn.preprocessing import MaxAbsScaler
    transformer = MaxAbsScaler().fit(df)
    df = transformer.transform(df)
    return df

def H_D_A(x):
    if(x[0]>x[1]):
        return 'H'
    elif(x[0]<x[1]):
        return 'A'
    else:
        return 'D'

def add_winner_column(df):
    df['winner'] = df[['team_goal_home','team_goal_away']].apply(H_D_A, axis=1)
    return df






