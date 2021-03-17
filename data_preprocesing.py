import pandas as pd
###########################################
#### S Q L   P r e p r o c e s s i n g ####
###########################################

def match_data_preprocessing(match_df):
    all_cols = list(match_df.columns)
    drop_cols = ['country_id', 'date', 'home_team_api_id', 'away_team_api_id',          # match_df properties
                 'team_api_id_home', 'team_long_name_home', 'team_short_name_home',     # team_att home
                 'team_fifa_api_id_home','id_home',
                 'team_api_id_away', 'team_long_name_away','team_short_name_away',      # team_att away
                 'team_fifa_api_id_away', 'id_away' ]
    home_cols = [ 'buildUpPlaySpeed_home', 'buildUpPlaySpeedClass_home',
                  'buildUpPlayDribbling_home', 'buildUpPlayDribblingClass_home', 'buildUpPlayPassing_home',
                  'buildUpPlayPassingClass_home', 'buildUpPlayPositioningClass_home', 'chanceCreationPassing_home',
                  'chanceCreationPassingClass_home', 'chanceCreationCrossing_home', 'chanceCreationCrossingClass_home',
                  'chanceCreationShooting_home', 'chanceCreationShootingClass_home', 'chanceCreationPositioningClass_home',
                  'defencePressure_home', 'defencePressureClass_home', 'defenceAggression_home',
                  'defenceAggressionClass_home', 'defenceTeamWidth_home', 'defenceTeamWidthClass_home',
                  'defenceDefenderLineClass_home', 'team_goal_home',
                  'B365_home','B365_draw','IW_home','IW_draw','LB_home','LB_draw']
    away_cols = ['buildUpPlaySpeed_away','buildUpPlaySpeedClass_away',
                 'buildUpPlayDribbling_away', 'buildUpPlayDribblingClass_away', 'buildUpPlayPassing_away',
                 'buildUpPlayPassingClass_away', 'buildUpPlayPositioningClass_away', 'chanceCreationPassing_away',
                 'chanceCreationPassingClass_away', 'chanceCreationCrossing_away', 'chanceCreationCrossingClass_away',
                 'chanceCreationShooting_away', 'chanceCreationShootingClass_away', 'chanceCreationPositioningClass_away',
                 'defencePressure_away', 'defencePressureClass_away', 'defenceAggression_away',
                 'defenceAggressionClass_away', 'defenceTeamWidth_away', 'defenceTeamWidthClass_away',
                 'defenceDefenderLineClass_away', 'team_goal_away',
                 'B365_away','IW_away','LB_away']

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

def match_table_update(team_df : pd.DataFrame, team_att_df : pd.DataFrame, country_df : pd.DataFrame, match_df : pd.DataFrame):
    print(f"=== Match df update ===")

    pd.options.display.max_columns = 999
    pd.set_option("expand_frame_repr", True)
    pd.options.display.width = 0

    cols = list(match_df.columns)
    print(f"Default match_df columns: {cols}")
    drop_columns = ['id','league_id','season', 'stage', 'match_api_id']
    drop_odds_columns = ['BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH',
                         'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']
    odds_columns = ['B365H', 'B365D', 'B365A', 'IWH', 'IWD', 'IWA','LBH', 'LBD', 'LBA',]
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
    match_df = match_df[["country_id", "country_name", "date", "home_team_api_id", "away_team_api_id",  "home_team_goal",  "away_team_goal"] + odds_columns ]

    # odds:
    match_df[odds_columns] = match_df[odds_columns].fillna(1)
    match_df = match_df.rename(columns={"B365H": "B365_home", "B365A": "B365_away", "B365D": "B365_draw",
                                        "IWH": "IW_home", "IWA": "IW_away", "IWD": "IW_draw",
                                        "LBH": "LB_home", "LBA": "LB_away", "LBD": "LB_draw", })

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

