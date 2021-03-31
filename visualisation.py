from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import data_preprocesing as dpp

# 1) Odds prediction accuracies [DONE]
def check_odds(x):
    xH, xD, xA, w = x
    if(xH < xD and xH<xA and w=='H'):
        return 1
    elif(xD < xH and xD<xA and w=='D'):
        return 1
    elif(xA < xH and xA<xD and w=='A'):
        return 1
    else:
        return 0

def odds_accuracies(df: pd.DataFrame):
    # odds_columns = [['B365_home', 'B365_draw', 'B365_away'], ['IW_home', 'IW_draw', 'IW_away'], ['LB_home', 'LB_draw', 'LB_away'], ['BW_home', 'BW_draw', 'BW_away'], ['PS_home', 'PS_draw', 'PS_away'], ['WH_home', 'WH_draw', 'WH_away'], ['SJ_home', 'SJ_draw', 'SJ_away'], ['VC_home', 'VC_draw', 'VC_away'], ['GB_home', 'GB_draw', 'GB_away'], ['BS_home', 'BS_draw', 'BS_away']]
    odds_columns = [   ['B365_home', 'B365_draw', 'B365_away'],
    ['VC_home', 'VC_draw', 'VC_away'],
    ['BW_home', 'BW_draw', 'BW_away'],]


    res = {}
    for col_gr in odds_columns:
        betting_comp = col_gr[0].split('_')[0]
        res[betting_comp] = df[col_gr+['winner']].dropna(how='any',axis=0).apply(check_odds, axis=1).mean()*100
    print(f"Odds accuracies: {res}")
    vals = [float(res[k]) for k in res.keys()]
    vals.sort(reverse=True)
    comp_names = [k for k in res.keys()]
    comp_names.sort(key = lambda x: res[x], reverse=True)
    sns.barplot(x=comp_names, y=vals).set_title('Betting companies accuracies')
    plt.ylim(50, 55)
    plt.yticks(range(50,55,1))
    plt.ylabel('Winner prediction accuracy in %')
    plt.show()
    plt.close()

# 2) Teams parameters distribution by countries
def params_by_countries(df: pd.DataFrame):
    pass

# 3) Losses, Draws, Wins at home [DONE]
def win_lose_at_home(df : pd.DataFrame ):
    res = df['winner'].value_counts().to_dict()
    print(f"Wins distribtions: {res}")
    keys = ['Home','Draws', 'Away']
    vals = [float(res[k]/len(df)*100) for k in ['H','D','A']]
    sns.barplot(x=keys, y=vals).set_title('Wins distribtions')
    plt.ylabel('Matches results distribution in %')
    plt.show()
    plt.close()

# 4) Losses, Draws, Wins by country [DONE]
def win_lose_by_countries(df: pd.DataFrame, condition = 'No_conditions'):
    y1,y2,y3 = [],[],[]
    res = {}
    countries = list(df['country_name'].unique())

    def matches_num(country):
        return len(df[(df['country_name'] == country) & (df['winner'] == 'H')])

    countries.sort(key=matches_num)
    for country in countries:
        res[country] = df[df['country_name'] == country]['winner'].value_counts().to_dict()
        y1.append(res[country]['H'])
        if('D' in res[country].keys()):
            y2.append(res[country]['D'])
        else:
            y2.append(0)
        y3.append(res[country]['A'])
    plt.style.use('fivethirtyeight')
    plt.bar(np.linspace(1, 30, len(countries)), y1, width=0.5, color='b')
    plt.bar(np.linspace(1, 30, len(countries)) + 0.5 * np.ones(len(countries)), y2, width=0.5, color='green')
    plt.bar(np.linspace(1, 30, len(countries)) + 1 * np.ones(len(countries)), y3, width=0.5, color='red')
    plt.xticks(np.linspace(1, 30, len(countries)) - 0.5 * np.ones(len(countries)), res, size='small', rotation=53)
    plt.subplots_adjust(bottom=0.2, left=0.135)
    plt.ylabel('Matches Number')
    if(condition == 'No_conditions'):
        plt.title('Home Wins Advantage')
    else:
        plt.title(f'Home Wins Advantage with {condition}')
    plt.legend(['Home Wins', 'Draws', 'Away Wins'], loc=2)
    plt.show()
    plt.close()

# 5) Goals distribution by countries (Maybe goal diffs) [DONE]
def goals_distribution_by_countries(df : pd.DataFrame):
    y1, y2 = [], []
    res = {}
    countries = list(df['country_name'].unique())

    def matches_num(country):
        return df[(df['country_name'] == country)]['team_goal_home'].mean()

    countries.sort(key=matches_num)
    for country in countries:
        res[country] = df[(df['country_name'] == country)][['team_goal_home','team_goal_away']].mean().to_dict()
        y1.append(res[country]['team_goal_home'])
        y2.append(res[country]['team_goal_away'])
    plt.style.use('fivethirtyeight')
    plt.bar(np.linspace(1, 20, len(countries)), y1, width=0.5, color='b')
    plt.bar(np.linspace(1, 20, len(countries)) + 0.5 * np.ones(len(countries)), y2, width=0.5, color='red')
    plt.xticks(np.linspace(1, 20, len(countries))  - 0.3 * np.ones(len(countries)), res, size='small', rotation=53)
    plt.subplots_adjust(bottom=0.2, left=0.135)
    plt.ylabel('Goals number expectation')
    plt.title('Home goals Advantage')
    plt.legend(['Home goals', 'Away goals'], loc=2)
    plt.show()
    # plt.close()

def visualise_all(match_df_gappy_odds : pd.DataFrame, flags : dict):
    if flags['GraphsFlag']:
        match_df_gappy_odds = dpp.add_winner_column(match_df_gappy_odds)
        # Possible graphs:
        # 1) Odds prediction accuracies [DONE]
        odds_accuracies(match_df_gappy_odds)
        # 2) Teams parameters distribution by countries [TODO]
        params_by_countries(match_df_gappy_odds)
        # 3) Losses, Drawns, Wins at home [DONE]
        win_lose_at_home(match_df_gappy_odds)
        # 4) Losses, Drawns, Wins by country [DONE]
        win_lose_by_countries(match_df_gappy_odds)
        # 5) Goals distribution by countries (Maybe goal diffs) [DONE]
        goals_distribution_by_countries(match_df_gappy_odds)
        match_df_gappy_odds = match_df_gappy_odds.drop('winner', axis = 1)

