from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
    odds_columns = [["B365_home", "B365_away", "B365_draw"],
                    ["IW_home", "IW_away", "IW_draw"],
                    ["LB_home", "LB_away", "LB_draw"] ]
    res = {}
    for col_gr in odds_columns:
        betting_comp = col_gr[0].split('_')[0]
        res[betting_comp] = df[col_gr+['winner']].dropna(how='any',axis=0) .apply(check_odds, axis=1).mean()*100
    print(f"Odds accuracies: {res}")
    keys = ['B365','LB','IW']
    vals = [float(res[k]) for k in keys]
    comp_names = ['Bet365','Ladbrokes','Interwetten']
    sns.barplot(x=comp_names, y=vals).set_title('Betting companies accuracies')
    plt.ylim(0, 50)
    plt.yticks(range(0,50,5))
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
def win_lose_by_countries(df: pd.DataFrame):
    y1,y2,y3 = [],[],[]
    res = {}
    countries = list(df['country_name'].unique())

    def matches_num(country):
        return len(df[(df['country_name'] == country) & (df['winner'] == 'H')])

    countries.sort(key=matches_num)
    for country in countries:
        res[country] = df[df['country_name'] == country]['winner'].value_counts().to_dict()
        y1.append(res[country]['H'])
        y2.append(res[country]['D'])
        y3.append(res[country]['A'])
    plt.style.use('fivethirtyeight')
    plt.bar(np.linspace(1, 30, len(countries)), y1, width=0.5, color='b')
    plt.bar(np.linspace(1, 30, len(countries)) + 0.5 * np.ones(len(countries)), y2, width=0.5, color='green')
    plt.bar(np.linspace(1, 30, len(countries)) + 1 * np.ones(len(countries)), y3, width=0.5, color='red')
    plt.xticks(np.linspace(1, 30, len(countries)) - 0.5 * np.ones(len(countries)), res, size='small', rotation=53)
    plt.subplots_adjust(bottom=0.2, left=0.135)
    plt.ylabel('Matches Number')
    plt.title('Home Wins Advantage')
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
    plt.legend(['Home Wins', 'Draws', 'Away Wins'], loc=2)
    plt.show()
    # plt.close()
# 6)
