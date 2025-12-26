#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np





def explanatory_analysis(df):

    
    # Analysis of opponent level
    
    df['Opponent_tier'] = df['Opponent'].apply(assign_opponent_tier)
    #Explanatory analysis for the team
    print("Descriptive Analysis")
    descriptive_stats = df[['Poss', 'GF', 'GA', 'xG', 'xGA','Equipe_type']].describe()
    print(descriptive_stats.round(2))

    # Correlations
    print("Correlation with victory")
    correlation_with_win = df[['Poss', 'GF', 'GA', 'xG', 'xGA', 'Equipe_type','Victory' ]].corr()['Victory'].sort_values(ascending=False)
    print(correlation_with_win.round(2))
    
    
    #Performences with and without Key Player
    print("\n" + "-" * 40)
    
    print("With vs Without Key Players ")
    
    print("-" * 40)
    
    with_match= df[df['Equipe_type'] == 1]
    without_match= df[df['Equipe_type'] == 0]
    
    print(f"Number of matches with Key players:{len(with_match)}")
    print(f"With Key players Win Rate: {(with_match['Victory'].mean()*100):.1f}%")
    print(f"With Key players Goals by game: {(with_match['GF'].mean()):.2f}")
    print(f"With Key players Goals against by game: {(with_match['GA'].mean()):.2f}")
    
    print(f"Number of  matches without Key players:{len(without_match)}")
    print(f"Without Key players Win Rate: {(without_match['Victory'].mean()*100):.1f}%")
    print(f"Without Key players Goals by game: {(without_match['GF'].mean()):.2f}")
    print(f"Without Key players Goals against by game: {(without_match['GA'].mean()):.2f}")
    
    #Performences of Home game vs Away game
    
    print("\n" + "-" * 40)
    
    print("Home vs Away games")
    
    print("-" * 40)
    
    home_match= df[df['Venue'] == 1]
    away_match= df[df['Venue'] == 0]
    
    print(f"Number of home matches:{len(home_match)}")
    print(f"Home Win Rate: {(home_match['Victory'].mean()*100):.1f}%")
    print(f"Home Goals by game: {(home_match['GF'].mean()):.2f}")
    
    print(f"Number of Away matches:{len(away_match)}")
    print(f"Away Win Rate: {(away_match['Victory'].mean()*100):.1f}%")
    print(f"Away Goals by game: {(away_match['GF'].mean()):.2f}")
    
    
    # xG efficiency analysis
    print("\n" + "-" * 40)
    print("xG efficiency analysis")
    
    df['xG_efficiency'] = np.where(df['xG'] > 0, df['GF'] / df['xG'],0)
    df['xGA_efficiency'] = np.where(df['xGA'] > 0,  df['GA'] / df['xGA'],0)
    
    df['xG_efficiency'] = df['xG_efficiency'].round(2)
    df['xGA_efficiency'] = df['xGA_efficiency'].round(2)
    
    print(f"Overall xG Efficiency: {df['xG_efficiency'].mean():.2f}")
    print(f"Overall xGA Efficiency: {df['xGA_efficiency'].mean():.2f}")
    
    #  Key Insights
    print("\n" + "-" * 40)
    print("key of the analysis")
    print("-" * 40)
    
    strongest_predictor = correlation_with_win.index[1]  # Skip 'Victory' itself
    home_advantage = home_match['Victory'].mean() - away_match['Victory'].mean()
    
    print(f"Strongest Victory Predictor: {strongest_predictor} (r={correlation_with_win[1]:.3f})")
    print(f"Home Advantage: +{(home_advantage*100):.1f}% win rate")
    print(f"Best xG Efficiency: {df['xG_efficiency'].max():.3f}")
    
    print("\n" + "=" * 70)
    print("Analysis finished")
    print("=" * 70)
    
    result_df = df.copy()
    result_df = result_df.fillna(0).round(2)
    
    return {
        'correlation_with_win': correlation_with_win,
        'descriptive_stats': descriptive_stats,
        'processed_data': result_df }
def assign_opponent_tier(opponent_name):
    """
    Manual classification based on my knowledge about football
    Tier 1: Elite teams
    Tier 2: Good teams 
    Tier 3: Mid-table teams
    Tier 4: Relegation teams
    """
    elite_teams = ['Real Madrid', 'Atlético Madrid']
    good_teams = ['Rayo Vallecano', 'Celta Vigo', 'Betis', 'Villarreal', 'Athletic Club', 'Osasuna', 'Real Sociedad']
    mid_teams = ['Mallorca', 'Girona', 'Getafe', 'Valencia', 'Alavés', 'Sevilla', 'Espanyol']
    relegation_teams = ['Leganés', 'Las Palmas', 'Valladolid']
    
    if opponent_name in elite_teams:
        return 1
    elif opponent_name in good_teams:
        return 2
    elif opponent_name in mid_teams:
        return 3
    elif opponent_name in relegation_teams:
        return 4
    






    
    
    

