import pandas as pd
import numpy as np





    
def create_advanced_measures(players_df):
    # Creating some other features for the players
    
    # CORR: To be sure that everything is correct we convert comas into points
    numeric_cols_to_convert = ['xG', 'PrgC', 'PrgP', 'Pass', 'Shots', 'Take_On', 'Tkl', 'TklW', 'Int', 'Recov', 
                               'Min', 'MP', 'Starts', 'Succes_P%', 'Sot%', 'TO%','Defduel', 'Defduel%']
    
    for col in numeric_cols_to_convert:
        if col in players_df.columns and players_df[col].dtype == 'object':
            players_df[col] = players_df[col].astype(str).str.replace(',', '.').astype(float)
    
    

    # measures about stamina        
    players_df['Stamina']=(players_df['Min']/ players_df['MP'])
    players_df['Availability_Rate'] = (players_df['Min'] / (players_df['MP'] * 90)).round(2)
    
    
    
    # Per 90 minutes statistics (essential for comparison specially for football because a game has a duration of 90 mins)
    players_df['Goals_p90'] = (players_df['Goals'] / players_df['Min']) * 90
    players_df['Assists_p90'] = (players_df['Assists'] / players_df['Min']) * 90
    players_df['xG_p90'] = (players_df['xG'] / players_df['Min']) * 90
    players_df['Shots_p90'] = (players_df['Shots'] / players_df['Min']) * 90
    players_df['Shots_on_target_p90'] = (players_df['Shots'] * players_df['Sot%']  / players_df['Min']) * 90 /100
    
    # Offensive Efficiency Measures
    players_df['Conversion_Rate'] = (players_df['Goals'] / players_df['Shots'] * 100).round(1)
    players_df['Goal_Contribution_p90'] = players_df['Goals_p90'] + players_df['Assists_p90']
    players_df['xG_efficiency'] = (players_df['Goals'] / players_df['xG']).round(2)
    
    # Creative impact Measures
    players_df['Progressive_Passes_p90'] = (players_df['PrgP'] / players_df['Min']) * 90
    players_df['Progressive_Carries_p90'] = (players_df['PrgC'] / players_df['Min']) * 90
    players_df['Total_Progressive_Actions_p90'] = players_df['Progressive_Passes_p90'] + players_df['Progressive_Carries_p90']
    
    # Involvement measures
    players_df['Passes_p90'] = (players_df['Pass'] / players_df['Min']) * 90
    players_df['Successful_Passes_p90'] = (players_df['Pass'] * players_df['Succes_P%']  / players_df['Min']) * 90 / 100
    
    # Defensive measures 
    players_df['Tackles_p90'] = (players_df['Tkl'] / players_df['Min']) * 90
    players_df['Successful_Tackles_p90'] = (players_df['TklW'] / players_df['Min']) * 90
    players_df['Interceptions_p90'] = (players_df['Int'] / players_df['Min']) * 90
    players_df['Ball_Recoveries_p90'] = (players_df['Recov'] / players_df['Min']) * 90
    players_df['Def_Duels_Won'] = ((players_df['Defduel'] * players_df['Defduel%']) / 100).round()
    players_df['Def_Duels_Won_p90'] = (players_df['Def_Duels_Won'] /  players_df['Min']) * 90
    
    
    
    # Dribbling and 1v1 measures
    players_df['Take_Ons_p90'] = (players_df['Take_On'] / players_df['Min']) * 90
    players_df['Successful_Take_Ons_p90'] = (players_df['Take_On'] * players_df['TO%']  / players_df['Min']) * 90 /100
    
    # Some measures that are interistics for clustering
    players_df['Versatility_index'] = (players_df['Goal_Contribution_p90'] +           players_df['Total_Progressive_Actions_p90']+players_df['Tackles_p90']).round(2)
    
    players_df['Attack_Defense_Ratio'] = np.where(players_df['Tackles_p90'] > 0, players_df['Goal_Contribution_p90'] / players_df['Tackles_p90'], 0).round(2)
                                                  
    # Player role classification
    players_df['Player_Role'] = players_df.apply(classify_player_role, axis=1)
    
    # Overall impact score (composite metric)
    players_df['Overall_Impact_Score'] = players_df.apply(calculate_impact_score, axis=1)
    
    """ I did add this safety because of the function: final_data_formatting for getting rid of the errors and convertir number into float"""
    
    new_metrics = ['Goals_p90', 'Assists_p90', 'xG_p90', 'Shots_p90', 'Shots_on_target_p90',
                   'Conversion_Rate', 'Goal_Contribution_p90', 'xG_efficiency',
                   'Progressive_Passes_p90', 'Progressive_Carries_p90', 'Total_Progressive_Actions_p90',
                   'Passes_p90', 'Successful_Passes_p90', 'Tackles_p90', 'Successful_Tackles_p90',
                   'Interceptions_p90', 'Ball_Recoveries_p90', 'Take_Ons_p90', 'Successful_Take_Ons_p90',
                   'Versatility_Index', 'Attack_Defense_Ratio', 'Overall_Impact_Score', 'Def_Duels_Won',
                   'Def_Duels_Won_p90','Availability_Rate', 'Stamina']
    for metric in new_metrics:
        if metric in players_df.columns:
            players_df[metric] = pd.to_numeric(players_df[metric], errors='coerce').fillna(0)
    
    print("Advanced metrics creation completed")
    
    return players_df

def classify_player_role(row):
    #Classifying the different players role into the team
    
    if row['Pos'] == 'DF':
        return 'Defender'
    elif row['Pos'] == 'MF':
        if row['Assists_p90'] > row['Goals_p90']:
            return 'Creative Midfielder'
        else:
            return 'Box-to-Box Midfielder'
    elif row['Pos'] == 'FW':
        if row['Goals_p90'] > 0.4:
            return 'Goalscorer'
        elif row['Assists_p90'] > 0.3:
            return 'Playmaker'
        else:
            return 'Winger'
    return 'Utility Player'

def calculate_impact_score(row):
    """Calculate impact score for a single player based on their role and like a Fifa card with a score """
    
    # Normalize all measures to 0-1 scale for having a more objective impact scores
    max_values = {
        'Goals_p90': 0.7,
        'Sot%': 50.0,
        'Conversion_Rate': 25.0,
        'xG_efficiency': 2.0,
        'Successful_Take_Ons_p90': 6.0,
        'Progressive_Carries_p90': 8.0,
        'Successful_Passes_p90': 80.0,
        'Assists_p90': 0.8,
        'Progressive_Passes_p90': 15.0,
        'Defduel%': 80.0,
        'Interceptions_p90': 2.0,
        'Successful_Tackles_p90': 2.0,
        'Ball_Recoveries_p90': 8.0,
        'Availability_Rate': 1.0
    }
    
    # Calculate normalized dimension scores
    offensive_score = (
        min(row['Goals_p90'] / max_values['Goals_p90'], 1.0) * 0.5 +
        min(row['Sot%'] / max_values['Sot%'], 1.0) * 0.1 +
        min(row['Conversion_Rate'] / max_values['Conversion_Rate'], 1.0) * 0.2 +
        min(row['xG_efficiency'] / max_values['xG_efficiency'], 1.0) * 0.2)
    
    dribble_score = (
        min(row['Successful_Take_Ons_p90'] / max_values['Successful_Take_Ons_p90'], 1.0) * 0.5 +
        min(row['Progressive_Carries_p90'] / max_values['Progressive_Carries_p90'], 1.0) * 0.5)
    
    passing_score = (
        min(row['Successful_Passes_p90'] / max_values['Successful_Passes_p90'], 1.0) * 0.3 +
        min(row['Assists_p90'] / max_values['Assists_p90'], 1.0) * 0.4 +
        min(row['Progressive_Passes_p90'] / max_values['Progressive_Passes_p90'], 1.0) * 0.3)
    
    defensive_score = (
        min(row['Defduel%'] / max_values['Defduel%'], 1.0) * 0.4 +
        min(row['Interceptions_p90'] / max_values['Interceptions_p90'], 1.0) * 0.2 +
        min(row['Successful_Tackles_p90'] / max_values['Successful_Tackles_p90'], 1.0) * 0.3 +
        min(row['Ball_Recoveries_p90'] / max_values['Ball_Recoveries_p90'], 1.0) * 0.1)
            
    
    """"Role-specific weights like in fifa because you can't judge a defender and a forward on the same basis of criteria and I
    just kept the roles of goalscorer, playmaker, creative midfielder and defender because they are the output of the precedent
    function that I seen in my precedent files"""
        
    player_role = row['Player_Role']
        
    if player_role == 'Goalscorer':
        weights = [0.4, 0.2, 0.2, 0.1, 0.1]
    elif player_role == 'Playmaker':
        weights = [0.3, 0.4, 0.3, 0.1, 0.1]
    elif player_role == 'Creative Midfielder':
        weights = [0.05, 0.3, 0.3, 0.2, 0.15]
    elif player_role == 'Defender':
        weights = [0.05, 0.15, 0.3, 0.3, 0.2]
    else:
            weights = [0.25, 0.25, 0.2, 0.2, 0.1]
            
    dimension_scores = [offensive_score, dribble_score, passing_score ,defensive_score,row['Availability_Rate']]
            
    weighted_score = sum(score * weight for score, weight in zip(dimension_scores, weights))
        
    result = round(weighted_score * 100 , 1)
        
    
    return result






def final_data_formatting(players_df):
    #Final formatting and cleaning
    
    # Round numeric columns for readability
    rounding_rules = {
        'Goals_p90': 2,
        'Assists_p90': 2,
        'xG_p90': 2,
        'Shots_p90': 2,
        'Shots_on_target_p90': 2,
        'Conversion_Rate': 2,
        'Goal_Contribution_p90': 2,
        'xG_efficiency': 2,
        'Progressive_Passes_p90': 2,
        'Progressive_Carries_p90': 2,
        'Total_Progressive_Actions_p90': 2,
        'Passes_p90': 2,
        'Successful_Passes_p90': 2,
        'Tackles_p90': 2,
        'Successful_Tackles_p90': 2,
        'Interceptions_p90': 2,
        'Ball_Recoveries_p90': 2,
        'Take_Ons_p90': 2,
        'Successful_Take_Ons_p90': 2,
        'Def_Duels_Won': 0,           
        'Def_Duels_Won_p90': 2,
        'Versality_index': 2,
        'Attack_Defense_Ratio': 2,
        'Stamina': 2,
        'Availability_Rate': 2,
        'Overall_Impact_Score': 1     
    }
    
    for col, decimals in rounding_rules.items():
        if col in players_df.columns:
            players_df[col] = players_df[col].round(decimals)
    
    # Handle in case of infinite values from division
    players_df['xG_efficiency'] = players_df['xG_efficiency'].replace([np.inf, -np.inf], 0)
    
    # Sort by position and impact score
    players_df = players_df.sort_values(['Pos', 'Overall_Impact_Score'], ascending=[True, False])
    
    # Reset index for clean output
    players_df = players_df.reset_index(drop=True)
    
    return players_df


def get_player_statistics_summary(players_df):
    """Print  summary without returning data because it wasn't beautiful with the returning of data"""
    
    df_display = players_df.copy()
    
    print("\n" + "="*70)
    print("Player analysis report")
    print("="*70)
    
    print(f"\nRoster: {len(df_display)} players analyzed ")
    print(f"Positions: {df_display['Pos'].value_counts().to_dict()}")
    print(f"Roles: {df_display['Player_Role'].value_counts().to_dict()}")
    
    print("\nAvg performances by positions:")
    position_stats = df_display.groupby('Pos').agg({
        'Goals_p90': 'mean', 'Assists_p90': 'mean', 'Overall_Impact_Score': 'mean'
    }).round(2)
    print(position_stats.to_string())
    
    print("\nTop performers:")
    print(f"   - Most Goals: {df_display.loc[df_display['Goals_p90'].idxmax()]['Players']}")
    print(f"   - Most Assists: {df_display.loc[df_display['Assists_p90'].idxmax()]['Players']}")
    print(f"   - Most Creative: {df_display.loc[df_display['Progressive_Passes_p90'].idxmax()]['Players']}")
    print(f"   - Best Defender: {df_display.loc[df_display['Defduel%'].idxmax()]['Players']}")
    print(f"   - Highest Impact: {df_display.loc[df_display['Overall_Impact_Score'].idxmax()]['Players']}")
    
    print("\nGlobal efficiency:")
    print(f"   - Avg Conversion Rate: {df_display['Conversion_Rate'].mean().round(1)}")
    print(f"   - Avg Passing Accuracy: {df_display['Succes_P%'].mean().round(1)}")
    print(f"   - Avg Goal Contribution: {df_display['Goal_Contribution_p90'].mean().round(2)}")
    
    print("\n" + "="*70)
    
    
def display_all_new_metrics(players_df):
    """display all new metrics created in the analysis for a better output and adjust some value like the impact score"""
    
    print(" all advanced metrics")
    print("=" * 50)
    
    # original columns from raw data
    original_columns = ['Players', 'Pos', 'MP', 'Starts', 'Min', 'Goals', 'Assists', 'xG', 'PrgC', 'PrgP', 'Pass', 'Succes_P%', 'Shots', 'Sot%', 'Take_On', 'TO%', 'Tkl', 'TklW', 'Int', 'Recov', 'Defduel', 'Defduel%']
    
    # find all new columns created in advanced measures
    all_columns = players_df.columns.tolist()
    new_metrics = [col for col in all_columns if col not in original_columns]
    
    print(f"total new metrics created: {len(new_metrics)}")
    print(f"new metrics: {new_metrics}")
    print()
    
    # display all new metrics for each player
    for player in players_df['Players'].unique():
        player_data = players_df[players_df['Players'] == player].iloc[0]
        
        print(f"player: {player} ({player_data['Player_Role']})")
        print("-" * 40)
        
        for col in new_metrics:
            value = player_data[col]
            
            print(f"  {col:30} : {value}")
        
        print()
        print("-" * 60)
    




    
    
    







