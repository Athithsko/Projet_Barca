
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_team_data():
    # We are going to load the data set of the team and clean this one
    
    # 1. Uploading the data
    df = pd.read_csv('/files/Projet_Barca/Data_set/ProjetBarca.csv', sep=';', encoding='latin1')

    # 2. Clean the dataset
    
    print("=" * 70)
    
    print(f"Dataset Shape: {df.shape[0]} matches, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    """" I wrote this for being sure that the dataset isn't wrong"""
    print("\nFirst 5 matches preview:")
    print(df.head())

    # Date removing
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    
    # Convert commas to dots for numeric values to avoid issue with float and int
    numeric_columns = ['xG', 'xGA', 'Poss']
    for col in numeric_columns:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Create some target variables
    df['Victory'] = (df['Result'] == 'W').astype(int)
    df['Defeat'] = (df['Result'] == 'L').astype(int)
    df['Draw'] = (df['Result'] == 'D').astype(int)

    print("\n" + "-" * 40)
    print("Results distribution")
    print("-" * 40)
    print(df['Result'].value_counts())
    print(f"Win rate: {(df['Victory'].mean()*100):.1f}%")
    
    
    
    
    return df


def load_raw_key_players_data():
    # Load and process key players (Pedri, Raphinia, Lamine Yamal and Inigo Martinez) performance data
    
    players_df = pd.read_csv('/files/Projet_Barca/Data_set/Key_players.csv', sep=';', encoding='latin1')
    
    # CORR : Fix encoding issues before any display
    
    """ Beacause, I got issues to find the column for the players because the column had some bugged caracter and also for Martinez
    I have used this code just to see the name of my differents features and for fixing this later:
    print(f"Shape: {players_df.shape}")
    print(f"Colonnes: {players_df.columns.tolist()}")
    print(players_df.head())"""
    
    players_df = players_df.rename(columns={'ï»¿Player': 'Players'})
    players_df['Players'] = players_df['Players'].replace('MartÃ­nez', 'Martinez')
    
    print("\n" + "=" * 70)
    print("Key Players Data analysis")
    print("=" * 70)
    
    print(f"Raw dataset shape: {players_df.shape[0]} players, {players_df.shape[1]} metrics")
    print(f"Available columns: {players_df.columns.tolist()}")
    
    print("\nFirst rows of cleaned data:")
    print(players_df.head())
    
    print("\n" + "-" * 40)

    
    print("Column 'Players' successfully standardized")
    print(f"Players identified: {players_df['Players'].tolist()}")
    
    print(f"Initial dataset shape: {players_df.shape}")
    
    return players_df



def validate_players_data(players_df):
    
    
    # Converting commas into points
    numeric_columns = ['Succes_P%', 'Sot%', 'TO%', 'xG', 'PrgC', 'PrgP', 'Pass', 'Shots', 'Take_On', 'Tkl', 'TklW', 'Int', 'Recov']
    
    for col in numeric_columns:
        if col in players_df.columns and players_df[col].dtype == 'object':
            players_df[col] = players_df[col].astype(str).str.replace(',', '.').astype(float)
    
    # Validation of the different percentages
    percentage_cols = ['Succes_P%', 'Sot%', 'TO%']
    for col in percentage_cols:
        if col in players_df.columns:
            players_df[col] = players_df[col].clip(0, 100)
    

    print("Data validation: Numeric conversion completed")
    
    return players_df
    


"""" All the code before serve the purpose of cleaning and improving the data set for this function"""

def clear_key_player_data(players_df):
    
    # Data validation
    players_df = validate_players_data(players_df)
    

    
    print("\n" + "-" * 40)
    print("Data processing completed")
    print("-" * 40)
    
    print(f"Final dataset shape: {players_df.shape}")
    
    
    
    
    print("\n" + "=" * 70)
    
    return players_df





# Main execution


if __name__ == "__main__":
    
    """" I've wrote this if to import these function in other code"""
    
    print("Fc Barcelona 2024-2025 season stats")
    print("=" * 70)

    team_df = load_team_data()
    players_raw_df = load_raw_key_players_data()
    player_clear_df= clear_key_player_data(players_raw_df)

    print("\n" + "=" * 70)

    print(f"Team data: {team_df.shape[0]} matches ready for analysis")
    print(f"Players data: {player_clear_df.shape[0]} players with {player_clear_df.shape[1]} metrics ready for analysis")
    print("=" * 70)













