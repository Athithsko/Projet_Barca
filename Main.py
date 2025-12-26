import pandas as pd


import os
import sys

# Get the directory where Main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, 'Src'))
sys.path.append(os.path.join(BASE_DIR, 'Src', 'Analysis'))
sys.path.append(os.path.join(BASE_DIR, 'Src', 'ML'))
sys.path.append(os.path.join(BASE_DIR, 'Src', 'Graphics'))


from Team_Data_Loader import load_team_data, load_raw_key_players_data, clear_key_player_data
from Analysis_team import explanatory_analysis, assign_opponent_tier
from Analysis_key_player import create_advanced_measures, get_player_statistics_summary, display_all_new_metrics, final_data_formatting, calculate_impact_score, classify_player_role

from ML_Team import prepare_temporal_split, train_ml_models, predict_future_matches, compare_predictions_with_reality, calculate_auc_scores, cross_validate_models
from ML_Key_Player import kmeans_clustering_analysis, KMeansClustering

from Visu_Team import create_comprehensive_dashboard, plot_roc_curves_comparison, plot_roc_auc_bar
from Visu_Key import create_key_players_dashboard

def main():
    # Main orchestor function that call everything
    
    print("\n" + "=" * 70)
    print("Fc Barcelona Complete Analysis")
    print("=" * 70 + "\n")
    
    # Loading data
    print("Loading data...")
    team_df = load_team_data()
    players_raw_df = load_raw_key_players_data()
    player_clear_df = clear_key_player_data(players_raw_df)
    
    # Analyzing data
    print("\nAnalyzing team data...")
    results = explanatory_analysis(team_df)
    processed_df = results['processed_data']
    
    # Defining features
    print("\nDefining features...")
    features = ['Equipe_type', 'xG', 'xGA', 'Poss', 'Venue', 'Opponent_tier', 'xG_efficiency', 'xGA_efficiency']
    
    # Creating the ML models for the team
    print("\nCreating ML models...")
    
    # Train ML models
    X_train, X_test, y_train, y_test, features, test_df = prepare_temporal_split(processed_df)
    
    
    
    # Train and evaluate ML models 
    ml_results, scaler = train_ml_models(X_train, X_test, y_train, y_test, features)

    # Compare predictions vs actual results 
    comparison_df, accuracy_summary = compare_predictions_with_reality(ml_results, X_test, y_test, test_df)

    # Calculate AUC scores 
    auc_results = calculate_auc_scores(y_test, ml_results, X_test, scaler)
    
    # Cross-validation for more accurate ROC-AUC scores
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    cv_results = cross_validate_models(X_full, y_full, features)
    
    print("\nML Trainig finish")
    
    # Visualizations of team
    
    create_comprehensive_dashboard(processed_df, ml_results, comparison_df, features, accuracy_summary, auc_results, y_test, X_test, scaler, cv_results)
    
    
    

    
    #Analysis Key player
    print("\nAnalyzing data of Key Players...")
    players_analyzed = create_advanced_measures(players_raw_df)
    
    players_final = final_data_formatting(players_analyzed)
    
    display = display_all_new_metrics(players_final)
    
    
    summary = get_player_statistics_summary(players_final)
    
    #The Ml for the key players
    print(f"\nStarting K-means clustering analysis with {len(players_analyzed)} players...")
    
    
    analyzer = kmeans_clustering_analysis(players_analyzed)
    results_df = analyzer.players_df
    impactful_player = analyzer.find_most_impactful()
    
    
    
    print("\nFinal cluster summary:")
    for cluster in sorted(results_df['KMeans_Cluster'].unique()):
        cluster_players = results_df[results_df['KMeans_Cluster'] == cluster]
        print(f"\nCluster {cluster}: {', '.join(cluster_players['Players'].tolist())}")
    #Visualization according to Key Players
    
    create_key_players_dashboard(results_df)
    
    
    
        
    
    
    
    #Summary
    print("\n" + "=" * 70)
    print("Analysis Done ")
    print("=" * 70)
    print(f"Team data: {team_df.shape[0]} matches analyzed with {len(features)} features")
    print(f"Players data: {player_clear_df.shape[0]} players with {player_clear_df.shape[1]} metrics")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()





