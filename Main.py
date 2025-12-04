


import sys
sys.path.append('/files/Projet_Barca/Src/')
sys.path.append('/files/Projet_Barca/Src/Analysis/')
sys.path.append('/files/Projet_Barca/Src/ML/')
sys.path.append('/files/Projet_Barca/Src/Graphics/')


from Team_Data_Loader import load_team_data, load_raw_key_players_data, clear_key_player_data
from Analysis_team import explanatory_analysis, assign_opponent_tier
from Analysis_key_player import create_advanced_measures, get_player_statistics_summary, display_all_new_metrics, final_data_formatting, calculate_impact_score, classify_player_role
from Visu_Team import create_comprehensive_dashboard
from Visu_Key import create_key_players_dashboard
from ML_Team import prepare_temporal_split, train_ml_models, predict_future_matches, compare_predictions_with_reality
from ML_Key_Player import kmeans_clustering_analysis, KMeansClustering

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
    
    X_train, X_test, y_train, y_test, features, test_df = prepare_temporal_split(processed_df)
    ml_results = train_ml_models(X_train, X_test, y_train, y_test, features)
    comparison_df, accuracy_summary = compare_predictions_with_reality(ml_results, X_test, y_test, test_df)
    
    print("\nML Trainig finish")
    
    
    

    
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
    
    
    
        
    
    
    
    #Summary
    print("\n" + "=" * 70)
    print("Analysis Done ")
    print("=" * 70)
    print(f"Team data: {team_df.shape[0]} matches analyzed with {len(features)} features")
    print(f"Players data: {player_clear_df.shape[0]} players with {player_clear_df.shape[1]} metrics")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()





