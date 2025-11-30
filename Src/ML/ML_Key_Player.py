import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import sys
sys.path.append('/files/Projet_Barca/')
sys.path.append('/files/Projet_Barca/Analysis/')

from Team_Data_Loader import load_raw_key_players_data
from Analysis_key_player import create_advanced_measures


class KMeansClustering:
    def __init__(self, players_df):
        self.players_df = players_df
        self.features = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.clusters = None
    
    def prepare_features(self):
        # Prepare key features for clustering
        """" I chnage this data into a decimal one"""
        
        self.players_df['Defduel%'] = self.players_df['Defduel%'] / 100
        
        """"I've choose some instesting values and with a good balance beetween attack, defense, dribbling, passing
            and not too much because for the K-means clustering because we have only 4 players and I get rid of redundance"""
        
        key_features = [
            'Goals_p90', 'Assists_p90', 'Progressive_Passes_p90',
            'Defduel%', 'Overall_Impact_Score', 'Conversion_Rate',
            'xG_efficiency', 'Successful_Take_Ons_p90', 'Interceptions_p90', 
            'Def_Duels_Won_p90','Successful_Tackles_p90', 'Ball_Recoveries_p90' ]
        
        available_features = [f for f in key_features if f in self.players_df.columns]
        self.features = self.players_df[available_features].fillna(0)
        
        print(f"Dataset: {len(self.players_df)} players")
        print(f"Features: {available_features}")
        return self.features
    
    def prepare_dimensional_features(self):
        # Prepare features by football dimensions (ratios)
        
        dimensional_features = {
            'scoring_ratio': ['Goals_p90', 'xG_p90', 'Conversion_Rate'],
            'playmaking_ratio': ['Assists_p90', 'Progressive_Passes_p90'],
            'defensive_ratio': ['Defduel%', 'Interceptions_p90', 'Tackles_p90', 'Successful_Tackles_p90', 'Ball_Recoveries_p90' ],
            'dribbling_ratio': ['Successful_Take_Ons_p90', 'Progressive_Carries_p90']}
    
        # Calculate mean of each dimension
    
        for dim_name, features in dimensional_features.items():
            available = [f for f in features if f in self.players_df.columns]
            if available:
                self.players_df[dim_name] = self.players_df[available].mean(axis=1)
    
        # Use these ratios for clustering
        ratio_features = [f for f in dimensional_features.keys() if f in self.players_df.columns]
        self.features = self.players_df[ratio_features].fillna(0)
    
        return self.features 
    
    def elbow_method_analysis(self, max_k=3):
        # Find optimal k using elbow method
        X_scaled = self.scaler.fit_transform(self.features)
    
        wcss = []
        silhouette_scores = []
    
        # Start from k=2 for silhouette score (k=1 is invalid)
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
        
            wcss.append(kmeans.inertia_)
        
            if len(np.unique(clusters)) > 1:
                silhouette_scores.append(silhouette_score(X_scaled, clusters))
            else:
                silhouette_scores.append(0)
    
        # Calculate WCSS (= Within-Cluster Sum of Squares) for k=1 manually
        
        wcss_k1 = np.sum((X_scaled - X_scaled.mean(axis=0))**2)
        wcss.insert(0, wcss_k1)
    
        optimal_k = self._find_elbow_point(wcss)
        print(f"Optimal clusters: {optimal_k}")
    
        return optimal_k, wcss, silhouette_scores
    
    def _find_elbow_point(self, wcss):
        # Find elbow point in WCSS curve
        
        n = len(wcss)
        if n <= 2:
            return 2
        
        # Calculate second derivatives to find elbow
        
        first_deriv = np.diff(wcss)
        second_deriv = np.diff(first_deriv)
        
        # Find point of maximum curvature (elbow)
        
        """" +2 because of double diff """
        
        if len(second_deriv) > 0:
            elbow_point = np.argmin(second_deriv) + 2  
        else:
            elbow_point = 2
        
        # For small datasets, limit to reasonable k
        
        max_reasonable_k = min(3, len(self.players_df) - 1)
        return min(elbow_point, max_reasonable_k)
    
    def perform_clustering(self, n_clusters=None):
        # Perform K-means clustering with optimal k
        if n_clusters is None:
            n_clusters, _, _ = self.elbow_method_analysis()
        
        X_scaled = self.scaler.fit_transform(self.features)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        self.players_df['KMeans_Cluster'] = self.clusters
        self.players_df['Cluster_Label'] = self.players_df['KMeans_Cluster'].apply(
            lambda x: f'Cluster_{x}'
        )
        
        # Calculate metrics
        if len(np.unique(self.clusters)) > 1:
            silhouette_avg = silhouette_score(X_scaled, self.clusters)
        else:
            silhouette_avg = 0
        
        print(f"\nK-means Clustering Results:")
        print(f"- Number of clusters: {n_clusters}")
        print(f"- Silhouette Score: {silhouette_avg:.3f}")
        print(f"- Cluster distribution:")
        print(self.players_df['KMeans_Cluster'].value_counts().sort_index())
        
        return self.clusters, silhouette_avg
    
    def analyze_clusters(self):
        # Analyze the resulting clusters 
        print("\n" + "="*50)
        print("Cluster analysis")
        print("="*50)
        
        # Basic stats per cluster
        numeric_cols = self.features.columns.tolist()
        
        cluster_stats = self.players_df.groupby('KMeans_Cluster')[numeric_cols].mean().round(3)
        
        print("Average metrics per cluster:")
        print(cluster_stats)
        
        # Compare with manual classification
        if 'Player_Role' in self.players_df.columns:
            comparison = pd.crosstab(
                self.players_df['Player_Role'], 
                self.players_df['KMeans_Cluster']
            )
            print("\nComparison with Manual Role Classification:")
            print(comparison)
            
            agreement = (comparison.max(axis=1).sum() / len(self.players_df)) * 100
            print(f"Agreement with manual classification: {agreement:.1f}%")
        
        # Player distribution
        print("\nPlayers per cluster:")
        for cluster in sorted(self.players_df['KMeans_Cluster'].unique()):
            cluster_players = self.players_df[self.players_df['KMeans_Cluster'] == cluster]
            print(f"\nCluster {cluster} ({len(cluster_players)} players):")
            for _, player in cluster_players.iterrows():
                role = player.get('Player_Role', 'N/A')
                impact = player.get('Overall_Impact_Score', 'N/A')
                print(f"  - {player['Players']} ({role}, Impact: {impact})")
        
        return cluster_stats
    
 
    
    def find_most_impactful(self):
        """Find most impactful player using K-means clustering without Overall_Impact_Score bias"""
        print("\n" + "="*60)
        print("Ranking analysis")
        print("="*60)
    
        # Use dimensional ratios instead of raw stats
        dimensional_features = {
            'scoring_ratio': ['Goals_p90', 'xG_p90', 'Conversion_Rate'],
            'playmaking_ratio': ['Assists_p90', 'Progressive_Passes_p90'],
            'defensive_ratio': ['Def_Duels_Won_p90', 'Interceptions_p90', 'Tackles_p90'],
            'dribbling_ratio': ['Successful_Take_Ons_p90', 'Progressive_Carries_p90']}
        # Calculate ratios
        ratio_data = {}
        for dim_name, features in dimensional_features.items():
            available = [f for f in features if f in self.players_df.columns]
            if available:
                ratio_data[dim_name] = self.players_df[available].mean(axis=1)
    
   
    
        # Create feature matrix from ratios
        X_ratios = pd.DataFrame(ratio_data).fillna(0)
        X_scaled = self.scaler.fit_transform(X_ratios)
    
        # Force k=4 clusters (1 per player)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        ranking_clusters = kmeans.fit_predict(X_scaled)
    
        # Calculate distance to global center for ranking
        global_center = X_scaled.mean(axis=0)
        distances = np.linalg.norm(X_scaled - global_center, axis=1)
    
        # Impact score = inverse distance (closer to center = more balanced)
        impact_scores = 1 / (1 + distances)
    
        self.players_df['Ranking_Score'] = impact_scores
        self.players_df['Ranking_Position'] = self.players_df['Ranking_Score'].rank(ascending=False)
    
        print(f"Features used: {list(ratio_data.keys())}")
        print(f"\nFinal Ranking:")
    
        ranked_players = self.players_df.sort_values('Ranking_Position')[
            ['Players', 'Player_Role', 'Ranking_Position', 'Ranking_Score']]
        print(ranked_players.to_string(index=False))
    
        # Compare with Overall_Impact_Score
        impact_ranking = self.players_df.sort_values('Overall_Impact_Score', ascending=False)['Players'].tolist()
        clustering_ranking = self.players_df.sort_values('Ranking_Position')['Players'].tolist()
    
        print(f"\nCompare:")
        print(f"\nOverall_Impact_Score ranking: {impact_ranking}")
        print(f"Clustering ranking: {clustering_ranking}")
    
        if impact_ranking == clustering_ranking:
            print(" Same result ")
        else:
            print("Different results: Methods disagree on ranking")
        
        best_player = self.players_df.nlargest(1, 'Ranking_Score').iloc[0]
        print(f"\nThe best of Barcelona during 2024-2025: {best_player['Players']}")
        print(f"   Role: {best_player['Player_Role']}")
        print(f"   Ranking Score: {best_player['Ranking_Score']:.3f}")
    
        return ranked_players

def kmeans_clustering_analysis(players_df):
    # Complete K-means clustering analysis
    analyzer = KMeansClustering(players_df)
    
    print("="*50)
    print("K-means clustering analysis")
    print("="*50)
    
    # Prepare features
    analyzer.prepare_features()
    
    # Find optimal k and perform clustering
    optimal_k, wcss, silhouette_scores = analyzer.elbow_method_analysis()
    clusters, score = analyzer.perform_clustering(n_clusters=optimal_k)
    
    # Analyze results
    cluster_stats = analyzer.analyze_clusters()
    
    
    
    print("\n" + "="*50)
    print("K-means Clustering completed")
    print("="*50)
    
    return analyzer
    


if __name__ == "__main__":
    
    raw_data = load_raw_key_players_data()
    players_df = create_advanced_measures(raw_data)
    
    print(f"Starting K-means clustering analysis with {len(players_df)} players...")
    
    # Run analysis
    analyzer = kmeans_clustering_analysis(players_df)
    results_df = analyzer.players_df
    impactful_player = analyzer.find_most_impactful()
    
    print("\nFinal cluster summary:")
    for cluster in sorted(results_df['KMeans_Cluster'].unique()):
        cluster_players = results_df[results_df['KMeans_Cluster'] == cluster]
        print(f"\nCluster {cluster}: {', '.join(cluster_players['Players'].tolist())}")







