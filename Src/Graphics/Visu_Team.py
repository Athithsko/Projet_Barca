import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import os

OUTPUT_DIR = 'Graphics_Output'
os.makedirs(OUTPUT_DIR, exist_ok=True)



def setup_visualization_style():
    # Setup Barcelona-themed color  visualization style
    plt.style.use('default')
    sns.set_palette(["blue", "crimson", "gold", "green"])
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_model_performance(accuracy_summary):
    # Plot model performance comparison
    setup_visualization_style()
    
    if not accuracy_summary:
        print("No model performance data available")
        return
    
    models = list(accuracy_summary.keys())
    accuracies = [accuracy_summary[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'crimson', 'gold'], alpha=0.8)
    
    plt.title('Model Performance Comparison', fontweight='bold', fontsize=14)
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy values on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{accuracy:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_model_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(ml_results, feature_names):
    # Plot feature importance from ML models 
    setup_visualization_style()
    
    if not ml_results or not feature_names:
        print("No feature importance data available")
        return
    
    fig, axes = plt.subplots(1, len(ml_results), figsize=(6*len(ml_results), 5))
    
    # Handle single model case
    if len(ml_results) == 1:
        axes = [axes]
    
    for idx, (model_name, model) in enumerate(ml_results.items()):
        if hasattr(model, 'feature_importances_'):
            importances = np.array(model.feature_importances_)
            
            # Ensure we have the right number of features
            if len(importances) != len(feature_names):
                print(f"Warning: {model_name} has {len(importances)} importances but {len(feature_names)} features")
                continue
            
            # Sort by importance
            sorted_idx = np.argsort(importances)[::-1].tolist()
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_importances = importances[sorted_idx]
            
            # Plot
            axes[idx].barh(sorted_features, sorted_importances, color='blue', alpha=0.8)
            axes[idx].set_title(f'{model_name}\nFeature Importance', fontweight='bold')
            axes[idx].set_xlabel('Importance')
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(df):
    # Plot correlation matrix for victory analysis
    setup_visualization_style()
    
    # Select numeric features for correlation
    numeric_features = ['Victory', 'Equipe_type', 'xG', 'xGA', 'Poss', 'GF', 'GA', 'Opponent_tier', 'xG_efficiency', 'xGA_efficiency']
    available_features = [f for f in numeric_features if f in df.columns]
    
    corr_matrix = df[available_features].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True, 
                linewidths=0.5,
                cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix - Victory Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_opponent_tier_analysis(df):
    # Plot analysis by opponent tier 
    setup_visualization_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Win rate by opponent tier
    win_rate_by_tier = df.groupby('Opponent_tier')['Victory'].agg(['mean', 'count'])
    
    ax1.bar(win_rate_by_tier.index, win_rate_by_tier['mean'], color='blue', alpha=0.8)
    ax1.set_title('Win Rate by Opponent Tier', fontweight='bold')
    ax1.set_xlabel('Opponent Tier (1=Elite, 4=Relegation)')
    ax1.set_ylabel('Win Rate')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Goals by opponent tier
    goals_by_tier = df.groupby('Opponent_tier')[['GF', 'GA']].mean()
    
    x = np.arange(len(goals_by_tier))
    width = 0.35
    
    ax2.bar(x - width/2, goals_by_tier['GF'], width, label='Goals For', color='blue', alpha=0.8)
    ax2.bar(x + width/2, goals_by_tier['GA'], width, label='Goals Against', color='crimson', alpha=0.8)
    ax2.set_title('Average Goals by Opponent Tier', fontweight='bold')
    ax2.set_xlabel('Opponent Tier (1=Elite, 4=Relegation)')
    ax2.set_ylabel('Average Goals')
    ax2.set_xticks(x)
    ax2.set_xticklabels(goals_by_tier.index)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # xG by opponent tier
    xg_by_tier = df.groupby('Opponent_tier')[['xG', 'xGA']].mean()
    
    ax3.bar(x - width/2, xg_by_tier['xG'], width, label='xG', color='blue', alpha=0.8)
    ax3.bar(x + width/2, xg_by_tier['xGA'], width, label='xGA', color='gold', alpha=0.8)
    ax3.set_title('Expected Goals by Opponent Tier', fontweight='bold')
    ax3.set_xlabel('Opponent Tier (1=Elite, 4=Relegation)')
    ax3.set_ylabel('Expected Goals')
    ax3.set_xticks(x)
    ax3.set_xticklabels(xg_by_tier.index)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Match distribution by opponent tier
    tier_distribution = df['Opponent_tier'].value_counts().sort_index()
    
    ax4.pie(tier_distribution.values, labels=[f'Tier {t}' for t in tier_distribution.index], 
            autopct='%1.1f%%', colors=['blue', 'crimson', 'gold', 'green'])
    ax4.set_title('Match Distribution by Opponent Tier', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_opponent_tier_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_team_type_comparison(df):
    # Plot Equipe_type analysis with Win Rate, Goals, and xG Efficiency 
    setup_visualization_style()
    
    # 3 subplots in one row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Win rate by Equipe_type
    win_rate = df.groupby('Equipe_type')['Victory'].mean()
    match_count = df.groupby('Equipe_type').size()
    
    bars = ax1.bar(win_rate.index, win_rate.values, 
                   color=['crimson', 'blue'], alpha=0.8)
    ax1.set_title('Win Rate by Team Type', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Team Type (0=Without Stars, 1=With Stars)')
    ax1.set_ylabel('Win Rate')
    ax1.set_ylim(0, 1)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Without Stars', 'With Stars'])
    ax1.grid(True, alpha=0.3)
    
    # Adding values on bars
    for i, (bar, rate, count) in enumerate(zip(bars, win_rate.values, match_count.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}\n({count} matches)', ha='center', fontweight='bold')
    
    # Goals comparison
    goals_by_type = df.groupby('Equipe_type')[['GF', 'GA']].mean()
    x = np.arange(len(goals_by_type))
    width = 0.35
    
    ax2.bar(x - width/2, goals_by_type['GF'], width, 
            label='Goals For', color='blue', alpha=0.8)
    ax2.bar(x + width/2, goals_by_type['GA'], width, 
            label='Goals Against', color='crimson', alpha=0.8)
    ax2.set_title('Average Goals by Team Type', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Team Type (0=Without Stars, 1=With Stars)')
    ax2.set_ylabel('Average Goals')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Without Stars', 'With Stars'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # xG efficiency comparison
    efficiency_by_type = df.groupby('Equipe_type')[['xG_efficiency', 'xGA_efficiency']].mean()
    
    ax3.bar(x - width/2, efficiency_by_type['xG_efficiency'], width, 
            label='xG Efficiency', color='blue', alpha=0.8)
    ax3.bar(x + width/2, efficiency_by_type['xGA_efficiency'], width, 
            label='xGA Efficiency', color='gold', alpha=0.8)
    ax3.set_title('xG Efficiency by Team Type', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Team Type (0=Without Stars, 1=With Stars)')
    ax3.set_ylabel('Efficiency Ratio')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Without Stars', 'With Stars'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_team_type_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # Display key statistics
    print("\nTeam type analysis Summary:")
    print("-" * 40)
    for team_type in df['Equipe_type'].unique():
        subset = df[df['Equipe_type'] == team_type]
        team_label = "With Stars" if team_type == 1 else "Without Stars"
        print(f"\n{team_label}:")
        print(f"  Matches: {len(subset)}")
        print(f"  Win Rate: {subset['Victory'].mean():.1%}")
        print(f"  Avg Goals: {subset['GF'].mean():.2f}")
        print(f"  Avg xG: {subset['xG'].mean():.2f}")
        print(f"  xG Efficiency: {subset['xG_efficiency'].mean():.2f}")


def plot_team_type_vs_opponent(df):
    # Plot interaction between Equipe_type and Opponent_tier 
    setup_visualization_style()
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(df['Equipe_type'], df['Opponent_tier'], 
                           values=df['Victory'], aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_tab, annot=True, cmap='RdYlGn', center=0.5, 
                fmt='.2%', cbar_kws={'label': 'Win Rate'})
    plt.title('Win Rate: Team Type vs Opponent Tier', fontweight='bold', fontsize=16)
    plt.xlabel('Opponent Tier (1=Elite, 4=Relegation)')
    plt.ylabel('Team Type (0=Without Stars, 1=With Stars)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_team_type_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, '06_team_type_vs_opponent.png'), dpi=300, bbox_inches='tight')
    plt.close()



def create_comprehensive_dashboard(df, ml_results, comparison_df, feature_names, accuracy_summary):
    # Create comprehensive visualization dashboard 
    print("Generating Comprehensive Visualization Dashboard...")
    
    # Basic ML analyses
    plot_model_performance(accuracy_summary)
    plot_feature_importance(ml_results, feature_names)
    
    # Correlation analyses
    plot_correlation_matrix(df)
    plot_opponent_tier_analysis(df)
    
    # Team analyses
    plot_team_type_comparison(df)
    plot_team_type_vs_opponent(df)
    
    print(f"\nAll graphics saved to: {OUTPUT_DIR}/")
    
def plot_roc_curves_comparison(y_test, ml_results, X_test, scaler=None):
    # Plot ROC curves for all models on the same figure
    setup_visualization_style()
    
    X_test_scaled = scaler.transform(X_test) if scaler else X_test
    colors = ['blue', 'crimson', 'gold']
    
    
    plt.figure(figsize=(10, 8))
    
    auc_scores = {}
    for idx, (model_name, result) in enumerate(ml_results.items()):
        model = result['model']
        X_use = X_test_scaled if model_name == 'Logistic Regression' else X_test
        
        y_prob = model.predict_proba(X_use)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc
        
        plt.plot(fpr, tpr, color=colors[idx], lw=2, 
                 label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curves Comparison - Barça Victory Prediction', fontweight='bold', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, '07_roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return auc_scores


def plot_roc_auc_bar(auc_scores):
    # Plot AUC scores as bar chart
    setup_visualization_style()
    
    models = list(auc_scores.keys())
    scores = list(auc_scores.values())
    
    # Sort by score
    sorted_pairs = sorted(zip(scores, models), reverse=True)
    scores, models = zip(*sorted_pairs)
    
    colors = ['blue', 'crimson', 'gold']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=colors, alpha=0.8)
    
    plt.axhline(y=0.5, color='gray', linestyle='--', lw=2, label='Random Classifier')
    plt.title('Model Comparison by ROC-AUC Score', fontweight='bold', fontsize=14)
    plt.ylabel('AUC Score', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, '08_roc_auc_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_auc_scores(y_test, ml_results, X_test, scaler=None):
    """Print ROC-AUC scores and generate ROC plots"""
    
    # Generate ROC curves plot
    auc_scores = plot_roc_curves_comparison(y_test, ml_results, X_test, scaler)
    
    # Generate AUC bar chart
    plot_roc_auc_bar(auc_scores)
    

    print(f"ROC graphics saved to: {OUTPUT_DIR}/")
    
    return auc_scores

    









