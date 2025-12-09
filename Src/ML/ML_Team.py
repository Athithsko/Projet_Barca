import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler





def prepare_temporal_split(df, train_size=30):
    """Split data temporally - first 30 matches for training, last 8 for testing"""
    
    # Features for modeling
    features = ['Equipe_type', 'xG', 'xGA', 'Poss', 'Venue', 'Opponent_tier', 'xG_efficiency', 'xGA_efficiency']
    
    # Verify all features exist
    available_features = [f for f in features if f in df.columns]
    print(f"Features used: {available_features}")
    
    # Temporal split
    train_df = df.head(train_size)
    test_df = df.tail(len(df) - train_size)
    
    X_train = train_df[available_features]
    y_train = train_df['Victory']
    X_test = test_df[available_features]
    y_test = test_df['Victory']
    
    print(f"Training set: {len(X_train)} matches (first {train_size} matches)")
    print(f"Test set: {len(X_test)} matches (last {len(df) - train_size} matches)")
    
    return X_train, X_test, y_train, y_test, available_features, test_df

def train_ml_models(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate the 3 ML models with temporal split"""
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    print("\n" + "="*70)
    print("Ml model results")
    print("="*70)
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Use scaled data for Logistic Regression, raw for tree-based models
        if name == 'Logistic Regression':
            X_tr = X_train_scaled
            X_te = X_test_scaled
        else:
            X_tr = X_train
            X_te = X_test
        
        # Train model
        model.fit(X_tr, y_train)
        
        # Predictions
        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Print results
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nFeature Importance:")
            print(feature_importance.to_string(index=False))
    
    return results, scaler

def predict_future_matches(model, scaler, new_data, feature_names, model_name):
    """Make predictions on new matches"""
    
    # Prepare features
    X_new = new_data[feature_names]
    
    # Scale if Logistic Regression
    if model_name == 'Logistic Regression':
        X_new = scaler.transform(X_new)
    
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1] if hasattr(model, 'predict_proba') else None
    
    return predictions, probabilities

def compare_predictions_with_reality(ml_results, X_test, y_test, test_df):
    """Compare model predictions with actual results"""
    
    print("\n" + "="*80)
    print("Predictions vs Reality")
    print("="*80)
    
    # Create comparison DataFrame
    comparison_data = {
        'Actual_Result': y_test,
        'Actual_Outcome': test_df['Result']
    }
    
    # Add predictions from each model
    for model_name, results in ml_results.items():
        comparison_data[f'{model_name}_Pred'] = results['predictions']
        comparison_data[f'{model_name}_Prob'] = results['probabilities']
    
    comparison_df = pd.DataFrame(comparison_data, index=test_df.index)
    
    # Add match information
    comparison_df['Opponent'] = test_df['Opponent']
    comparison_df['Venue'] = test_df['Venue'].map({1: 'Home', 0: 'Away'})
    comparison_df['GF'] = test_df['GF']
    comparison_df['GA'] = test_df['GA']
    comparison_df['xG'] = test_df['xG']
    comparison_df['xGA'] = test_df['xGA']
    
    # Reorder columns for better readability
    cols = ['Opponent', 'Venue', 'Actual_Outcome', 'GF', 'GA', 'xG', 'xGA', 'Actual_Result']
    for model_name in ml_results.keys():
        cols.extend([f'{model_name}_Pred', f'{model_name}_Prob'])
    
    comparison_df = comparison_df[cols]
    
    print("\nMatch by match comparision:")
    print("-" * 80)
    
    for idx, row in comparison_df.iterrows():
        print(f"\n {row['Opponent']} ({row['Venue']})")
        print(f"   Actual: {row['Actual_Outcome']} ({row['GF']}-{row['GA']}) | xG: {row['xG']}-{row['xGA']}")
        
        for model_name in ml_results.keys():
            pred = row[f'{model_name}_Pred']
            prob = row[f'{model_name}_Prob']
            correct = "True" if pred == row['Actual_Result'] else "False"
            print(f"   {model_name}: {correct} {'Win' if pred == 1 else 'Loss/Draw'} (prob: {prob:.2f})")
    
    # Summary statistics
    print("\n" + "="*80)
    print("Prediction summary")
    print("="*80)
    
    accuracy_summary = {}
    for model_name, results in ml_results.items():
        correct_predictions = (results['predictions'] == y_test).sum()
        total_predictions = len(y_test)
        accuracy = correct_predictions / total_predictions
        
        accuracy_summary[model_name] = {
            'correct': correct_predictions,
            'total': total_predictions,
            'accuracy': accuracy
        }
        
        print(f"\n{model_name}:")
        print(f"  Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Wrong predictions: {total_predictions - correct_predictions}")
    
    # Best performing model
    best_model = max(accuracy_summary.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best Model: {best_model[0]} ({best_model[1]['accuracy']:.1%} accuracy)")
    
    return comparison_df, accuracy_summary


def print_auc_scores(y_test, ml_results, X_test, scaler=None):
    #Print ROC-AUC scores without plots
    print("\n" + "="*50)
    print("ROC-AUC Scores")
    print("="*50)
    
    # Scale X_test for Logistic Regression
    X_test_scaled = scaler.transform(X_test) if scaler else X_test
    
    auc_scores = {}
    for model_name, result in ml_results.items():
        model = result['model']
        X_use = X_test_scaled if model_name == 'Logistic Regression' else X_test
        
        y_prob = model.predict_proba(X_use)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
        auc_scores[model_name] = auc_score
        
        print(f"  {model_name}: {auc_score:.4f}")
    
    best = max(auc_scores, key=auc_scores.get)
    print(f"\nBest model by AUC: {best} ({auc_scores[best]:.4f})")
    print("="*50)





