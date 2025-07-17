import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save_model(data_path='tennis_data_features_fixed.csv'):
    """Trains and saves an XGBoost model on the tennis data with proper time-based validation."""
    print("Loading data...")
    data = pd.read_csv(data_path, low_memory=False)
    data['Date'] = pd.to_datetime(data['Date'])
    data.fillna(0, inplace=True)
    data.sort_values(by='Date', inplace=True)

    features = [
        'career_win_pct_diff',
        'surface_win_pct_diff',
        'recent_form_diff',
        'h2h_win_pct_diff',
        'ranking_diff'
    ]
    
    print("Preparing features...")
    X = data[features]
    y = data['Win']
    
    # Use TimeSeriesSplit for proper temporal validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize model with conservative parameters
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("\nTraining model with time series cross-validation...")
    accuracies = []
    
    # Perform time series cross-validation
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Get date range for this fold
        train_dates = data.iloc[train_index]['Date']
        test_dates = data.iloc[test_index]['Date']
        print(f"\nFold {fold}")
        print(f"Training period: {train_dates.min()} to {train_dates.max()}")
        print(f"Testing period: {test_dates.min()} to {test_dates.max()}")
        
        # Train the model
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        
        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    
    print(f"\nAverage Accuracy across folds: {np.mean(accuracies):.4f}")
    print(f"Std Dev of Accuracy: {np.std(accuracies):.4f}")
    
    # Final fit on all data
    print("\nTraining final model on all data...")
    model.fit(X, y)
    
    # Feature importance analysis
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print("------------------")
    for _, row in importance.iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    print("\nSaving model...")
    # Save the model
    joblib.dump(model, 'tennis_model_fixed.joblib')
    
    # Save feature importance for future reference
    importance.to_csv('feature_importance_fixed.csv', index=False)
    print("Model and feature importance saved successfully!")

if __name__ == '__main__':
    train_and_save_model()
