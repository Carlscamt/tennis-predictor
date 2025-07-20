import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime
from betting_strategy_v2 import AdaptiveBettingStrategy

def train_and_backtest_model(
    data_path='tennis_data_features_fixed.csv',
    train_cutoff='2024-12-31',
    test_start='2025-01-01'
):
    """
    Trains model on data until 2024 and tests on 2025 data.
    Also runs betting strategy backtest.
    """
    print("Loading data...")
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Split features
    feature_cols = [
        'career_win_pct_diff',
        'surface_win_pct_diff',
        'recent_form_diff',
        'h2h_win_pct_diff',
        'ranking_diff'
    ]
    
    # Split data by date
    train_data = data[data['Date'] <= train_cutoff]
    test_data = data[(data['Date'] > train_cutoff) & (data['Date'] >= test_start)]
    
    print(f"\nTraining data period: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"Testing data period: {test_data['Date'].min()} to {test_data['Date'].max()}")
    print(f"Number of training matches: {len(train_data)}")
    print(f"Number of testing matches: {len(test_data)}")
    
    # Train model
    print("\nTraining model...")
    model = xgb.XGBClassifier(
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    X_train = train_data[feature_cols]
    y_train = train_data['Win']
    model.fit(X_train, y_train)
    
    # Test predictions
    print("\nEvaluating on 2025 data...")
    X_test = test_data[feature_cols]
    y_test = test_data['Win']
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Performance:")
    print("----------------")
    print(f"Accuracy on 2025 data: {accuracy:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Betting strategy backtest
    print("\nRunning betting strategy backtest...")
    betting_strategy = AdaptiveBettingStrategy(initial_bankroll=1000)
    
    results = []
    current_bankroll = betting_strategy.bankroll
    bets_placed = 0
    winning_bets = 0
    total_profit = 0
    
    for i, (idx, row) in enumerate(test_data.iterrows()):
        prob = y_prob[i]
        actual = y_test.iloc[i]
        
        # Only bet when probability exceeds threshold
        if prob > betting_strategy.min_prob:
            kelly_fraction, fair_odds = betting_strategy.calculate_kelly_fraction(prob)
            bet_amount = current_bankroll * kelly_fraction
            
            if actual == 1:
                profit = bet_amount * (fair_odds - 1)
                winning_bets += 1
            else:
                profit = -bet_amount
                
            current_bankroll += profit
            total_profit += profit
            bets_placed += 1
            
            results.append({
                'date': row['Date'],
                'probability': prob,
                'bet_amount': bet_amount,
                'profit': profit,
                'bankroll': current_bankroll
            })
    
    # Calculate betting performance metrics
    roi = (total_profit / (1000)) * 100 if bets_placed > 0 else 0
    win_rate = (winning_bets / bets_placed * 100) if bets_placed > 0 else 0
    
    print("\nBetting Performance:")
    print("------------------")
    print(f"Final Bankroll: ${current_bankroll:.2f}")
    print(f"Total Return: {((current_bankroll - 1000) / 1000):.2%}")
    print(f"ROI: {roi:.2f}%")
    print(f"Bets Placed: {bets_placed}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('backtest_results_2025.csv', index=False)
    print("\nBacktest results saved to 'backtest_results_2025.csv'")
    
    return model, results_df

if __name__ == "__main__":
    model, results = train_and_backtest_model()
