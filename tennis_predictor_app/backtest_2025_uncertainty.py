import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime
from betting_strategy_uncertainty import UncertaintyShrinkageBetting
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_results(results_df, save_path='backtest_analysis_2025.png'):
    """Plot betting performance over time"""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    plt.subplot(2, 1, 1)
    plt.plot(results_df['date'], results_df['bankroll'], label='Bankroll')
    plt.title('Bankroll Over Time')
    plt.xlabel('Date')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.hist(results_df['bet_size_percentage'], bins=50, alpha=0.75)
    plt.title('Distribution of Bet Sizes (% of Bankroll)')
    plt.xlabel('Bet Size (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_and_backtest_model(
    data_path='tennis_data_features_fixed.csv',
    train_cutoff='2024-12-31',
    test_start='2025-01-01',
    initial_bankroll=1000,
    min_prob=0.55,
    shrinkage_factor=0.5,
    uncertainty_threshold=0.2
):
    """
    Trains model on data until 2024 and tests on 2025 data using Uncertainty Shrinkage betting.
    """
    print("Loading data...")
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
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
    
    # Calculate accuracy and calibration
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Performance:")
    print("----------------")
    print(f"Accuracy on 2025 data: {accuracy:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Betting strategy backtest
    print("\nRunning uncertainty shrinkage betting backtest...")
    betting_strategy = UncertaintyShrinkageBetting(
        initial_bankroll=initial_bankroll,
        min_prob=min_prob,
        shrinkage_factor=shrinkage_factor,
        uncertainty_threshold=uncertainty_threshold
    )
    
    results = []
    current_bankroll = betting_strategy.bankroll
    initial_bankroll = current_bankroll
    bets_placed = 0
    winning_bets = 0
    total_wagered = 0
    total_profit = 0
    
    for i, (idx, row) in enumerate(test_data.iterrows()):
        prob = y_prob[i]
        actual = y_test.iloc[i]
        
        # Only bet when probability exceeds threshold
        if prob > betting_strategy.min_prob:
            kelly_fraction, fair_odds = betting_strategy.calculate_kelly_fraction(prob)
            bet_amount = initial_bankroll * kelly_fraction  # Use initial bankroll for flat betting
            total_wagered += bet_amount
            
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
                'bet_size_percentage': kelly_fraction * 100,
                'profit': profit,
                'bankroll': current_bankroll,
                'uncertainty': betting_strategy.calculate_uncertainty(prob)
            })
    
    # Calculate betting performance metrics
    total_profit = current_bankroll - initial_bankroll
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    win_rate = (winning_bets / bets_placed * 100) if bets_placed > 0 else 0
    
    print("\nBetting Performance:")
    print("------------------")
    print(f"Initial Bankroll: ${initial_bankroll:.2f}")
    print(f"Final Bankroll: ${current_bankroll:.2f}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Total Wagered: ${total_wagered:.2f}")
    print(f"ROI (Profit/Wagered): {roi:.2f}%")
    print(f"Total Return (Profit/Initial): {((total_profit) / initial_bankroll):.2%}")
    print(f"Bets Placed: {bets_placed}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    # Additional uncertainty metrics
    results_df = pd.DataFrame(results)
    avg_uncertainty = results_df['uncertainty'].mean()
    avg_bet_size = results_df['bet_size_percentage'].mean()
    
    print("\nUncertainty Analysis:")
    print("-------------------")
    print(f"Average Uncertainty: {avg_uncertainty:.2%}")
    print(f"Average Bet Size: {avg_bet_size:.2f}%")
    
    # Save results and generate plots
    results_df.to_csv('backtest_results_uncertainty_2025.csv', index=False)
    plot_results(results_df)
    
    print("\nBacktest results and analysis saved to:")
    print("- 'backtest_results_uncertainty_2025.csv'")
    print("- 'backtest_analysis_2025.png'")
    
    return model, results_df

if __name__ == "__main__":
    model, results = train_and_backtest_model()
