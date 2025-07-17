import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from datetime import datetime

class AdaptiveBettingStrategy:
    def __init__(self, initial_bankroll=1000, max_bet_pct=2.5, min_odds=1.2):
        """
        Initialize betting strategy.
        
        Args:
            initial_bankroll: Starting bankroll amount
            max_bet_pct: Maximum percentage of bankroll to bet (default 2.5%)
            min_odds: Minimum odds to consider for betting
        """
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.max_bet_pct = max_bet_pct / 100  # Convert to decimal
        self.min_odds = min_odds
        self.bets = []
        self.running_profit = []
        self.edge_threshold = 0.05  # Minimum edge to place a bet
        self.kelly_fraction = 0.5   # Conservative Kelly criterion fraction
        
    def calculate_optimal_bet(self, predicted_prob, odds, edge_multiplier=1.0):
        """
        Calculate optimal bet size using fractional Kelly criterion.
        
        Args:
            predicted_prob: Model's predicted probability of winning
            odds: Betting odds (decimal format)
            edge_multiplier: Multiplier for edge threshold based on confidence
        """
        if odds < self.min_odds:
            return 0
            
        # Calculate edge (expected value)
        fair_prob = 1 / odds
        edge = predicted_prob - fair_prob
        
        if edge < self.edge_threshold * edge_multiplier:
            return 0
            
        # Kelly criterion formula
        kelly_bet = (predicted_prob * (odds - 1) - (1 - predicted_prob)) / (odds - 1)
        kelly_bet = max(0, kelly_bet)  # No negative bets
        
        # Apply fractional Kelly and max bet limit
        bet_size = min(
            kelly_bet * self.kelly_fraction * self.bankroll,
            self.bankroll * self.max_bet_pct
        )
        
        return bet_size
        
    def place_bet(self, predicted_prob, actual_outcome, odds, date, confidence=1.0):
        """
        Place a bet and record the outcome.
        
        Args:
            predicted_prob: Model's predicted probability
            actual_outcome: Actual match result (1 for win, 0 for loss)
            odds: Betting odds (decimal format)
            date: Date of the match
            confidence: Confidence multiplier for edge threshold
        """
        bet_amount = self.calculate_optimal_bet(predicted_prob, odds, confidence)
        
        if bet_amount > 0:
            # Record bet details
            bet_result = {
                'date': date,
                'bankroll_before': self.bankroll,
                'bet_amount': bet_amount,
                'predicted_prob': predicted_prob,
                'odds': odds,
                'actual_outcome': actual_outcome,
                'won': actual_outcome == 1
            }
            
            # Update bankroll
            if actual_outcome == 1:
                profit = bet_amount * (odds - 1)
                self.bankroll += profit
                bet_result['profit'] = profit
            else:
                self.bankroll -= bet_amount
                bet_result['profit'] = -bet_amount
                
            bet_result['bankroll_after'] = self.bankroll
            bet_result['roi'] = bet_result['profit'] / bet_amount
            
            self.bets.append(bet_result)
            self.running_profit.append(self.bankroll - self.initial_bankroll)
            
    def get_performance_metrics(self):
        """Calculate and return performance metrics for the betting strategy."""
        if not self.bets:
            return None
            
        bets_df = pd.DataFrame(self.bets)
        metrics = {
            'total_bets': len(self.bets),
            'winning_bets': sum(bet['won'] for bet in self.bets),
            'win_rate': sum(bet['won'] for bet in self.bets) / len(self.bets),
            'total_profit': self.bankroll - self.initial_bankroll,
            'roi': (self.bankroll - self.initial_bankroll) / self.initial_bankroll,
            'avg_bet_size': np.mean([bet['bet_amount'] for bet in self.bets]),
            'max_drawdown': self.calculate_max_drawdown(),
            'profit_factor': self.calculate_profit_factor(),
            'sharpe_ratio': self.calculate_sharpe_ratio()
        }
        
        return metrics
        
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown from peak bankroll."""
        peaks = pd.Series(self.running_profit).expanding(min_periods=1).max()
        drawdowns = pd.Series(self.running_profit) - peaks
        return abs(drawdowns.min())
        
    def calculate_profit_factor(self):
        """Calculate ratio of gross profits to gross losses."""
        if not self.bets:
            return 0
            
        profits = sum(bet['profit'] for bet in self.bets if bet['profit'] > 0)
        losses = abs(sum(bet['profit'] for bet in self.bets if bet['profit'] < 0))
        return profits / losses if losses != 0 else float('inf')
        
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio of betting returns."""
        if not self.bets:
            return 0
            
        returns = pd.Series([bet['roi'] for bet in self.bets])
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
    def plot_performance(self, save_path=None):
        """Plot betting performance over time."""
        if not self.bets:
            print("No bets to plot")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.running_profit)), self.running_profit)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Number of Bets')
        plt.ylabel('Profit')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
def backtest_strategy(model_path, data_path, strategy_params=None):
    """
    Backtest the betting strategy on historical data.
    
    Args:
        model_path: Path to the trained model file
        data_path: Path to the test data CSV
        strategy_params: Dictionary of strategy parameters
    """
    # Load model and data
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Sort by date
    data = data.sort_values('Date')
    
    # Initialize strategy
    if strategy_params is None:
        strategy_params = {
            'initial_bankroll': 1000,
            'max_bet_pct': 2.5,
            'min_odds': 1.2
        }
    
    strategy = AdaptiveBettingStrategy(**strategy_params)
    
    # Get feature columns (assuming same as training)
    feature_cols = [
        'career_win_pct_diff',
        'surface_win_pct_diff',
        'recent_form_diff',
        'h2h_win_pct_diff',
        'ranking_diff'
    ]
    
    # Backtest on each match
    for idx, row in data.iterrows():
        # Get match features
        X = row[feature_cols].values.reshape(1, -1)
        
        # Get model prediction probability
        pred_prob = model.predict_proba(X)[0][1]
        
        # Use B365 odds if available, otherwise skip
        if pd.isna(row['B365W']) or pd.isna(row['B365L']):
            continue
            
        # Get appropriate odds based on whether it's a win or loss
        odds = row['B365W'] if row['Win'] == 1 else row['B365L']
        
        # Place bet
        strategy.place_bet(
            predicted_prob=pred_prob,
            actual_outcome=row['Win'],
            odds=odds,
            date=row['Date']
        )
    
    # Calculate and print performance metrics
    metrics = strategy.get_performance_metrics()
    print("\nBetting Strategy Performance:")
    print("============================")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    strategy.plot_performance(save_path='betting_performance.png')
    
    return strategy, metrics

if __name__ == '__main__':
    # Run backtest
    strategy, metrics = backtest_strategy(
        model_path='tennis_model_fixed.joblib',
        data_path='tennis_data_features_fixed.csv',
        strategy_params={
            'initial_bankroll': 1000,
            'max_bet_pct': 2.5,
            'min_odds': 1.2
        }
    )
