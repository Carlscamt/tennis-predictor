import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from datetime import datetime

class AdaptiveBettingStrategy:
    def __init__(self, initial_bankroll=1000, max_bet_pct=2.5, min_prob=0.55):
        """
        Initialize betting strategy using model probabilities.
        
        Args:
            initial_bankroll: Starting bankroll amount
            max_bet_pct: Maximum percentage of bankroll to bet (default 2.5%)
            min_prob: Minimum probability threshold for placing bets
        """
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.max_bet_pct = max_bet_pct / 100  # Convert to decimal
        self.min_prob = min_prob
        self.bets = []
        self.running_profit = []
        
    def estimate_odds(self, prob_win):
        """
        Estimate fair odds from model probability.
        
        Args:
            prob_win: Model's predicted probability of winning
            
        Returns:
            float: Estimated fair decimal odds
        """
        # Add a small margin to create realistic market odds
        margin = 0.05
        if prob_win < 0.1:  # Cap minimum probability
            prob_win = 0.1
        elif prob_win > 0.9:  # Cap maximum probability
            prob_win = 0.9
        
        # Convert probability to fair odds with margin
        fair_odds = 1 / prob_win
        market_odds = fair_odds * (1 - margin)
        return market_odds
        
    def calculate_kelly_fraction(self, prob_win):
        """
        Calculate the optimal Kelly Criterion bet size using estimated odds.
        Uses a fractional Kelly approach for more conservative betting.
        
        Args:
            prob_win: Probability of winning from model
            
        Returns:
            tuple: (optimal fraction, estimated odds)
        """
        odds = self.estimate_odds(prob_win)
        
        if prob_win * odds <= 1:  # No edge
            return 0, odds
        
        kelly = (prob_win * odds - 1) / (odds - 1)
        fractional_kelly = kelly * 0.5  # Use half Kelly for safety
        
        # Apply maximum bet size constraint
        return min(fractional_kelly, self.max_bet_pct), odds
        
    def place_bet(self, predicted_prob, actual_outcome, date):
        """
        Place a bet based on model probability and track results.
        
        Args:
            predicted_prob: Model's predicted probability
            actual_outcome: Actual match outcome (1=win, 0=loss)
            date: Date of the match
        """
        if predicted_prob < self.min_prob:
            return
            
        kelly_fraction, estimated_odds = self.calculate_kelly_fraction(predicted_prob)
        bet_amount = self.bankroll * kelly_fraction
        
        if bet_amount == 0:
            return
            
        profit = bet_amount * (estimated_odds - 1) if actual_outcome == 1 else -bet_amount
        self.bankroll += profit
        
        self.bets.append({
            'date': date,
            'prob': predicted_prob,
            'odds': estimated_odds,
            'amount': bet_amount,
            'outcome': actual_outcome,
            'profit': profit,
            'bankroll': self.bankroll
        })
        self.running_profit.append(profit)
        
    def get_performance_metrics(self):
        """Calculate performance metrics for the betting strategy."""
        if not self.bets:
            return {}
            
        df = pd.DataFrame(self.bets)
        
        # Basic metrics
        total_bets = len(df)
        winning_bets = len(df[df['outcome'] == 1])
        win_rate = winning_bets / total_bets
        
        # ROI
        total_stakes = df['amount'].sum()
        total_profit = df['profit'].sum()
        roi = (total_profit / total_stakes) * 100
        
        # Maximum drawdown
        running_bankroll = [self.initial_bankroll] + list(df['bankroll'])
        peak = self.initial_bankroll
        max_drawdown = 0
        
        for bankroll in running_bankroll:
            if bankroll > peak:
                peak = bankroll
            drawdown = (peak - bankroll) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Risk metrics
        returns = np.array(self.running_profit)
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate * 100,
            'roi': roi,
            'profit': total_profit,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
    def plot_performance(self, save_path=None):
        """Plot betting performance over time."""
        if not self.bets:
            return
            
        df = pd.DataFrame(self.bets)
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['bankroll'], label='Bankroll')
        plt.axhline(y=self.initial_bankroll, color='r', linestyle='--', label='Initial Bankroll')
        plt.title('Betting Strategy Performance')
        plt.xlabel('Number of Bets')
        plt.ylabel('Bankroll')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def backtest_strategy(
    model_path,
    data_path='tennis_data_features_fixed.csv',
    start_date='2020-01-01',
    initial_bankroll=1000,
    max_bet_pct=2.5,
    min_prob=0.55,
    metrics_to_track={
        'win_rate': True,
        'roi': True,
        'max_drawdown': True,
        'sharpe_ratio': True
    }
):
    """
    Run backtest of betting strategy on historical data using model probabilities.
    
    Args:
        model_path: Path to trained model file
        data_path: Path to test data CSV
        start_date: Start date for backtest period
        initial_bankroll: Starting bankroll amount
        max_bet_pct: Maximum percentage of bankroll to bet
        min_prob: Minimum probability threshold for placing bets
        metrics_to_track: Dict of metrics to track during backtest
        
    Returns:
        tuple: (BettingStrategy object, dict of performance metrics)
    """
    # Load model and data
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    
    # Convert date and filter for test period
    data['Date'] = pd.to_datetime(data['Date'])
    test_data = data[data['Date'] >= start_date].copy()
    
    # Initialize strategy
    strategy = AdaptiveBettingStrategy(
        initial_bankroll=initial_bankroll,
        max_bet_pct=max_bet_pct,
        min_prob=min_prob
    )
    
    # Run backtest
    for idx, row in test_data.iterrows():
        # Prepare features for prediction
        features = np.array([
            row['career_win_pct_diff'], 
            row['surface_win_pct_diff'], 
            row['recent_form_diff'], 
            row['h2h_win_pct_diff'],
            row['ranking_diff']
        ]).reshape(1, -1)
        
        # Get model prediction probability
        pred_prob = model.predict_proba(features)[0][1]
        
        # Place bet using estimated odds
        strategy.place_bet(
            predicted_prob=pred_prob,
            actual_outcome=row['Win'],
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
        start_date='2020-01-01',
        initial_bankroll=1000,
        max_bet_pct=2.5,
        min_prob=0.55
    )
