import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm

class BacktestingEngineOptimized:
    def __init__(self, data_path, initial_bankroll=1000, bet_size=10):
        self.data = pd.read_csv(data_path, low_memory=False)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.sort_values(by='Date', inplace=True)
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.bet_size = bet_size
        self.bets_history = []
        self.model = None
        self.features = ['Elo_diff', 'career_win_percentage_diff', 
                         'career_surface_win_percentage_diff', 
                         'h2h_win_percentage_diff', 'rolling_form_diff']
        self.target = 'Win'

        # Convert all feature columns to float
        for col in self.features + [self.target]:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').astype(float)
        self.data.fillna(0, inplace=True)

    def run_backtest(self, start_date='2000-01-01', end_date='2025-12-31', train_window_years=1):
        unique_dates = np.sort(self.data['Date'].unique())

        for current_date in tqdm(unique_dates, desc="Backtesting Progress"):
            current_date = pd.to_datetime(current_date)
            if current_date < pd.to_datetime(start_date):
                continue
            if current_date > pd.to_datetime(end_date):
                break

            train_end_date = current_date - pd.Timedelta(days=1)
            train_start_date = train_end_date - pd.DateOffset(years=train_window_years)
            
            train_data = self.data[(self.data['Date'] >= train_start_date) & (self.data['Date'] <= train_end_date)]

            if train_data.empty:
                continue

            X_train = train_data[self.features]
            y_train = train_data[self.target]

            self.model = xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='logloss',
                use_label_encoder=False, tree_method='hist',
                n_estimators=100, max_depth=5, learning_rate=0.05
            )
            
            # Ensure features are float type
            X_train = X_train.astype(float)
            self.model.fit(X_train, y_train)

            current_day_matches = self.data[self.data['Date'] == current_date]

            for idx, match_row in current_day_matches.iterrows():
                features_for_prediction = match_row[self.features].to_frame().T.astype(float)
                predicted_prob = self.model.predict_proba(features_for_prediction)[:, 1][0]

                winner_odds = match_row['B365W']
                loser_odds = match_row['B365L']

                if winner_odds == 0 or loser_odds == 0: continue

                implied_prob_winner = 1 / winner_odds
                bet_on_winner = predicted_prob > implied_prob_winner

                if bet_on_winner:
                    bet_odds = winner_odds
                    actual_outcome = match_row['Win']
                else:
                    bet_odds = loser_odds
                    actual_outcome = 1 - match_row['Win']

                if self.bankroll >= self.bet_size:
                    self.bankroll -= self.bet_size
                    pnl = (self.bet_size * (bet_odds - 1)) if actual_outcome == 1 else -self.bet_size
                    if actual_outcome == 1:
                        self.bankroll += self.bet_size * bet_odds
                    
                    self.bets_history.append({
                        'Date': current_date, 'Player': match_row['Player'], 'Opponent': match_row['Opponent'],
                        'Predicted_Prob': predicted_prob, 'Implied_Prob': implied_prob_winner, 'Bet_On_Winner': bet_on_winner,
                        'Bet_Size': self.bet_size, 'Bet_Odds': bet_odds, 'Actual_Outcome': actual_outcome,
                        'PnL': pnl, 'Bankroll': self.bankroll
                    })
                else:
                    print("Bankroll depleted.")
                    break
            if self.bankroll < self.bet_size:
                break

        print("\nBacktesting complete.")
        print(f"Initial Bankroll: {self.initial_bankroll}")
        print(f"Final Bankroll: {self.bankroll:.2f}")
        total_profit = self.bankroll - self.initial_bankroll
        total_wagered = len(self.bets_history) * self.bet_size
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        print(f"Total Profit: {total_profit:.2f}")
        print(f"Total Wagered: {total_wagered}")
        print(f"ROI: {roi:.2f}%")
        
        # Calculate prediction accuracy
        bets_df = pd.DataFrame(self.bets_history)
        if not bets_df.empty:
            accuracy = (bets_df['Actual_Outcome'] == 1).mean()
            print(f"Prediction Accuracy: {accuracy:.2%}")
            
            # Monthly returns analysis
            bets_df['Month'] = bets_df['Date'].dt.to_period('M')
            monthly_roi = bets_df.groupby('Month').agg({
                'PnL': 'sum',
                'Bet_Size': 'count'
            }).assign(ROI=lambda x: (x['PnL'] / (x['Bet_Size'] * self.bet_size)) * 100)
            
            print("\nMonthly Performance:")
            print(monthly_roi.round(2))
            
            return accuracy, roi, monthly_roi
        return None, None, None

if __name__ == '__main__':
    data_path = 'C:/Users/Carlos/Documents/ODST/tennis_data_features.csv'
    engine = BacktestingEngineOptimized(data_path, initial_bankroll=1000, bet_size=10)
    
    # Run backtest for year 2023 using 3 years of training data
    print("\nRunning backtest for 2023...")
    engine.run_backtest(start_date='2023-01-01', end_date='2023-12-31', train_window_years=3)

if __name__ == '__main__':
    data_path = 'C:/Users/Carlos/Documents/ODST/tennis_data_features.csv'
    engine = BacktestingEngineOptimized(data_path, initial_bankroll=1000, bet_size=10)
    engine.run_backtest(start_date='2023-01-01', end_date='2023-12-31', train_window_years=3)
