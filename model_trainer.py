import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.features = ['Elo_diff', 'career_win_percentage_diff', 
                        'career_surface_win_percentage_diff', 
                        'h2h_win_percentage_diff', 'rolling_form_diff']
        self.target = 'Win'
        
    def prepare_fold(self, train_data, val_data):
        """Prepare data for a single fold, ensuring no data leakage"""
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[self.features])
        X_val = scaler.transform(val_data[self.features])
        
        y_train = train_data[self.target]
        y_val = val_data[self.target]
        
        return X_train, X_val, y_train, y_val
    
    def evaluate_predictions(self, y_true, y_pred, probas, odds_w, odds_l):
        """Evaluate model predictions"""
        accuracy = (y_true == (probas > 0.5)).mean()
        
        # Calculate betting results
        bet_size = 10
        bankroll = 1000
        pnl = []
        
        for idx in range(len(y_true)):
            if bankroll < bet_size:
                break
                
            prob = probas[idx]
            imp_prob_w = 1 / odds_w[idx] if odds_w[idx] > 0 else 0
            
            if prob > imp_prob_w and odds_w[idx] > 0:
                # Bet on winner
                if y_true.iloc[idx] == 1:
                    pnl.append(bet_size * (odds_w[idx] - 1))
                    bankroll += bet_size * odds_w[idx]
                else:
                    pnl.append(-bet_size)
                bankroll -= bet_size
                
        roi = (np.sum(pnl) / (len(pnl) * bet_size)) * 100 if pnl else 0
        
        return {
            'accuracy': accuracy,
            'roi': roi,
            'final_bankroll': bankroll,
            'n_bets': len(pnl)
        }
    
    def train_and_evaluate(self, n_splits=5):
        """Train and evaluate model using time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        
        # Sort by date to ensure proper time-based splits
        self.data = self.data.sort_values('Date')
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.data), 1):
            train_data = self.data.iloc[train_idx]
            val_data = self.data.iloc[val_idx]
            
            # Prepare data for this fold
            X_train, X_val, y_train, y_val = self.prepare_fold(train_data, val_data)
            
            # Train model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            probas = model.predict_proba(X_val)[:, 1]
            
            # Evaluate
            fold_results = self.evaluate_predictions(
                y_val,
                probas > 0.5,
                probas,
                val_data['B365W'].values,
                val_data['B365L'].values
            )
            
            fold_results['fold'] = fold
            fold_results['train_size'] = len(train_data)
            fold_results['val_size'] = len(val_data)
            results.append(fold_results)
            
            print(f"\nFold {fold} Results:")
            print(f"Accuracy: {fold_results['accuracy']:.4f}")
            print(f"ROI: {fold_results['roi']:.2f}%")
            print(f"Final Bankroll: ${fold_results['final_bankroll']:.2f}")
            print(f"Number of bets: {fold_results['n_bets']}")
            
        return results

if __name__ == '__main__':
    data_path = 'tennis_data_features.csv'
    trainer = ModelTrainer(data_path)
    
    print("Starting model training and evaluation...")
    results = trainer.train_and_evaluate(n_splits=5)
    
    # Calculate average metrics
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_roi = np.mean([r['roi'] for r in results])
    
    print("\nOverall Results:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average ROI: {avg_roi:.2f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print("\nResults saved to model_evaluation_results.csv")
