
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

class EloRating:
    def __init__(self, k=30, g=20, initial_rating=1500):
        self.k = k
        self.g = g
        self.initial_rating = initial_rating
        self.ratings = {}

    def get_rating(self, player):
        return self.ratings.get(player, self.initial_rating)

    def update_rating(self, winner, loser):
        winner_rating = self.get_rating(winner)
        loser_rating = self.get_rating(loser)

        expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))

        new_winner_rating = winner_rating + self.k * (1 - expected_winner)
        new_loser_rating = loser_rating + self.k * (0 - expected_loser)

        self.ratings[winner] = new_winner_rating
        self.ratings[loser] = new_loser_rating

def backtest_model(data_path, train_year_end, test_year_start):
    """Trains an XGBoost model on data up to train_year_end and tests on data from test_year_start."""
    df = pd.read_csv(data_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df.fillna(0, inplace=True)
    df.sort_values(by='Date', inplace=True)

    # Elo Ratings
    elo_calculators = {
        'Hard': EloRating(),
        'Clay': EloRating(),
        'Grass': EloRating(),
        'Carpet': EloRating(),
        'Other': EloRating()
    }
    df['Player_Elo'] = 0
    df['Opponent_Elo'] = 0
    for index, row in df.iterrows():
        surface = row['Surface'] if row['Surface'] in elo_calculators else 'Other'
        elo_calc = elo_calculators[surface]
        player_elo = elo_calc.get_rating(row['Player'])
        opponent_elo = elo_calc.get_rating(row['Opponent'])
        df.at[index, 'Player_Elo'] = player_elo
        df.at[index, 'Opponent_Elo'] = opponent_elo
        if row['Win'] == 1:
            elo_calc.update_rating(row['Player'], row['Opponent'])

    # Career Win Percentage
    df['career_win_percentage'] = df.groupby('Player')['Win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)

    # Career Surface Win Percentage
    df['career_surface_win_percentage'] = df.groupby(['Player', 'Surface'])['Win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)

    # H2H Win Percentage
    df['h2h_win_percentage'] = df.groupby(['Player', 'Opponent'])['Win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)

    # Rolling Form
    df['rolling_form'] = df.groupby('Player')['Win'].transform(lambda x: x.shift().rolling(10, min_periods=1).mean()).fillna(0)

    # Create differential features
    df['Elo_diff'] = df['Player_Elo'] - df['Opponent_Elo']
    df['career_win_percentage_diff'] = df['career_win_percentage'] - df.groupby('Opponent')['Win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['career_surface_win_percentage_diff'] = df['career_surface_win_percentage'] - df.groupby(['Opponent', 'Surface'])['Win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['h2h_win_percentage_diff'] = df['h2h_win_percentage'] - df.groupby(['Opponent', 'Player'])['Win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['rolling_form_diff'] = df['rolling_form'] - df.groupby('Opponent')['Win'].transform(lambda x: x.shift().rolling(10, min_periods=1).mean()).fillna(0)

    # Select features and target
    features = ['Elo_diff', 'career_win_percentage_diff', 'career_surface_win_percentage_diff', 'h2h_win_percentage_diff', 'rolling_form_diff']
    target = 'Win' 
    
    # Chronological split for backtesting
    train_df = df[df['Date'].dt.year <= train_year_end]
    test_df = df[df['Date'].dt.year == test_year_start]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }

    # Initialize the XGBoost model with GPU parameters
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method='gpu_hist',
        predictor='gpu_predictor'
    )

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Use the best model for predictions and evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("Model Evaluation for {test_year_start} data:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

def train_and_save_model():
    """Trains and saves an XGBoost model on the tennis data."""
    print("Loading data...")
    data = pd.read_csv('tennis_data_features.csv', low_memory=False)
    data['Date'] = pd.to_datetime(data['Date'])
    data.fillna(0, inplace=True)
    data.sort_values(by='Date', inplace=True)

    features = [
        # Base features
        'Elo_diff',
        'Surface',
        'career_win_percentage_diff',
        'career_surface_win_percentage_diff',
        'h2h_win_percentage_diff',
        'rolling_form_diff',
        'Momentum_diff',
        'Experience_diff',
        'Surface_Experience_diff',
        'Surface_Form_diff',
        'Weighted_Form_diff',
        'Tournament_Level',
        'Ranking_Diff',
        # New features
        'Round_Progress',
        'Best_of_Performance_diff',
        'Court_Performance_diff',
        'Score_Dominance',
        'Points_Ratio',
        'Dominance_diff'
    ]
    
    # Encode surface categorical variable
    le = LabelEncoder()
    data['Surface'] = le.fit_transform(data['Surface'])
    
    print("Preparing features...")
    X = data[features]
    y = data['Win']
    
    # Use TimeSeriesSplit for proper temporal validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize model with optimized parameters for expanded feature set
    model = xgb.XGBClassifier(
        n_estimators=500,  # Increased for more complex feature interactions
        learning_rate=0.03,  # Reduced to prevent overfitting with more features
        max_depth=6,  # Slightly increased to capture more complex patterns
        min_child_weight=3,  # Increased to prevent overfitting
        subsample=0.85,
        colsample_bytree=0.85,  # Adjusted for more features
        gamma=0.2,  # Increased to encourage more conservative tree splitting
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
    for _, row in importance.head(10).iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    print("\nSaving model...")
    # Save the model and label encoder
    joblib.dump(model, 'tennis_model_with_new_features.joblib')
    joblib.dump(le, 'surface_encoder.joblib')
    
    # Save feature importance for future reference
    importance.to_csv('feature_importance.csv', index=False)
    print("Model and feature importance saved successfully!")

if __name__ == '__main__':
    train_and_save_model()
