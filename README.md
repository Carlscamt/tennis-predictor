
# Tennis Match Prediction and Betting System

An advanced machine learning system for predicting tennis match outcomes and providing intelligent betting recommendations. The system uses historical match data, player statistics, and adaptive betting strategies to make informed predictions and betting decisions.

## Features

- **Match Prediction**
  - Advanced machine learning model with 66.20% accuracy
  - Proper temporal validation with no data leakage
  - Surface-specific performance analysis
  - Head-to-head statistics consideration

- **Player Analysis**
  - Elo rating system with temporal decay
  - Surface-specific performance tracking
  - Recent form evaluation
  - Career statistics analysis

- **Advanced Betting Strategy**
  - Uncertainty-Adjusted Kelly Criterion
  - Dynamic position sizing based on prediction confidence
  - Risk factor customization
  - Uncertainty shrinkage for conservative betting
  - Real-time bankroll management
  - Market odds integration and value analysis

## Project Structure

### Core Components
- `match_predictor.py`: Interactive UI for match predictions and betting with uncertainty analysis
- `betting_strategy_uncertainty.py`: Uncertainty-adjusted betting strategy implementation
- `feature_engineering_fixed.py`: Feature engineering with temporal boundaries and Elo ratings
- `train_model_fixed.py`: Model training with temporal validation

### Support Files
- `surface_encoder.joblib`: Surface encoding for predictions
- `tennis_model_fixed.joblib`: Trained prediction model
- `tennis_data_features_fixed.csv`: Processed match data

### Utilities
- `backtest_2025_uncertainty.py`: Uncertainty-aware backtesting engine
- `backtest_2025.py`: Basic backtesting implementation
- `test_features.py`: Feature validation
- `model_trainer.py`: Model training utilities

## Installation

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Download tennis data to `tennis_data/` directory
4. Run `feature_engineering_fixed.py` to process data
5. Train model with `train_model_fixed.py`

## Usage

1. Start the prediction UI:
   ```bash
   streamlit run match_predictor.py
   ```

2. Match Prediction:
   - Select players and surface
   - Review head-to-head statistics and surface performance
   - Get win probability predictions

3. Betting Analysis:
   - Input current market odds
   - Adjust bankroll and risk settings
   - Review uncertainty analysis
   - Get uncertainty-adjusted betting recommendations

4. Risk Management:
   - View confidence levels (High/Medium/Low)
   - Check uncertainty-adjusted expected value
   - Follow color-coded betting recommendations
   - Use risk factor adjustment for personalization

## Version History

- v1.3.0 (Latest): 
  - Added uncertainty-adjusted betting strategy
  - Implemented backtesting with uncertainty analysis
  - Enhanced UI with risk management features
  - ROI: 39.97% in backtesting (2025 data)
- v1.2.0: Added basic betting strategy and player analysis
- v1.1.0: Fixed implementation with proper temporal validation
- v1.0.0: Initial release

## Note

This system is for educational purposes only. Please gamble responsibly and follow your local regulations regarding sports betting.

## Model Performance

### Latest Model Performance (v1.3.0)
- Prediction Accuracy: 63.98% (2025 validation)
- Win Rate: 67.57% (with uncertainty filtering)
- ROI: 39.97% (backtested on 2025 data)
- Average Bet Size: 4.02% of bankroll

### Feature Importance
1. Ranking difference: 56.31%
2. Career win percentage difference: 29.26%
3. Surface win percentage difference: 7.76%
4. Recent form difference: 4.08%
5. Head-to-head win percentage difference: 2.59%

### Uncertainty Management
- Average Uncertainty: 62.14%
- Uncertainty Threshold: 0.4
- Shrinkage Factor: 0.25
- Minimum Win Probability: 0.60

### Key Improvements
1. Implemented strict chronological feature calculation
2. Removed post-match features that caused data leakage
3. Added proper handling of time boundaries
4. Improved handling of rankings (NR cases)
5. Used more conservative model parameters

## Usage

1. Generate features:
```bash
python feature_engineering_fixed.py
```

2. Train the model:
```bash
python train_model_fixed.py
```

## Data

The model uses historical tennis match data including:
- Player rankings
- Match results
- Surface types
- Tournament information
- Head-to-head statistics

## Version History

### v1.3.0 (Latest)
- Implemented uncertainty-based betting strategy
- Added temporal decay to Elo ratings
- Enhanced UI with risk management features
- Achieved 63.98% accuracy with 39.97% ROI
- Proper uncertainty quantification and management

### v1.2.0
- Added basic Kelly Criterion betting
- Enhanced player analysis with Elo ratings
- Improved feature engineering

### v1.1.0
- Fixed data leakage issues
- Implemented proper temporal validation
- Achieved 66.20% accuracy (realistic performance)
- Added proper feature engineering

### v1.0.0
- Initial implementation with basic features
- Basic match prediction capability
- Had data leakage issues (93.40% accuracy)
