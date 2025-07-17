# Tennis Match Prediction Model

This project implements a machine learning model for predicting tennis match outcomes using historical match data.

## Project Structure

- `feature_engineering_optimized.py`: Original feature engineering implementation
- `feature_engineering_fixed.py`: Fixed feature engineering with proper temporal boundaries
- `train_model.py`: Original model training implementation
- `train_model_fixed.py`: Fixed model training with proper validation and no data leakage

## Model Performance

### Latest Model (Fixed Implementation)
- Average Accuracy: 66.20%
- Standard Deviation: 1.74%
- Feature Importance:
  1. Ranking difference: 56.31%
  2. Career win percentage difference: 29.26%
  3. Surface win percentage difference: 7.76%
  4. Recent form difference: 4.08%
  5. Head-to-head win percentage difference: 2.59%

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

### v1.0.0 (Current)
- Initial implementation with basic features
- Achieved 93.40% accuracy (with data leakage)

### v1.1.0 (Latest)
- Fixed data leakage issues
- Implemented proper temporal validation
- Achieved 66.20% accuracy (realistic performance)
- Added proper feature engineering with chronological boundaries
