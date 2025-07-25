import pandas as pd
import numpy as np
from feature_engineering_optimized import calculate_features

def test_feature_engineering():
    """Test the feature engineering pipeline."""
    print("Loading test data...")
    test_data_path = "C:/Users/Carlos/Documents/ODST/tennis_data/tennis_data.csv"
    df = pd.read_csv(test_data_path, low_memory=False, nrows=1000)  # Test with subset

    print("\nTesting feature engineering pipeline...")
    try:
        features_df = calculate_features(df)
        print("\n✓ Feature calculation completed successfully")
        
        # Test 1: Check for NaN values
        nan_cols = features_df.columns[features_df.isna().any()].tolist()
        if nan_cols:
            print("\n❌ Found NaN values in columns:", nan_cols)
        else:
            print("\n✓ No NaN values found")
        
        # Test 2: Check for infinite values
        numeric_df = features_df.select_dtypes(include=np.number)
        inf_cols = numeric_df.columns[np.isinf(numeric_df).any()].tolist()
        if inf_cols:
            print("\n❌ Found infinite values in columns:", inf_cols)
        else:
            print("\n✓ No infinite values found")
        
        # Test 3: Check feature ranges
        print("\nFeature value ranges:")
        numeric_cols = features_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            min_val = features_df[col].min()
            max_val = features_df[col].max()
            print(f"{col:30} Min: {min_val:10.2f} Max: {max_val:10.2f}")
            
        # Test 4: Check feature presence
        print("\nVerifying features...")
        required_features = ["Elo_diff", "career_win_percentage_diff", "Experience_diff"]
        missing_features = [f for f in required_features if f not in features_df.columns]
        if missing_features:
            print("\n❌ Missing required features:", missing_features)
        else:
            print("\n✓ All required features present")
        
        # Test 5: Check basic data validity
        if features_df["Win"].min() >= 0 and features_df["Win"].max() <= 1:
            print("\n✓ Win column values are valid")
        else:
            print("\n❌ Invalid values in Win column")
            
        print("\nFeature engineering validation complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during feature engineering: {str(e)}")
        return False

if __name__ == "__main__":
    test_feature_engineering()
