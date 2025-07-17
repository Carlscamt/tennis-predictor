import streamlit as st
import pandas as pd
import numpy as np
import joblib
from feature_engineering_optimized import EloRating
from betting_strategy_v2 import AdaptiveBettingStrategy

# Load the trained model and preprocessed data
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('tennis_model_fixed.joblib')  # Using our fixed model
        surface_encoder = joblib.load('surface_encoder.joblib')
        historical_data = pd.read_csv('tennis_data_features_fixed.csv')  # Using fixed features
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        return model, historical_data, surface_encoder
    except FileNotFoundError:
        st.error("Model or data files not found. Running model training first...")
        import train_model
        train_model.train_and_save_model()
        model = joblib.load('tennis_model.joblib')
        surface_encoder = joblib.load('surface_encoder.joblib')
        historical_data = pd.read_csv('tennis_data_features.csv')
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        return model, historical_data, surface_encoder

# Get unique players and surfaces
@st.cache_data
def get_unique_values(historical_data):
    players = sorted(list(set(historical_data['Player'].unique()) | set(historical_data['Opponent'].unique())))
    surfaces = sorted(historical_data['Surface'].unique())
    return players, surfaces

def calculate_elo_rating(historical_data, player, surface, as_of_date):
    # Initialize with reasonable parameters for tennis
    elo = EloRating(k=32, initial_rating=1500, decay_factor=0.97)
    
    # Get all relevant matches before the given date
    relevant_matches = historical_data[
        (historical_data['Date'] < as_of_date) &
        (historical_data['Surface'] == surface)
    ].sort_values('Date')
    
    # Process each match to update Elo ratings
    for _, match in relevant_matches.iterrows():
        # Only process matches where the player participated
        if match['Player'] == player or match['Opponent'] == player:
            winner = match['Player'] if match['Win'] == 1 else match['Opponent']
            loser = match['Opponent'] if match['Win'] == 1 else match['Player']
            
            # Convert date string to datetime if needed
            match_date = pd.to_datetime(match['Date']) if isinstance(match['Date'], str) else match['Date']
            
            # Calculate margin of victory if sets information is available
            if 'Wsets' in match.index and 'Lsets' in match.index:
                try:
                    margin = float(match['Wsets']) / (float(match['Wsets']) + float(match['Lsets']))
                except (ValueError, ZeroDivisionError):
                    margin = 1.0
            else:
                margin = 1.0
                
            elo.update_rating(winner, loser, match_date, margin_of_victory=margin)
    
    # Return the player's current Elo rating with decay
    return elo.get_rating(player, current_date=as_of_date)

def get_player_stats(historical_data, player, surface, as_of_date):
    # Get matches where player appears as either Player or Opponent
    player_matches = historical_data[
        ((historical_data['Player'] == player) | (historical_data['Opponent'] == player)) &
        (historical_data['Date'] < as_of_date)
    ].copy()
    
    # Career stats
    total_matches = len(player_matches)
    if total_matches > 0:
        wins = len(player_matches[
            ((player_matches['Player'] == player) & (player_matches['Win'] == 1)) |
            ((player_matches['Opponent'] == player) & (player_matches['Win'] == 0))
        ])
        career_win_percentage = wins / total_matches
        
        # Surface stats
        surface_matches = player_matches[player_matches['Surface'] == surface]
        surface_total = len(surface_matches)
        if surface_total > 0:
            surface_wins = len(surface_matches[
                ((surface_matches['Player'] == player) & (surface_matches['Win'] == 1)) |
                ((surface_matches['Opponent'] == player) & (surface_matches['Win'] == 0))
            ])
            surface_win_percentage = surface_wins / surface_total
        else:
            surface_win_percentage = career_win_percentage  # Use career stats if no surface data
        
        # Recent form
        recent_matches = player_matches.sort_values('Date', ascending=False).head(10)
        if len(recent_matches) > 0:
            recent_wins = len(recent_matches[
                ((recent_matches['Player'] == player) & (recent_matches['Win'] == 1)) |
                ((recent_matches['Opponent'] == player) & (recent_matches['Win'] == 0))
            ])
            recent_form = recent_wins / len(recent_matches)
        else:
            recent_form = career_win_percentage  # Use career stats if no recent matches
    else:
        # Default values for new players
        career_win_percentage = 0.5
        surface_win_percentage = 0.5
        recent_form = 0.5
    
    # Calculate Elo rating
    elo_rating = calculate_elo_rating(historical_data, player, surface, as_of_date)
    
    return career_win_percentage, surface_win_percentage, recent_form, elo_rating
    
    # Surface stats
    surface_data = player_data[player_data['Surface'] == surface]
    surface_matches = len(surface_data)
    if surface_matches > 0:
        surface_win_percentage = surface_data['Win'].mean()
    else:
        surface_win_percentage = 0
    
    # Recent form (last 10 matches)
    recent_matches = player_data.sort_values('Date', ascending=False).head(10)
    if len(recent_matches) > 0:
        recent_form = recent_matches['Win'].mean()
    else:
        recent_form = 0
    
    # Calculate Elo rating
    elo_rating = calculate_elo_rating(historical_data, player, surface, as_of_date)
    
    return career_win_percentage, surface_win_percentage, recent_form, elo_rating

def get_h2h_stats(historical_data, player1, player2, as_of_date):
    h2h_matches = historical_data[
        (historical_data['Date'] < as_of_date) &
        (
            ((historical_data['Player'] == player1) & (historical_data['Opponent'] == player2)) |
            ((historical_data['Player'] == player2) & (historical_data['Opponent'] == player1))
        )
    ].copy()
    
    if len(h2h_matches) == 0:
        # Default to 50-50 if no head-to-head history
        return 0.5
    
    # Count wins for player1 (both when player1 is Player or Opponent)
    player1_wins = len(h2h_matches[
        ((h2h_matches['Player'] == player1) & (h2h_matches['Win'] == 1)) |
        ((h2h_matches['Opponent'] == player1) & (h2h_matches['Win'] == 0))
    ])
    
    return player1_wins / len(h2h_matches)

def main():
    st.title("Tennis Match Predictor")
    
    try:
        model, historical_data, surface_encoder = load_model_and_data()
        players, surfaces = get_unique_values(historical_data)
        
        # Initialize betting strategy
        betting_strategy = AdaptiveBettingStrategy()
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Player 1")
                # Search box for Player 1
                search_player1 = st.text_input("Search Player 1", key="search_player1").lower()
                filtered_players1 = [p for p in players if search_player1 in p.lower()] if search_player1 else players
                player1 = st.selectbox("Select Player 1", filtered_players1, key="player1")
            
            with col2:
                st.subheader("Player 2")
                # Search box for Player 2
                search_player2 = st.text_input("Search Player 2", key="search_player2").lower()
                filtered_players2 = [p for p in players if search_player2 in p.lower()] if search_player2 else players
                player2 = st.selectbox("Select Player 2", filtered_players2, key="player2")
            
            surface = st.selectbox("Select Surface", surfaces)
            
            submit = st.form_submit_button("Predict Match")
        
        if submit:
            if player1 == player2:
                st.error("Please select different players")
                return
            
            # Current date for calculating recent stats
            current_date = pd.Timestamp.now()
            
            # Get player stats
            p1_career_win_pct, p1_surface_win_pct, p1_form, p1_elo = get_player_stats(
                historical_data, player1, surface, current_date
            )
            p2_career_win_pct, p2_surface_win_pct, p2_form, p2_elo = get_player_stats(
                historical_data, player2, surface, current_date
            )
            
            # Get H2H stats
            h2h_win_pct = get_h2h_stats(historical_data, player1, player2, current_date)
            
            # Create feature vector
            features = pd.DataFrame({
                'career_win_pct_diff': [p1_career_win_pct - p2_career_win_pct],
                'surface_win_pct_diff': [p1_surface_win_pct - p2_surface_win_pct],
                'recent_form_diff': [p1_form - p2_form],
                'h2h_win_pct_diff': [h2h_win_pct - (1 - h2h_win_pct)],
                'ranking_diff': [p1_elo - p2_elo]  # Using Elo as a proxy for ranking
            })
            
            # Make prediction
            feature_cols = [
                'career_win_pct_diff',
                'surface_win_pct_diff',
                'recent_form_diff',
                'h2h_win_pct_diff',
                'ranking_diff'
            ]
            win_prob = model.predict_proba(features[feature_cols])[0][1]
            
            # Display results
            st.subheader("Match Prediction")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"{player1} Win Probability", f"{win_prob:.1%}")
            with col2:
                st.metric(f"{player2} Win Probability", f"{(1-win_prob):.1%}")
                
                # Betting Strategy Section
            st.subheader("Betting Analysis")
            
            # Initialize betting strategy for this prediction
            strategy = AdaptiveBettingStrategy()
            
            # Add odds input
            use_odds = st.checkbox("Input bookmaker odds")
            
            if use_odds:
                col1, col2 = st.columns(2)
                with col1:
                    p1_odds = st.number_input(f"{player1} Odds", min_value=1.0, value=2.0, step=0.1)
                with col2:
                    p2_odds = st.number_input(f"{player2} Odds", min_value=1.0, value=2.0, step=0.1)
                    
                # Calculate betting recommendation
                kelly_fraction, fair_odds = strategy.calculate_kelly_fraction(win_prob)
                recommended_bet = kelly_fraction * strategy.bankroll
                
                if win_prob > strategy.min_prob:
                    if win_prob > 1/p1_odds:  # Check if there's value
                        expected_value = recommended_bet * (win_prob * (p1_odds-1) - (1-win_prob))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"✓ Bet on {player1}")
                            st.metric("Recommended Stake", f"${recommended_bet:.2f}")
                        with col2:
                            st.metric("Model Edge", f"{((1/p1_odds - win_prob) * 100):.1f}%")
                            st.metric("Expected Value", f"${expected_value:.2f}")
                    else:
                        st.warning("⚠️ No value at current odds")
                        st.info(f"Need minimum odds of {(1/win_prob):.2f} to bet")
                else:
                    st.warning("⚠️ Win probability below threshold (55%)")
            else:
                # Use estimated odds from betting strategy
                kelly_fraction, fair_odds = strategy.calculate_kelly_fraction(win_prob)
                recommended_bet = kelly_fraction * strategy.bankroll
                
                if win_prob > strategy.min_prob:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"✓ Bet on {player1}")
                        st.metric("Recommended Stake", f"${recommended_bet:.2f}")
                    with col2:
                        st.metric("Estimated Fair Odds", f"{fair_odds:.2f}")
                        st.metric("Implied Probability", f"{(1/fair_odds):.1%}")
                else:
                    st.warning("⚠️ Win probability below threshold (55%)")            # Display player stats
            st.subheader("Player Statistics")
            
            # Calculate Elo difference and interpret it
            elo_diff = p1_elo - p2_elo
            elo_advantage = player1 if elo_diff > 0 else player2
            elo_exp_win = 1 / (1 + 10 ** (-abs(elo_diff) / 400))
            
            stats_df = pd.DataFrame({
                'Statistic': [
                    'Elo Rating (Surface)',
                    'Career Win Rate',
                    f'{surface} Court Win Rate',
                    'Recent Form (Last 10)',
                    'Head-to-Head Win Rate'
                ],
                player1: [
                    f'{p1_elo:.0f} ({"+" if elo_diff > 0 else ""}{elo_diff:.0f})',
                    f'{p1_career_win_pct:.1%}',
                    f'{p1_surface_win_pct:.1%}',
                    f'{p1_form:.1%}',
                    f'{h2h_win_pct:.1%}'
                ],
                player2: [
                    f'{p2_elo:.0f}',
                    f'{p2_career_win_pct:.1%}',
                    f'{p2_surface_win_pct:.1%}',
                    f'{p2_form:.1%}',
                    f'{(1-h2h_win_pct):.1%}'
                ]
            })
            
            st.table(stats_df)
            
            # Add Elo interpretation
            if abs(elo_diff) > 100:
                st.info(f"📊 Based on Elo ratings alone, {elo_advantage} would have a {elo_exp_win:.1%} chance of winning")
            
    except FileNotFoundError as e:
        st.error("Error: Required model or data files not found. Please ensure you have trained the model first.")
        st.error(str(e))

if __name__ == "__main__":
    main()
