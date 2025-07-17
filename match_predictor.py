import streamlit as st
import pandas as pd
import numpy as np
import joblib
from feature_engineering_optimized import EloRating

# Load the trained model and preprocessed data
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('tennis_model.joblib')
        surface_encoder = joblib.load('surface_encoder.joblib')
        historical_data = pd.read_csv('tennis_data_features.csv')
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
    elo = EloRating()
    relevant_matches = historical_data[
        (historical_data['Date'] < as_of_date) &
        (historical_data['Surface'] == surface)
    ].sort_values('Date')
    
    for _, match in relevant_matches.iterrows():
        if match['Player'] == player and match['Win'] == 1:
            elo.update_rating(match['Player'], match['Opponent'])
        elif match['Opponent'] == player and match['Win'] == 0:
            elo.update_rating(match['Opponent'], match['Player'])
    
    return elo.get_rating(player)

def get_player_stats(historical_data, player, surface, as_of_date):
    player_data = historical_data[
        (historical_data['Player'] == player) & 
        (historical_data['Date'] < as_of_date)
    ].copy()
    
    # Career stats
    total_matches = len(player_data)
    if total_matches > 0:
        career_win_percentage = player_data['Win'].mean()
    else:
        career_win_percentage = 0
    
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
        return 0
    
    player1_wins = h2h_matches[
        (h2h_matches['Player'] == player1) & (h2h_matches['Win'] == 1)
    ].shape[0]
    
    return player1_wins / len(h2h_matches)

def main():
    st.title("Tennis Match Predictor")
    
    try:
        model, historical_data, surface_encoder = load_model_and_data()
        players, surfaces = get_unique_values(historical_data)
        
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
                'Elo_diff': [p1_elo - p2_elo],
                'career_win_percentage_diff': [p1_career_win_pct - p2_career_win_pct],
                'career_surface_win_percentage_diff': [p1_surface_win_pct - p2_surface_win_pct],
                'h2h_win_percentage_diff': [h2h_win_pct - (1 - h2h_win_pct)],
                'rolling_form_diff': [p1_form - p2_form],
                'Surface': surface_encoder.transform([surface])
            })
            
            # Make prediction
            win_prob = model.predict_proba(features)[0][1]
            
            # Display results
            st.subheader("Match Prediction")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"{player1} Win Probability", f"{win_prob:.1%}")
            with col2:
                st.metric(f"{player2} Win Probability", f"{(1-win_prob):.1%}")
            
            # Display player stats
            st.subheader("Player Statistics")
            stats_df = pd.DataFrame({
                'Statistic': [
                    'Elo Rating',
                    'Career Win %',
                    f'{surface} Court Win %',
                    'Recent Form (Last 10)',
                    'H2H Win %'
                ],
                player1: [
                    f'{p1_elo:.0f}',
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
            
    except FileNotFoundError as e:
        st.error("Error: Required model or data files not found. Please ensure you have trained the model first.")
        st.error(str(e))

if __name__ == "__main__":
    main()
