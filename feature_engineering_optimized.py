import pandas as pd
import numpy as np
from tqdm import tqdm
import os

class EloRating:
    def __init__(self, k=30, initial_rating=1500, decay_factor=0.95):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}
        self.last_match_date = {}  # Track last match date for decay
        self.decay_factor = decay_factor  # Rating decay for inactivity

    def get_rating(self, player, current_date=None):
        rating = self.ratings.get(player, self.initial_rating)
        
        if current_date and player in self.last_match_date:
            # Apply rating decay based on inactivity
            days_inactive = (current_date - self.last_match_date[player]).days
            decay_months = days_inactive / 30.0  # Convert to months
            rating *= self.decay_factor ** decay_months
        
        return rating

    def update_rating(self, winner, loser, date, margin_of_victory=1.0):
        winner_rating = self.get_rating(winner, date)
        loser_rating = self.get_rating(loser, date)
        
        # Calculate expected probability
        expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        
        # Adjust K factor based on margin of victory and ratings difference
        k_factor = self.k * margin_of_victory
        rating_diff = abs(winner_rating - loser_rating)
        if rating_diff > 400:
            k_factor *= 0.8  # Reduce impact for mismatched players
        
        # Update ratings with adjusted K factor
        new_winner_rating = winner_rating + k_factor * (1 - expected_winner)
        new_loser_rating = loser_rating - k_factor * (1 - expected_winner)
        
        # Update ratings and last match dates
        self.ratings[winner] = new_winner_rating
        self.ratings[loser] = new_loser_rating
        self.last_match_date[winner] = date
        self.last_match_date[loser] = date

def create_dyadic_dataset(data_path, output_path):
    """Creates a dyadic dataset from the original match data."""
    df = pd.read_csv(data_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(subset=['Winner', 'Loser'], inplace=True)
    df.sort_values(by='Date', inplace=True)

    p1_data = df.copy()
    p1_data['Player'] = p1_data['Winner']
    p1_data['Opponent'] = p1_data['Loser']
    p1_data['Win'] = 1

    p2_data = df.copy()
    p2_data['Player'] = p2_data['Loser']
    p2_data['Opponent'] = p2_data['Winner']
    p2_data['Win'] = 0

    dyadic_df = pd.concat([p1_data, p2_data], ignore_index=True)
    dyadic_df['match_id'] = dyadic_df.groupby(['Date', 'Winner', 'Loser']).ngroup()
    dyadic_df.sort_values(by=['Date', 'match_id', 'Win'], inplace=True)
    
    dyadic_df.to_csv(output_path, index=False)
    print(f"Dyadic dataset created and saved to {output_path}")
    return dyadic_df

def calculate_features(original_df):
    print("Calculating features...")
    original_df['Date'] = pd.to_datetime(original_df['Date'])
    original_df.sort_values(by='Date', inplace=True)
    original_df.fillna(0, inplace=True)

    # Precompute Elo on original matches (not dyadic)
    match_df = original_df[['Date', 'Winner', 'Loser', 'Surface']].copy()
    match_df = match_df.drop_duplicates()  # Ensure unique matches
    match_df.sort_values('Date', inplace=True)

    elo_calculators = {
        'Hard': EloRating(), 'Clay': EloRating(), 'Grass': EloRating(),
        'Carpet': EloRating(), 'Other': EloRating()
    }
    
    # Store pre-match Elo for all players
    pre_elo = {}
    surface_counts = {}  # Track matches played on each surface
    momentum_window = 90  # Days to consider for momentum
    
    # Initialize experience tracking
    experience = {}
    surface_experience = {}

    for idx, row in tqdm(match_df.iterrows(), total=match_df.shape[0], desc="Precomputing Elo Ratings"):
        date = row['Date']
        surface = row['Surface'] if row['Surface'] in elo_calculators else 'Other'
        elo = elo_calculators[surface]
        
        # Update experience counts
        for player in [row['Winner'], row['Loser']]:
            experience[player] = experience.get(player, 0) + 1
            if player not in surface_experience:
                surface_experience[player] = {s: 0 for s in elo_calculators.keys()}
            surface_experience[player][surface] += 1
        
        # Calculate momentum (recent win rate)
        recent_matches = match_df[
            (match_df['Date'] > date - pd.Timedelta(days=momentum_window)) &
            (match_df['Date'] < date)
        ]
        
        winner_momentum = len(recent_matches[recent_matches['Winner'] == row['Winner']]) / max(1, len(recent_matches[
            (recent_matches['Winner'] == row['Winner']) | (recent_matches['Loser'] == row['Winner'])
        ]))
        
        loser_momentum = len(recent_matches[recent_matches['Winner'] == row['Loser']]) / max(1, len(recent_matches[
            (recent_matches['Winner'] == row['Loser']) | (recent_matches['Loser'] == row['Loser'])
        ]))
        
        # Get pre-update ratings with time decay
        winner_pre = elo.get_rating(row['Winner'], date)
        loser_pre = elo.get_rating(row['Loser'], date)
        
        # Store comprehensive stats
        pre_elo[(date, row['Winner'], row['Loser'])] = {
            'winner_elo': winner_pre,
            'loser_elo': loser_pre,
            'winner_momentum': winner_momentum,
            'loser_momentum': loser_momentum,
            'winner_experience': experience[row['Winner']],
            'loser_experience': experience[row['Loser']],
            'winner_surface_exp': surface_experience[row['Winner']][surface],
            'loser_surface_exp': surface_experience[row['Loser']][surface]
        }
        
        # Calculate margin of victory based on sets won/lost
        winner_sets = row.get('Wsets', 0)
        loser_sets = row.get('Lsets', 0)
        if winner_sets > 0 or loser_sets > 0:
            # Calculate margin based on set differential
            set_diff = winner_sets - loser_sets
            margin = 1.0 + (0.1 * set_diff)  # Adjust margin by 10% per set difference
        else:
            margin = 1.0
        
        # Update Elo with the calculated margin
        elo.update_rating(row['Winner'], row['Loser'], date, margin)

    # Create dyadic dataset
    p1_data = original_df.copy()
    p1_data['Player'] = p1_data['Winner']
    p1_data['Opponent'] = p1_data['Loser']
    p1_data['Win'] = 1

    p2_data = original_df.copy()
    p2_data['Player'] = p2_data['Loser']
    p2_data['Opponent'] = p2_data['Winner']
    p2_data['Win'] = 0

    dyadic_df = pd.concat([p1_data, p2_data], ignore_index=True)
    dyadic_df['match_id'] = dyadic_df.groupby(['Date', 'Winner', 'Loser']).ngroup()
    dyadic_df.sort_values(by=['Date', 'match_id', 'Win'], inplace=True)

    # Map pre-match stats back to dyadic df
    def get_player_stats(row, is_winner):
        stats = pre_elo.get((row['Date'], row['Winner'], row['Loser']), {})
        if is_winner:
            return (
                stats.get('winner_elo', 1500),
                stats.get('winner_momentum', 0),
                stats.get('winner_experience', 0),
                stats.get('winner_surface_exp', 0)
            )
        else:
            return (
                stats.get('loser_elo', 1500),
                stats.get('loser_momentum', 0),
                stats.get('loser_experience', 0),
                stats.get('loser_surface_exp', 0)
            )
    
    # Apply the function to get all stats
    player_stats = dyadic_df.apply(
        lambda x: get_player_stats(x, x['Player'] == x['Winner']),
        axis=1
    )
    opponent_stats = dyadic_df.apply(
        lambda x: get_player_stats(x, x['Player'] != x['Winner']),
        axis=1
    )
    
    # Unpack the stats into separate columns
    dyadic_df['Player_Elo'] = player_stats.apply(lambda x: x[0])
    dyadic_df['Player_Momentum'] = player_stats.apply(lambda x: x[1])
    dyadic_df['Player_Experience'] = player_stats.apply(lambda x: x[2])
    dyadic_df['Player_Surface_Exp'] = player_stats.apply(lambda x: x[3])
    
    dyadic_df['Opponent_Elo'] = opponent_stats.apply(lambda x: x[0])
    dyadic_df['Opponent_Momentum'] = opponent_stats.apply(lambda x: x[1])
    dyadic_df['Opponent_Experience'] = opponent_stats.apply(lambda x: x[2])
    dyadic_df['Opponent_Surface_Exp'] = opponent_stats.apply(lambda x: x[3])

    # --- Vectorized Feature Calculations on dyadic_df ---

    # Career stats (prior matches only)
    dyadic_df['career_matches_prior'] = dyadic_df.groupby('Player').cumcount()
    dyadic_df['career_wins_prior'] = dyadic_df.groupby('Player')['Win'].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    dyadic_df['player_career_win_percentage'] = (dyadic_df['career_wins_prior'] / dyadic_df['career_matches_prior'].replace(0, np.nan)).fillna(0)

    # Calculate opponent career stats before merging
    opponent_stats = dyadic_df[['Date', 'Player', 'player_career_win_percentage']].copy()
    opponent_stats = opponent_stats.rename(columns={
        'Player': 'Opponent', 
        'player_career_win_percentage': 'opponent_career_win_percentage_val'
    })
    
    # Merge opponent stats using time-aware merge
    dyadic_df = pd.merge_asof(
        dyadic_df.sort_values('Date'),
        opponent_stats.sort_values('Date'),
        by='Opponent',
        on='Date',
        direction='backward'
    )
    dyadic_df['opponent_career_win_percentage_val'] = dyadic_df['opponent_career_win_percentage_val'].fillna(0)

    # Career Surface Win Percentage (prior matches only)
    dyadic_df['surface_matches_prior'] = dyadic_df.groupby(['Player', 'Surface']).cumcount()
    dyadic_df['surface_wins_prior'] = dyadic_df.groupby(['Player', 'Surface'])['Win'].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    dyadic_df['player_career_surface_win_percentage'] = (dyadic_df['surface_wins_prior'] / dyadic_df['surface_matches_prior'].replace(0, np.nan)).fillna(0)

    # Calculate opponent surface stats before merging
    opponent_surface_stats = dyadic_df[['Date', 'Player', 'Surface', 'player_career_surface_win_percentage']].copy()
    opponent_surface_stats.rename(columns={
        'Player': 'Opponent',
        'player_career_surface_win_percentage': 'opponent_career_surface_win_percentage_val'
    }, inplace=True)

    # Merge opponent surface stats using time-aware merge
    dyadic_df = pd.merge_asof(
        dyadic_df.sort_values('Date'),
        opponent_surface_stats.sort_values('Date'),
        by=['Opponent', 'Surface'],
        on='Date',
        direction='backward'
    )
    dyadic_df['opponent_career_surface_win_percentage_val'] = dyadic_df['opponent_career_surface_win_percentage_val'].fillna(0)

    # Calculate prior H2H stats
    dyadic_df['h2h_matches_prior'] = dyadic_df.groupby(['Player', 'Opponent']).cumcount()  # Matches before current
    dyadic_df['h2h_wins_prior'] = dyadic_df.groupby(['Player', 'Opponent'])['Win'].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    dyadic_df['h2h_win_percentage'] = (dyadic_df['h2h_wins_prior'] / dyadic_df['h2h_matches_prior'].replace(0, np.nan)).fillna(0)
    
    # Calculate H2H differential
    dyadic_df['h2h_win_percentage_diff'] = dyadic_df['h2h_win_percentage'] - (1 - dyadic_df['h2h_win_percentage'])

    # Rolling Form (using shift to exclude current match)
    dyadic_df['player_rolling_form'] = dyadic_df.groupby('Player')['Win'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    dyadic_df['player_rolling_form'] = dyadic_df['player_rolling_form'].fillna(0)

    # Calculate opponent rolling form and merge using time-aware merge
    opponent_rolling_form_stats = dyadic_df[['Date', 'Player', 'player_rolling_form']].copy()
    opponent_rolling_form_stats = opponent_rolling_form_stats.rename(
        columns={'Player': 'Opponent', 'player_rolling_form': 'opponent_rolling_form_val'}
    )
    dyadic_df = pd.merge_asof(
        dyadic_df.sort_values('Date'),
        opponent_rolling_form_stats.sort_values('Date'),
        by='Opponent',
        on='Date',
        direction='backward'
    )
    dyadic_df['opponent_rolling_form_val'] = dyadic_df['opponent_rolling_form_val'].fillna(0)

    # Create differential features
    dyadic_df['Elo_diff'] = dyadic_df['Player_Elo'] - dyadic_df['Opponent_Elo']
    dyadic_df['career_win_percentage_diff'] = dyadic_df['player_career_win_percentage'] - dyadic_df['opponent_career_win_percentage_val']
    dyadic_df['career_surface_win_percentage_diff'] = dyadic_df['player_career_surface_win_percentage'] - dyadic_df['opponent_career_surface_win_percentage_val']
    dyadic_df['h2h_win_percentage_diff'] = dyadic_df['h2h_win_percentage']
    dyadic_df['rolling_form_diff'] = dyadic_df['player_rolling_form'] - dyadic_df['opponent_rolling_form_val']
    
    # Create new differential features
    dyadic_df['Momentum_diff'] = dyadic_df['Player_Momentum'] - dyadic_df['Opponent_Momentum']
    dyadic_df['Experience_diff'] = np.log1p(dyadic_df['Player_Experience']) - np.log1p(dyadic_df['Opponent_Experience'])
    dyadic_df['Surface_Experience_diff'] = np.log1p(dyadic_df['Player_Surface_Exp']) - np.log1p(dyadic_df['Opponent_Surface_Exp'])
    
    # Calculate weighted recent performance (more weight to recent matches)
    # Player weighted form
    dyadic_df['Player_Weighted_Form'] = (
        dyadic_df.sort_values('Date')
        .groupby('Player')['Win']
        .transform(lambda x: x.ewm(span=90).mean())
        .fillna(0)
    )
    
    # Opponent weighted form
    dyadic_df['Opponent_Weighted_Form'] = (
        dyadic_df.sort_values('Date')
        .groupby('Opponent')['Win']
        .transform(lambda x: x.ewm(span=90).mean())
        .fillna(0)
    )
    
    dyadic_df['Weighted_Form_diff'] = dyadic_df['Player_Weighted_Form'] - dyadic_df['Opponent_Weighted_Form']
    
    # Calculate tournament level performance
    tournament_levels = {
        'Grand Slam': 1.5,          # Grand Slams
        'Masters 1000': 1.3,        # Masters 1000
        'Tour Finals': 1.4,         # ATP Finals
        'ATP500': 1.1,              # ATP 500
        'ATP250': 1.0,              # ATP 250
        'International': 1.0,        # Regular ATP events
        'Challenger': 0.8,          # Challengers
        'Qualifying': 0.7           # Qualifying rounds
    }
    
    # Map tournament series to levels
    dyadic_df['Tournament_Level'] = dyadic_df['Series'].map(tournament_levels).fillna(1.0)
    
    # Add ranking difference feature
    dyadic_df['Ranking_Diff'] = np.log1p(pd.to_numeric(dyadic_df['LRank'], errors='coerce')) - np.log1p(pd.to_numeric(dyadic_df['WRank'], errors='coerce'))
    dyadic_df['Ranking_Diff'] = np.where(dyadic_df['Player'] == dyadic_df['Winner'], 
                                        dyadic_df['Ranking_Diff'], 
                                        -dyadic_df['Ranking_Diff'])
    
    # Calculate recent surface performance (last 180 matches)
    # Calculate player surface form
    dyadic_df['Player_Surface_Form'] = (
        dyadic_df.sort_values('Date')
        .groupby(['Player', 'Surface'])['Win']
        .transform(lambda x: x.rolling(180, min_periods=1).mean())
        .fillna(0)
    )

    # Calculate opponent surface form
    dyadic_df['Opponent_Surface_Form'] = (
        dyadic_df.sort_values('Date')
        .groupby(['Opponent', 'Surface'])['Win']
        .transform(lambda x: x.rolling(180, min_periods=1).mean())
        .fillna(0)
    )
    
    # Calculate the difference
    dyadic_df['Surface_Form_diff'] = dyadic_df['Player_Surface_Form'] - dyadic_df['Opponent_Surface_Form']
    
    # Calculate Round Progress feature
    round_order = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7
    }
    dyadic_df['Round_Progress'] = dyadic_df['Round'].map(round_order).fillna(0)
    
    # Calculate Match Format Performance
    dyadic_df['Player_Best_of_Win_Rate'] = (
        dyadic_df.sort_values('Date')
        .groupby(['Player', 'Best of'])['Win']
        .transform(lambda x: x.expanding().mean())
        .fillna(0.5)
    )
    dyadic_df['Opponent_Best_of_Win_Rate'] = (
        dyadic_df.sort_values('Date')
        .groupby(['Opponent', 'Best of'])['Win']
        .transform(lambda x: x.expanding().mean())
        .fillna(0.5)
    )
    dyadic_df['Best_of_Performance_diff'] = (
        dyadic_df['Player_Best_of_Win_Rate'] - dyadic_df['Opponent_Best_of_Win_Rate']
    )
    
    # Calculate Indoor/Outdoor Performance
    dyadic_df['Court_Type'] = dyadic_df['Court'].map({'Indoor': 1, 'Outdoor': 0}).fillna(0.5)
    dyadic_df['Player_Court_Win_Rate'] = (
        dyadic_df.sort_values('Date')
        .groupby(['Player', 'Court'])['Win']
        .transform(lambda x: x.expanding().mean())
        .fillna(0.5)
    )
    dyadic_df['Opponent_Court_Win_Rate'] = (
        dyadic_df.sort_values('Date')
        .groupby(['Opponent', 'Court'])['Win']
        .transform(lambda x: x.expanding().mean())
        .fillna(0.5)
    )
    dyadic_df['Court_Performance_diff'] = (
        dyadic_df['Player_Court_Win_Rate'] - dyadic_df['Opponent_Court_Win_Rate']
    )
    
    # Calculate Score Dominance
    set_columns = ['W1', 'W2', 'W3', 'W4', 'W5', 'L1', 'L2', 'L3', 'L4', 'L5']
    for col in set_columns:
        dyadic_df[col] = pd.to_numeric(dyadic_df[col], errors='coerce')
    
    # Calculate average set score difference for completed sets
    dyadic_df['Score_Dominance'] = 0
    for i in range(1, 6):
        set_diff = dyadic_df[f'W{i}'] - dyadic_df[f'L{i}']
        # Only consider completed sets (where either player scored points)
        completed_sets = (dyadic_df[f'W{i}'].notna() & dyadic_df[f'L{i}'].notna())
        dyadic_df.loc[completed_sets, 'Score_Dominance'] += set_diff
    
    # Normalize by number of sets played
    sets_played = dyadic_df[set_columns].notna().sum(axis=1) / 2
    dyadic_df['Score_Dominance'] = dyadic_df['Score_Dominance'] / sets_played.replace(0, 1)
    
    # Points-based features (using ranking points)
    dyadic_df['Points_Ratio'] = np.log1p(dyadic_df['WPts']) - np.log1p(dyadic_df['LPts'])
    
    # Calculate rolling average of Score_Dominance
    dyadic_df['Player_Avg_Dominance'] = (
        dyadic_df.sort_values('Date')
        .groupby('Player')['Score_Dominance']
        .transform(lambda x: x.rolling(10, min_periods=1).mean())
        .fillna(0)
    )
    dyadic_df['Opponent_Avg_Dominance'] = (
        dyadic_df.sort_values('Date')
        .groupby('Opponent')['Score_Dominance']
        .transform(lambda x: x.rolling(10, min_periods=1).mean())
        .fillna(0)
    )
    dyadic_df['Dominance_diff'] = dyadic_df['Player_Avg_Dominance'] - dyadic_df['Opponent_Avg_Dominance']

    final_features = [
        'Date', 'Player', 'Opponent', 'Win', 'Surface',
        'Elo_diff',                           # Base Elo difference
        'career_win_percentage_diff',         # Career win rate difference
        'career_surface_win_percentage_diff', # Surface-specific career win rate diff
        'h2h_win_percentage_diff',           # Head-to-head record
        'rolling_form_diff',                 # Recent form (10 matches)
        'Momentum_diff',                     # 90-day momentum
        'Experience_diff',                   # Total matches played (log)
        'Surface_Experience_diff',           # Surface-specific experience (log)
        'Surface_Form_diff',                 # Recent surface performance
        'Weighted_Form_diff',                # Exponentially weighted form
        'Tournament_Level',                  # Tournament importance
        'Ranking_Diff',                      # Ranking difference (log)
        'Round_Progress',                    # Tournament round progression
        'Best_of_Performance_diff',          # Performance in different match formats
        'Court_Performance_diff',            # Indoor/Outdoor performance difference
        'Location',                          # Tournament location
        'Score_Dominance',                   # Average set score difference
        'Points_Ratio',                      # Log ratio of ranking points
        'Dominance_diff'                     # Difference in average score dominance
    ]
    
    # Keep betting odds if available
    if 'B365W' in dyadic_df.columns and 'B365L' in dyadic_df.columns:
        final_features.extend(['B365W', 'B365L'])
    
    dyadic_df = dyadic_df[final_features].copy()
    for col in final_features:
        if col not in ['Date', 'Player', 'Opponent', 'Surface', 'Win']:
            dyadic_df[col] = pd.to_numeric(dyadic_df[col], errors='coerce')

    dyadic_df = dyadic_df.fillna(0)
    dyadic_df = dyadic_df.replace([np.inf, -np.inf], 0)
    
    return dyadic_df

if __name__ == '__main__':
    raw_data_path = 'C:/Users/Carlos/Documents/ODST/tennis_data/tennis_data.csv'
    features_output_path = 'C:/Users/Carlos/Documents/ODST/tennis_data_features.csv'

    original_df = pd.read_csv(raw_data_path, low_memory=False)
    
    features_df = calculate_features(original_df)
    
    features_df.to_csv(features_output_path, index=False)
    print(f"Feature-rich dataset saved to {features_output_path}")