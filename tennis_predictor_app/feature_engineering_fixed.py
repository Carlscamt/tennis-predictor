import pandas as pd
import numpy as np
from tqdm import tqdm

class EloRating:
    def __init__(self, k=30, initial_rating=1500, decay_factor=0.95):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}
        self.last_match_date = {}
        self.decay_factor = decay_factor
    
    def get_rating(self, player, current_date=None):
        """Get a player's current Elo rating with temporal decay"""
        if player not in self.ratings:
            self.ratings[player] = self.initial_rating
            self.last_match_date[player] = current_date
            return self.initial_rating
            
        if current_date and self.last_match_date[player]:
            days_since_last_match = (current_date - self.last_match_date[player]).days
            decay = self.decay_factor ** (days_since_last_match / 365)  # Yearly decay
            return self.ratings[player] * decay
        
        return self.ratings[player]
    
    def update_rating(self, winner, loser, match_date, margin_of_victory=1.0):
        """Update Elo ratings after a match"""
        winner_rating = self.get_rating(winner, match_date)
        loser_rating = self.get_rating(loser, match_date)
        
        # Calculate expected scores
        winner_expected = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        loser_expected = 1 - winner_expected
        
        # Update ratings
        rating_change = self.k * margin_of_victory * (1 - winner_expected)
        self.ratings[winner] = winner_rating + rating_change
        self.ratings[loser] = loser_rating - rating_change
        
        # Update last match dates
        self.last_match_date[winner] = match_date
        self.last_match_date[loser] = match_date


def parse_rank(rank):
    """Convert ranking to float, handling 'NR' (Not Ranked) case"""
    try:
        return float(rank)
    except (ValueError, TypeError):
        return 2000.0  # Assign high rank for unranked players
        rating = self.ratings.get(player, self.initial_rating)
        if current_date and player in self.last_match_date:
            days_inactive = (current_date - self.last_match_date[player]).days
            decay_months = days_inactive / 30.0
            rating *= self.decay_factor ** decay_months
        return rating

    def update_rating(self, winner, loser, date):
        winner_rating = self.get_rating(winner, date)
        loser_rating = self.get_rating(loser, date)
        
        expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        
        # Use fixed K-factor for more stable ratings
        new_winner_rating = winner_rating + self.k * (1 - expected_winner)
        new_loser_rating = loser_rating - self.k * (1 - expected_winner)
        
        self.ratings[winner] = new_winner_rating
        self.ratings[loser] = new_loser_rating
        self.last_match_date[winner] = date
        self.last_match_date[loser] = date

def calculate_features(df):
    print("Calculating features...")
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)

    # Initialize player stats dictionaries
    player_stats = {}
    surface_stats = {}
    h2h_stats = {}
    recent_matches = {}
    
    features = []
    
    # Process matches chronologically
    for idx, match in tqdm(df.iterrows(), total=len(df)):
        date = match['Date']
        winner = match['Winner']
        loser = match['Loser']
        surface = match['Surface']
        
        # Initialize stats for new players
        for player in [winner, loser]:
            if player not in player_stats:
                player_stats[player] = {'matches': 0, 'wins': 0}
            if player not in surface_stats:
                surface_stats[player] = {surf: {'matches': 0, 'wins': 0} for surf in df['Surface'].unique()}
            if player not in recent_matches:
                recent_matches[player] = []
                
        # Calculate pre-match features
        winner_features = {}
        loser_features = {}
        
        # Career stats
        for player, features_dict in [(winner, winner_features), (loser, loser_features)]:
            stats = player_stats[player]
            features_dict['career_win_pct'] = stats['wins'] / max(1, stats['matches'])
            
            # Surface stats
            surface_stat = surface_stats[player][surface]
            features_dict['surface_win_pct'] = surface_stat['wins'] / max(1, surface_stat['matches'])
            
            # Recent form (last 10 matches)
            recent = recent_matches[player][-10:] if recent_matches[player] else []
            features_dict['recent_form'] = sum(recent) / len(recent) if recent else 0
            
        # H2H stats
        h2h_key = tuple(sorted([winner, loser]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {'matches': 0, 'wins': {winner: 0, loser: 0}}
        h2h = h2h_stats[h2h_key]
        
        winner_features['h2h_win_pct'] = h2h['wins'][winner] / max(1, h2h['matches'])
        loser_features['h2h_win_pct'] = h2h['wins'][loser] / max(1, h2h['matches'])
        
        # Calculate differentials
        feature_row = {
            'Date': date,
            'Winner': winner,
            'Loser': loser,
            'Surface': surface,
            'career_win_pct_diff': winner_features['career_win_pct'] - loser_features['career_win_pct'],
            'surface_win_pct_diff': winner_features['surface_win_pct'] - loser_features['surface_win_pct'],
            'recent_form_diff': winner_features['recent_form'] - loser_features['recent_form'],
            'h2h_win_pct_diff': winner_features['h2h_win_pct'] - loser_features['h2h_win_pct'],
            'ranking_diff': np.log1p(parse_rank(match['LRank'])) - np.log1p(parse_rank(match['WRank']))
        }
        
        features.append(feature_row)
        
        # Update stats after recording features
        # Career stats
        player_stats[winner]['matches'] += 1
        player_stats[winner]['wins'] += 1
        player_stats[loser]['matches'] += 1
        
        # Surface stats
        surface_stats[winner][surface]['matches'] += 1
        surface_stats[winner][surface]['wins'] += 1
        surface_stats[loser][surface]['matches'] += 1
        
        # H2H stats
        h2h_stats[h2h_key]['matches'] += 1
        h2h_stats[h2h_key]['wins'][winner] += 1
        
        # Recent form
        recent_matches[winner].append(1)
        recent_matches[loser].append(0)
    
    features_df = pd.DataFrame(features)
    
    # Create dyadic version of the dataset
    p1_data = features_df.copy()
    p1_data['Player'] = p1_data['Winner']
    p1_data['Opponent'] = p1_data['Loser']
    p1_data['Win'] = 1

    p2_data = features_df.copy()
    p2_data['Player'] = p2_data['Loser']
    p2_data['Opponent'] = p2_data['Winner']
    p2_data['Win'] = 0
    # Reverse the differentials for player 2
    diff_columns = [col for col in features_df.columns if col.endswith('_diff')]
    for col in diff_columns:
        p2_data[col] = -p2_data[col]

    dyadic_df = pd.concat([p1_data, p2_data], ignore_index=True)
    dyadic_df['match_id'] = dyadic_df.groupby(['Date', 'Winner', 'Loser']).ngroup()
    dyadic_df.sort_values(by=['Date', 'match_id', 'Win'], inplace=True)
    
    return dyadic_df

if __name__ == '__main__':
    raw_data_path = 'tennis_data/tennis_data.csv'
    features_output_path = 'tennis_data_features_fixed.csv'

    original_df = pd.read_csv(raw_data_path, low_memory=False)
    features_df = calculate_features(original_df)
    features_df.to_csv(features_output_path, index=False)
    print(f"Feature-rich dataset saved to {features_output_path}")
