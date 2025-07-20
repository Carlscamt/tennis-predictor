import numpy as np

class UncertaintyShrinkageBetting:
    def __init__(self, initial_bankroll=1000, min_prob=0.55, shrinkage_factor=0.5, uncertainty_threshold=0.3):
        self.bankroll = initial_bankroll
        self.min_prob = min_prob
        self.shrinkage_factor = shrinkage_factor
        self.uncertainty_threshold = uncertainty_threshold
        self.max_bet_fraction = 0.10

    def calculate_uncertainty(self, prob):
        return 1 - 2 * abs(prob - 0.5)

    def calculate_kelly_fraction(self, prob, market_odds):
        # Standard Kelly Criterion calculation
        b = market_odds - 1
        q = 1 - prob
        f = (b * prob - q) / b if b > 0 else 0

        # Apply uncertainty shrinkage
        uncertainty = self.calculate_uncertainty(prob)
        if uncertainty > self.uncertainty_threshold:
            f *= (1 - self.shrinkage_factor)

        # Ensure bet size is within limits
        return min(max(f, 0), self.max_bet_fraction), market_odds
        
    def get_fair_odds(self, prob):
        """
        Calculate fair odds based on probability
        """
        return 1 / prob if prob > 0 else float('inf')
