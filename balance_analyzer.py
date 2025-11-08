import numpy as np
from typing import Dict, List, Any
from collections import Counter

class GameBalanceAnalyzer:
    """Analyzes game balance and fairness metrics."""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_game_results(self, simulation_results: List[Dict[str, Any]], 
                            num_players: int) -> Dict[str, Any]:
        """
        Analyze multiple game simulations for balance.
        
        Args:
            simulation_results: List of game simulation results
            num_players: Number of players in the game
            
        Returns:
            Dictionary containing balance metrics
        """
        if not simulation_results:
            return self._empty_analysis()
        
        winners = [r['winner'] for r in simulation_results if r['winner'] is not None]
        turns = [r['turns'] for r in simulation_results]
        
        analysis = {
            'total_games': len(simulation_results),
            'win_distribution': self._calculate_win_distribution(winners, num_players),
            'fairness_score': self._calculate_fairness_score(winners, num_players),
            'average_game_length': np.mean(turns) if turns else 0,
            'game_length_std': np.std(turns) if turns else 0,
            'complexity_score': self._calculate_complexity(simulation_results),
            'engagement_score': self._calculate_engagement(simulation_results),
            'balance_grade': 'N/A'
        }
        
        # Overall balance grade
        analysis['balance_grade'] = self._calculate_balance_grade(analysis)
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            'total_games': 0,
            'win_distribution': {},
            'fairness_score': 0.0,
            'average_game_length': 0.0,
            'game_length_std': 0.0,
            'complexity_score': 0.0,
            'engagement_score': 0.0,
            'balance_grade': 'N/A'
        }
    
    def _calculate_win_distribution(self, winners: List[int], 
                                   num_players: int) -> Dict[int, float]:
        """Calculate win percentage for each player."""
        if not winners:
            return {i: 0.0 for i in range(num_players)}
        
        win_counts = Counter(winners)
        total_games = len(winners)
        
        return {
            player_id: (win_counts.get(player_id, 0) / total_games) * 100
            for player_id in range(num_players)
        }
    
    def _calculate_fairness_score(self, winners: List[int], num_players: int) -> float:
        """
        Calculate fairness score (0-100).
        Higher score means more balanced (wins distributed evenly).
        """
        if not winners:
            return 0.0
        
        win_dist = self._calculate_win_distribution(winners, num_players)
        win_percentages = list(win_dist.values())
        
        # Ideal distribution is equal for all players
        ideal_percentage = 100.0 / num_players
        
        # Calculate variance from ideal
        variance = np.var(win_percentages)
        max_variance = ideal_percentage ** 2  # Maximum possible variance
        
        # Convert to 0-100 score (lower variance = higher score)
        fairness = 100 * (1 - min(variance / max_variance, 1.0))
        
        return fairness
    
    def _calculate_complexity(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate complexity score based on game length variation and decisions.
        Returns score 0-100.
        """
        if not results:
            return 0.0
        
        turns = [r['turns'] for r in results]
        
        # Games with moderate length and some variation are more complex
        avg_turns = np.mean(turns)
        std_turns = np.std(turns)
        
        # Normalize to 0-100 scale
        length_score = min(avg_turns / 50 * 50, 50)  # Up to 50 points for length
        variation_score = min(std_turns / 10 * 50, 50)  # Up to 50 points for variation
        
        return length_score + variation_score
    
    def _calculate_engagement(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate engagement score based on game dynamics.
        Returns score 0-100.
        """
        if not results:
            return 0.0
        
        turns = [r['turns'] for r in results]
        
        # Engagement is high when games are not too short or too long
        avg_turns = np.mean(turns)
        
        # Optimal game length around 20-40 turns
        if 20 <= avg_turns <= 40:
            length_engagement = 100
        elif avg_turns < 20:
            length_engagement = (avg_turns / 20) * 100
        else:
            length_engagement = max(0, 100 - (avg_turns - 40) * 2)
        
        # Check score progression
        score_engagement = 50  # Base engagement
        
        return (length_engagement + score_engagement) / 2
    
    def _calculate_balance_grade(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall balance grade A-F."""
        fairness = analysis['fairness_score']
        
        if fairness >= 90:
            return 'A (Excellent)'
        elif fairness >= 75:
            return 'B (Good)'
        elif fairness >= 60:
            return 'C (Fair)'
        elif fairness >= 45:
            return 'D (Poor)'
        else:
            return 'F (Unbalanced)'
    
    def get_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if analysis['fairness_score'] < 60:
            recommendations.append("⚠️ Game appears unbalanced. Consider adjusting starting positions or mechanics.")
        
        if analysis['average_game_length'] < 10:
            recommendations.append("⚡ Games are very short. Consider increasing complexity or board size.")
        elif analysis['average_game_length'] > 60:
            recommendations.append("⏱️ Games are quite long. Consider speeding up win conditions.")
        
        if analysis['complexity_score'] < 30:
            recommendations.append("📊 Low complexity detected. Add more mechanics or strategic depth.")
        elif analysis['complexity_score'] > 80:
            recommendations.append("🧩 High complexity. May be too difficult for casual players.")
        
        if analysis['engagement_score'] < 50:
            recommendations.append("😴 Low engagement score. Improve game dynamics and pacing.")
        
        if not recommendations:
            recommendations.append("✅ Game shows good balance! Continue training AI for further optimization.")
        
        return recommendations
    
    def compare_games(self, game1_results: List[Dict[str, Any]], 
                     game2_results: List[Dict[str, Any]], 
                     num_players: int) -> Dict[str, Any]:
        """Compare two different game designs."""
        analysis1 = self.analyze_game_results(game1_results, num_players)
        analysis2 = self.analyze_game_results(game2_results, num_players)
        
        return {
            'game1': analysis1,
            'game2': analysis2,
            'better_fairness': 1 if analysis1['fairness_score'] > analysis2['fairness_score'] else 2,
            'better_engagement': 1 if analysis1['engagement_score'] > analysis2['engagement_score'] else 2,
            'recommendation': self._get_comparison_recommendation(analysis1, analysis2)
        }
    
    def _get_comparison_recommendation(self, analysis1: Dict[str, Any], 
                                      analysis2: Dict[str, Any]) -> str:
        """Get recommendation from comparison."""
        score1 = (analysis1['fairness_score'] + analysis1['engagement_score']) / 2
        score2 = (analysis2['fairness_score'] + analysis2['engagement_score']) / 2
        
        if abs(score1 - score2) < 5:
            return "Both games are similarly balanced."
        elif score1 > score2:
            return "Game 1 shows better overall balance and engagement."
        else:
            return "Game 2 shows better overall balance and engagement."
