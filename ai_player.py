import numpy as np
import random
from typing import Dict, List, Tuple, Any
import json

class QLearningPlayer:
    """AI player using Q-learning reinforcement learning."""
    
    def __init__(self, player_id: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.9, epsilon: float = 0.2):
        """
        Initialize Q-learning player.
        
        Args:
            player_id: Unique player identifier
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.player_id = player_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # State-action value table
        self.wins = 0
        self.games_played = 0
        self.total_reward = 0
        
    def get_state_key(self, state: Dict[str, Any]) -> str:
        """Convert game state to hashable key."""
        # Simplified state representation
        player_pos = state.get('player_positions', {}).get(self.player_id, (0, 0))
        
        # Handle both tuple (2D) and int (1D racing) positions
        if isinstance(player_pos, tuple):
            pos_str = f"{player_pos[0]}_{player_pos[1]}"
        else:
            pos_str = str(player_pos)
        
        opponent_count = len([p for p in state.get('player_positions', {}) if p != self.player_id])
        resources = state.get('player_resources', {}).get(self.player_id, 0)
        turn = state.get('turn', 0)
        
        return f"{pos_str}_{opponent_count}_{resources}_{turn % 10}"
    
    def get_q_value(self, state_key: str, action: str) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table.get((state_key, action), 0.0)
    
    def get_valid_actions(self, game_state: Dict[str, Any]) -> List[str]:
        """Get list of valid actions for current state."""
        actions = []
        player_pos = game_state.get('player_positions', {}).get(self.player_id, (0, 0))
        board_size = game_state.get('board_size', 10)
        genre = game_state.get('genre', 'strategy')
        
        # Handle racing games (1D position)
        if genre == 'racing' or isinstance(player_pos, int):
            track_length = game_state.get('track_length', 30)
            if player_pos < track_length - 1:
                actions.append('move_right')
                actions.append('move_down')
            actions.append('stay')
        else:
            # Handle 2D grid games
            x, y = player_pos
            if x > 0:
                actions.append('move_left')
            if x < board_size - 1:
                actions.append('move_right')
            if y > 0:
                actions.append('move_up')
            if y < board_size - 1:
                actions.append('move_down')
            
            # Special actions based on game mechanics
            if game_state.get('can_collect_resource', False):
                actions.append('collect')
            if game_state.get('can_attack', False):
                actions.append('attack')
            
            actions.append('stay')
        
        return actions if actions else ['stay']
    
    def choose_action(self, game_state: Dict[str, Any], training: bool = True) -> str:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            game_state: Current game state
            training: If True, use exploration; if False, use pure exploitation
            
        Returns:
            Selected action
        """
        valid_actions = self.get_valid_actions(game_state)
        
        if not valid_actions:
            return 'stay'
        
        # Exploration vs exploitation
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: choose best action
        state_key = self.get_state_key(game_state)
        q_values = [self.get_q_value(state_key, action) for action in valid_actions]
        max_q = max(q_values)
        
        # Choose randomly among actions with max Q-value
        best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state: Dict[str, Any], action: str, reward: float, 
                      next_state: Dict[str, Any], done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        state_key = self.get_state_key(state)
        current_q = self.get_q_value(state_key, action)
        
        if done:
            # Terminal state
            target_q = reward
        else:
            # Get max Q-value for next state
            next_state_key = self.get_state_key(next_state)
            valid_next_actions = self.get_valid_actions(next_state)
            max_next_q = max([self.get_q_value(next_state_key, a) for a in valid_next_actions], 
                           default=0.0)
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[(state_key, action)] = new_q
        
        self.total_reward += reward
    
    def record_game_result(self, won: bool):
        """Record game outcome."""
        self.games_played += 1
        if won:
            self.wins += 1
    
    def get_win_rate(self) -> float:
        """Calculate win rate."""
        return self.wins / self.games_played if self.games_played > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get player statistics."""
        return {
            'player_id': self.player_id,
            'games_played': self.games_played,
            'wins': self.wins,
            'win_rate': self.get_win_rate(),
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / self.games_played if self.games_played > 0 else 0,
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon
        }
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """Gradually reduce exploration rate."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def save_model(self, filepath: str):
        """Save Q-table to file."""
        model_data = {
            'player_id': self.player_id,
            'q_table': {f"{k[0]}_{k[1]}": v for k, v in self.q_table.items()},
            'stats': self.get_stats()
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load Q-table from file."""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Reconstruct Q-table
            self.q_table = {}
            for key, value in model_data['q_table'].items():
                state, action = key.rsplit('_', 1)
                self.q_table[(state, action)] = value
            
            # Restore stats
            if 'stats' in model_data:
                stats = model_data['stats']
                self.wins = stats.get('wins', 0)
                self.games_played = stats.get('games_played', 0)
                self.total_reward = stats.get('total_reward', 0)
        except Exception as e:
            print(f"Error loading model: {e}")
