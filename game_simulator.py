import random
import numpy as np
from typing import Dict, List, Any, Tuple
from ai_player import QLearningPlayer

class GameSimulator:
    """Simulates board game gameplay with AI players."""
    
    def __init__(self, game_config: Dict[str, Any]):
        """
        Initialize game simulator.
        
        Args:
            game_config: Game configuration from generator
        """
        self.game_config = game_config
        self.genre = game_config.get('genre', 'strategy')
        self.board_size = game_config.get('board', {}).get('size', 8)
        self.num_players = game_config.get('num_players', 2)
        self.max_turns = 100
        self.simulation_history = []
        
    def initialize_game_state(self) -> Dict[str, Any]:
        """Initialize game state for new game."""
        state = {
            'turn': 0,
            'board_size': self.board_size,
            'genre': self.genre,
            'track_length': self.game_config.get('board', {}).get('length', 30),
            'player_positions': {},
            'player_resources': {},
            'player_scores': {},
            'eliminated_players': set(),
            'board': self._initialize_board(),
            'game_over': False,
            'winner': None
        }
        
        # Initialize player positions
        pieces_per_player = self.game_config.get('pieces_per_player', 1)
        for player_id in range(self.num_players):
            if pieces_per_player == 1:
                # Single piece (racing games)
                state['player_positions'][player_id] = (0, 0) if self.genre != 'racing' else 0
            else:
                # Multiple pieces (strategy/territory games)
                state['player_positions'][player_id] = self._get_starting_position(player_id)
            
            state['player_resources'][player_id] = 0
            state['player_scores'][player_id] = 0
        
        return state
    
    def _initialize_board(self) -> np.ndarray:
        """Initialize game board."""
        if self.genre == 'racing':
            track_length = self.game_config.get('board', {}).get('length', 30)
            return np.zeros(track_length)
        else:
            return np.zeros((self.board_size, self.board_size))
    
    def _get_starting_position(self, player_id: int) -> Tuple[int, int]:
        """Get starting position for player."""
        # Place players in corners
        corners = [(0, 0), (0, self.board_size - 1), 
                  (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        return corners[player_id % len(corners)]
    
    def simulate_game(self, ai_players: List[QLearningPlayer], 
                     training: bool = True, render: bool = False) -> Dict[str, Any]:
        """
        Simulate complete game with AI players.
        
        Args:
            ai_players: List of AI players
            training: If True, players learn from game
            render: If True, store visualization data
            
        Returns:
            Game results including winner, turns, and stats
        """
        state = self.initialize_game_state()
        game_history = [] if render else None
        
        for turn in range(self.max_turns):
            state['turn'] = turn
            
            if render:
                game_history.append(self._capture_state_snapshot(state))
            
            # Each player takes turn
            for player_id in range(self.num_players):
                if player_id in state['eliminated_players']:
                    continue
                
                # Get AI player
                ai_player = ai_players[player_id]
                
                # Choose action
                action = ai_player.choose_action(state, training=training)
                
                # Execute action and get reward
                next_state, reward, done = self._execute_action(state, player_id, action)
                
                # Check win condition
                if self._check_win_condition(next_state, player_id):
                    next_state['game_over'] = True
                    next_state['winner'] = player_id
                    done = True
                    reward += 100
                
                # Update Q-values if training
                if training:
                    ai_player.update_q_value(state, action, reward, next_state, done)
                
                state = next_state
                
                if state['game_over']:
                    break
            
            if state['game_over']:
                break
        
        # Record results
        for player_id, ai_player in enumerate(ai_players):
            won = state['winner'] == player_id
            ai_player.record_game_result(won)
            if training:
                ai_player.decay_epsilon()
        
        results = {
            'winner': state['winner'],
            'turns': state['turn'],
            'final_scores': state['player_scores'],
            'final_resources': state['player_resources'],
            'game_history': game_history
        }
        
        self.simulation_history.append(results)
        return results
    
    def _execute_action(self, state: Dict[str, Any], player_id: int, 
                       action: str) -> Tuple[Dict[str, Any], float, bool]:
        """
        Execute player action and return new state, reward, and done flag.
        
        Returns:
            (next_state, reward, done)
        """
        next_state = state.copy()
        reward = 0.0
        done = False
        
        # Movement actions
        if self.genre == 'racing':
            reward = self._execute_racing_action(next_state, player_id, action)
        elif self.genre == 'resource_management':
            reward = self._execute_resource_action(next_state, player_id, action)
        elif self.genre == 'territory_control':
            reward = self._execute_territory_action(next_state, player_id, action)
        else:  # strategy
            reward = self._execute_strategy_action(next_state, player_id, action)
        
        # Small negative reward for each turn (encourages efficiency)
        reward -= 0.01
        
        return next_state, reward, done
    
    def _execute_racing_action(self, state: Dict[str, Any], player_id: int, 
                               action: str) -> float:
        """Execute racing game action."""
        current_pos = state['player_positions'][player_id]
        track_length = self.game_config.get('board', {}).get('length', 30)
        
        movement = 0
        if action in ['move_right', 'move_down']:
            movement = self.game_config.get('rules', {}).get('base_movement', 2)
            if self.game_config.get('rules', {}).get('dice_modifier', False):
                movement += random.randint(0, 2)
        
        new_pos = min(current_pos + movement, track_length - 1)
        state['player_positions'][player_id] = new_pos
        state['player_scores'][player_id] = new_pos
        
        # Reward for forward progress
        reward = movement * 0.5
        
        # Big reward for finishing
        if new_pos >= track_length - 1:
            reward += 100
        
        return reward
    
    def _execute_resource_action(self, state: Dict[str, Any], player_id: int, 
                                 action: str) -> float:
        """Execute resource management action."""
        reward = 0.0
        
        if action == 'collect':
            collection_rate = self.game_config.get('rules', {}).get('collection_rate', 2)
            state['player_resources'][player_id] += collection_rate
            reward = collection_rate
        elif action.startswith('move_'):
            # Update position
            pos = state['player_positions'][player_id]
            new_pos = self._move_position(pos, action)
            state['player_positions'][player_id] = new_pos
            
            # Small reward for exploring
            reward = 0.1
        
        # Update score based on resources
        state['player_scores'][player_id] = state['player_resources'][player_id]
        
        return reward
    
    def _execute_territory_action(self, state: Dict[str, Any], player_id: int, 
                                  action: str) -> float:
        """Execute territory control action."""
        reward = 0.0
        
        if action.startswith('move_'):
            pos = state['player_positions'][player_id]
            new_pos = self._move_position(pos, action)
            
            # Check if claiming new territory
            if self._is_unclaimed_territory(state, new_pos):
                reward = 5.0
                state['player_scores'][player_id] += 3
            
            state['player_positions'][player_id] = new_pos
        
        return reward
    
    def _execute_strategy_action(self, state: Dict[str, Any], player_id: int, 
                                 action: str) -> float:
        """Execute strategy game action."""
        reward = 0.0
        win_condition = self.game_config.get('win_condition', {})
        goal_pos = win_condition.get('goal_position', (self.board_size - 1, self.board_size - 1))
        
        if action.startswith('move_'):
            pos = state['player_positions'][player_id]
            new_pos = self._move_position(pos, action)
            
            # Reward for getting closer to goal
            old_dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
            new_dist = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])
            
            if new_dist < old_dist:
                reward = 1.0
            elif new_dist > old_dist:
                reward = -0.5
            
            state['player_positions'][player_id] = new_pos
            state['player_scores'][player_id] += 1
        
        return reward
    
    def _move_position(self, current_pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Calculate new position based on movement action."""
        x, y = current_pos
        
        if action == 'move_left':
            x = max(0, x - 1)
        elif action == 'move_right':
            x = min(self.board_size - 1, x + 1)
        elif action == 'move_up':
            y = max(0, y - 1)
        elif action == 'move_down':
            y = min(self.board_size - 1, y + 1)
        
        return (x, y)
    
    def _is_unclaimed_territory(self, state: Dict[str, Any], position: Tuple[int, int]) -> bool:
        """Check if territory is unclaimed."""
        for player_id, pos in state['player_positions'].items():
            if pos == position:
                return False
        return random.random() > 0.7
    
    def _check_win_condition(self, state: Dict[str, Any], player_id: int) -> bool:
        """Check if player has won."""
        win_condition = self.game_config.get('win_condition', {})
        win_type = win_condition.get('type', 'position_or_elimination')
        
        if win_type == 'race_finish':
            track_length = self.game_config.get('board', {}).get('length', 30)
            return state['player_positions'][player_id] >= track_length - 1
        
        elif win_type == 'resource_collection':
            target = win_condition.get('target_resources', 50)
            return state['player_resources'][player_id] >= target
        
        elif win_type == 'territory_control':
            required_score = win_condition.get('control_percentage', 0.6) * 100
            return state['player_scores'][player_id] >= required_score
        
        else:  # position_or_elimination
            goal_pos = win_condition.get('goal_position', (self.board_size - 1, self.board_size - 1))
            return state['player_positions'][player_id] == goal_pos
        
        return False
    
    def _capture_state_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Capture state for visualization."""
        return {
            'turn': state['turn'],
            'positions': dict(state['player_positions']),
            'scores': dict(state['player_scores']),
            'resources': dict(state['player_resources'])
        }
