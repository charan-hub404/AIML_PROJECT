import random
import json
from typing import Dict, List, Any, Tuple

class BoardGameGenerator:
    """Generates board game concepts with rules, objectives, and mechanics."""
    
    def __init__(self):
        self.genres = ['strategy', 'resource_management', 'racing', 'territory_control']
        self.mechanics = ['dice_rolling', 'card_drawing', 'tile_placement', 'worker_placement', 
                         'area_control', 'set_collection', 'movement']
        self.objectives = ['reach_goal', 'collect_resources', 'eliminate_opponents', 
                          'control_territory', 'score_points']
        
    def generate_game(self, genre: str = None, complexity: float = 0.5) -> Dict[str, Any]:
        """
        Generate a complete board game concept.
        
        Args:
            genre: Specific genre or None for random
            complexity: 0.0 to 1.0, affects number of mechanics and rules
            
        Returns:
            Dictionary containing complete game specification
        """
        if genre is None or genre not in self.genres:
            genre = random.choice(self.genres)
        
        # Determine number of mechanics based on complexity
        num_mechanics = max(2, int(complexity * 5) + 2)
        selected_mechanics = random.sample(self.mechanics, min(num_mechanics, len(self.mechanics)))
        
        # Generate game based on genre
        if genre == 'strategy':
            game = self._generate_strategy_game(selected_mechanics, complexity)
        elif genre == 'resource_management':
            game = self._generate_resource_game(selected_mechanics, complexity)
        elif genre == 'racing':
            game = self._generate_racing_game(selected_mechanics, complexity)
        else:  # territory_control
            game = self._generate_territory_game(selected_mechanics, complexity)
        
        game['genre'] = genre
        game['mechanics'] = selected_mechanics
        game['complexity'] = complexity
        game['id'] = self._generate_game_id()
        
        return game
    
    def _generate_game_id(self) -> str:
        """Generate unique game identifier."""
        return f"game_{random.randint(1000, 9999)}"
    
    def _generate_strategy_game(self, mechanics: List[str], complexity: float) -> Dict[str, Any]:
        """Generate strategy board game."""
        board_size = random.randint(5, 10)
        num_players = random.randint(2, 4)
        
        return {
            'name': f"Strategic Conquest {random.randint(100, 999)}",
            'num_players': num_players,
            'board': {
                'type': 'grid',
                'size': board_size,
                'positions': board_size * board_size
            },
            'pieces_per_player': random.randint(3, 8),
            'objective': 'Reach the goal position or eliminate all opponents',
            'win_condition': {
                'type': 'position_or_elimination',
                'goal_position': (board_size - 1, board_size - 1),
                'points_to_win': 100
            },
            'rules': {
                'movement_range': random.randint(1, 3),
                'can_capture': 'dice_rolling' in mechanics or 'area_control' in mechanics,
                'resource_generation': 'resource_management' in mechanics or 'set_collection' in mechanics,
                'special_abilities': complexity > 0.6
            },
            'turn_structure': self._generate_turn_structure(mechanics),
            'scoring': {
                'capture_piece': 10,
                'reach_checkpoint': 5,
                'position_bonus': 2
            }
        }
    
    def _generate_resource_game(self, mechanics: List[str], complexity: float) -> Dict[str, Any]:
        """Generate resource management game."""
        num_players = random.randint(2, 4)
        num_resources = random.randint(2, 4)
        
        return {
            'name': f"Resource Empire {random.randint(100, 999)}",
            'num_players': num_players,
            'board': {
                'type': 'resource_map',
                'size': random.randint(6, 10),
                'resource_nodes': random.randint(8, 15)
            },
            'pieces_per_player': random.randint(2, 5),
            'objective': 'Collect target amount of resources first',
            'win_condition': {
                'type': 'resource_collection',
                'target_resources': random.randint(30, 60)
            },
            'rules': {
                'resource_types': num_resources,
                'collection_rate': random.randint(1, 3),
                'trading_allowed': complexity > 0.5,
                'resource_conversion': complexity > 0.7
            },
            'turn_structure': self._generate_turn_structure(mechanics),
            'scoring': {
                'resource_collected': 1,
                'set_bonus': 5 if 'set_collection' in mechanics else 0
            }
        }
    
    def _generate_racing_game(self, mechanics: List[str], complexity: float) -> Dict[str, Any]:
        """Generate racing board game."""
        track_length = random.randint(20, 40)
        num_players = random.randint(2, 6)
        
        return {
            'name': f"Speed Chase {random.randint(100, 999)}",
            'num_players': num_players,
            'board': {
                'type': 'track',
                'length': track_length,
                'hazards': random.randint(3, 8) if complexity > 0.4 else 0
            },
            'pieces_per_player': 1,
            'objective': 'Reach the finish line first',
            'win_condition': {
                'type': 'race_finish',
                'finish_position': track_length
            },
            'rules': {
                'base_movement': random.randint(1, 3),
                'dice_modifier': 'dice_rolling' in mechanics,
                'card_boosts': 'card_drawing' in mechanics,
                'obstacle_penalty': -1 if complexity > 0.4 else 0
            },
            'turn_structure': self._generate_turn_structure(mechanics),
            'scoring': {
                'finish_first': 100,
                'finish_second': 50,
                'finish_third': 25
            }
        }
    
    def _generate_territory_game(self, mechanics: List[str], complexity: float) -> Dict[str, Any]:
        """Generate territory control game."""
        board_size = random.randint(6, 10)
        num_players = random.randint(2, 4)
        
        return {
            'name': f"Territory Wars {random.randint(100, 999)}",
            'num_players': num_players,
            'board': {
                'type': 'grid',
                'size': board_size,
                'territories': random.randint(8, 16)
            },
            'pieces_per_player': random.randint(5, 12),
            'objective': 'Control majority of territories',
            'win_condition': {
                'type': 'territory_control',
                'control_percentage': random.uniform(0.5, 0.7)
            },
            'rules': {
                'placement_phase': True,
                'combat_system': 'dice_rolling' in mechanics or 'card_drawing' in mechanics,
                'reinforcements': complexity > 0.5,
                'alliances': complexity > 0.7
            },
            'turn_structure': self._generate_turn_structure(mechanics),
            'scoring': {
                'territory_controlled': 3,
                'continent_bonus': 10 if complexity > 0.6 else 0
            }
        }
    
    def _generate_turn_structure(self, mechanics: List[str]) -> List[str]:
        """Generate turn structure based on mechanics."""
        base_actions = ['move']
        
        if 'dice_rolling' in mechanics:
            base_actions.insert(0, 'roll_dice')
        if 'card_drawing' in mechanics:
            base_actions.insert(0, 'draw_card')
        if 'worker_placement' in mechanics:
            base_actions.append('place_worker')
        if 'tile_placement' in mechanics:
            base_actions.append('place_tile')
        
        base_actions.append('end_turn')
        return base_actions
    
    def get_game_description(self, game: Dict[str, Any]) -> str:
        """Generate human-readable game description."""
        desc = f"**{game['name']}** ({game['genre'].replace('_', ' ').title()})\n\n"
        desc += f"Players: {game['num_players']}\n"
        desc += f"Complexity: {game['complexity']:.1%}\n\n"
        desc += f"**Objective:** {game['objective']}\n\n"
        desc += f"**Mechanics:** {', '.join([m.replace('_', ' ').title() for m in game['mechanics']])}\n\n"
        desc += f"**Board:** {game['board']['type'].title()}"
        
        if 'size' in game['board']:
            desc += f" ({game['board']['size']}x{game['board']['size']})"
        
        return desc
