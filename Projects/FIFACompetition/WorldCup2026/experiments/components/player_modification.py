import numpy.typing as npt
import numpy as np
from .data import extract_features_from_old_player_vector
from .constants import FEW_APPEARANCES_THRESHOLD, AVG_APPEARANCES_THRESHOLD

def arrange_player_details_to_offense_defense(
    raw_player: npt.NDArray
):
    """ 
    Takes in a raw player with old details and returns player with details for offense defense test i.e.
        1. age
        2. offense
        3. defense
        4. few appearances
        5. avg appearances
        6. many appearances
        7. num_tournaments
        8. average time per team
        
    With shape (1,8)
    """
    age, forward, midfielder, defender, goalkeeper, num_tournamets, apperances, goals, average_time = extract_features_from_old_player_vector(player=raw_player)
    
    # Get offense and defense
    offense = 0
    defense = 0
    if (forward == 1 or midfielder == 1):
        offense = goals / apperances if apperances > 0 else 0
    elif (defender == 1 or goalkeeper ==1):
        defense = apperances
        
    # Categorizing appearances
    few_appearances = 0
    avg_appearances = 0
    many_appearances = 0
    if (apperances < FEW_APPEARANCES_THRESHOLD):
        few_appearances = 1
    elif (FEW_APPEARANCES_THRESHOLD <= apperances < AVG_APPEARANCES_THRESHOLD):
        avg_appearances = 1
    elif (apperances >= AVG_APPEARANCES_THRESHOLD):
        many_appearances = 1
    
    return np.array([
        [age, offense, defense, few_appearances, avg_appearances, many_appearances, num_tournamets, average_time]
    ])