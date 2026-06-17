import pandas as pd
from datetime import datetime
import numpy.typing as npt
import numpy as np
import math

def get_players_statistics(
    players: pd.DataFrame, 
    statistics: pd.DataFrame,
    player_id: str, 
    match_date: str
):
    """ 
    Return statistics for a player:
        1. age
        2. forward
        3. middlefielder
        4. defense
        5. goalkeeper
        6. num_tournaments
        7. appearances
        8. goals
        9. average time per team
    
    Args:
        players (DataFrame): Dataframe with player information
        statistics (DataFrame): Dataframe with statistics of each player
        player_id (str): id of player
        match_date (str): day match was played
        
    Returns:
        x_i (ndarray): (1,9) array with statistics of each player
    """    
    player = players[players['ID'] == player_id]
    if len(player) < 1:
        return np.array([
            [0,0,0,0,0,0,0,0,0]
        ])
    
    # Age of player
    # raw_birthdate = player['birth_date'].iloc[0]
    birthdate = player['birth_date'].iloc[0]
    if (type(birthdate) == str):
        birthdate = datetime.strptime(birthdate.strip(), "%Y-%m-%d")
    tournament_date = datetime.strptime(match_date, "%Y-%m-%d")
    match_year = tournament_date.year
    difference_in_days = (tournament_date - birthdate).days
    age = int(difference_in_days / 365.25)
    
    forward = player['forward'].iloc[0]
    middlefielder = player['midfielder'].iloc[0]
    defense = player['defender'].iloc[0]
    goalkeeper = player['goalkeeper'].iloc[0]
    
    raw_tournaments = player['list_tournaments'].iloc[0].split(",")
    tournaments = [int(t) < match_year for t in raw_tournaments]
    num_tournaments = sum(tournaments)
    
    appearances = 0
    goals = 0
    average_time_per_team = 0
    num_teams = 0
    player_statistics = statistics[statistics['id'] == player_id]
    
    if len(player_statistics) < 1:
        return np.array(
            [[age, forward, middlefielder, defense, goalkeeper, num_tournaments, appearances, goals, average_time_per_team]]
        )
    
    # Get stats as per tournament year
    for stat in player_statistics.iterrows():
        s_from = stat[1]["from"]
        s_to = match_year if np.isnan(stat[1]["to"]) else stat[1]["to"]
        s_appearance = 0 if np.isnan(stat[1]['appearances']) else stat[1]['appearances']
        s_goals = 0 if np.isnan(stat[1]['goals']) else stat[1]['goals']
        
        
        if s_from < match_year and s_to < match_year:
            appearances += s_appearance
            goals += s_goals
            num_teams += 1
            average_time_per_team += s_to - s_from
        elif s_from < match_year:
            total = s_to - s_from
            num_years = match_year - s_from
            appearances += int(s_appearance * (num_years / total))
            goals += int(s_goals * (num_years / total))
            num_teams += 1
            average_time_per_team += num_years
            
            
    if num_teams == 0:
        average_time_per_team = 0
    else:
        average_time_per_team /= num_teams
        
    return np.array([
            [age, forward, middlefielder, defense, goalkeeper, num_tournaments, appearances, goals, average_time_per_team]
    ]
        )
    
def get_team_statistics_from_squad(
    statistics: pd.DataFrame,
    players: pd.DataFrame,
    squad_members: pd.DataFrame, 
    squad_id: int,
    match_date: str
):
    """ 
    This function should return a (11,9) vector with the statistics of the players in squad provided
    
    Args:
        statistics: Dataframe with statistics of each player
        players: Dataframe with details of players
        squad_members: Dataframe with ids of players in squad
        squad_id (scalar): identifier of squad playing
        match_date (str): Date of the match in YYYY-MM-dd
        
    Returns:
        team_statistics (ndarray): (11,9) vector with statistics of the players from the team
    """
    team_members = squad_members[squad_members["squad_id"] == squad_id].head(n=11)
    player_ids = team_members['player_id']
    if len(player_ids) != 11:
        raise Exception("Could not get players for team")
    
    player_ids = player_ids.tolist()
    team = np.empty((0,9))
    for p_id in player_ids:
        stats = get_players_statistics(players=players, statistics=statistics, player_id=p_id, match_date=match_date)
        team = np.append(team, stats, axis=0)
        
    return team
    
def f_x(
    X: npt.NDArray,
    W1: npt.NDArray,
    b1: npt.NDArray,
    W2: npt.NDArray,
    b2: npt.NDArray,
    W3: npt.NDArray,
    b3: npt.NDArray
):
    """ 
    Function to get m predictions
    
    Args:
        X (ndarray): array with m feature vectors
        W1 (ndarray): array with weights of first layer
        b1 (ndarray): array with bias of first layer
        W2 (ndarray): array with weights of second layer
        b2 (ndarray): array with bias of second layer
        W3 (ndarray): array with weights of third layer
        b3 (ndarray): array with bias of third layer
        
    Returns:
        Y_pred (ndarray): array with m predictions
        A1 (ndarray): cached A1 value
        Z1 (ndarray): cached Z1 value
        A2 (ndarray): cached A2 value
        Z2 (ndarray): cached Z2 value
    """
    Z1 = np.matmul(W1, X) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = np.maximum(0, Z2)
    Z3 = np.matmul(W3, A2) + b3
    
    shifted_logits = Z3 - np.max(Z3, axis=0, keepdims=True)
    e_z = np.exp(shifted_logits)
    Y_pred = e_z / np.sum(e_z, axis=0, keepdims=True)
    
    return (Y_pred, A1, Z1, A2, Z2)

def get_feature_to_predict(
    mean_std: npt.NDArray,
    statistics: pd.DataFrame,
    players: pd.DataFrame,
    squad_members: pd.DataFrame,
    home_squad_id: int,
    away_squad_id: int,
    match_date: str
):
    """ 
    Get feature that I can use to make a prediction
    
    Args:
        mean_std: array with mean and std from training set
        statistics: dataframe with statistics of players
        players: dataframe with information of players
        squad_members: dataframe with players in a squad
        home_squad_id: Identifer of the squad for the home team
        away_squad_id: Identifer of the squad for the away team
        match_date: Date match is played in YYYY-MM-dd
        
    Returns:
        x (ndarray): a (18,1) array that has also been normalized
    """
    home_team = get_team_statistics_from_squad(
        statistics=statistics,
        players=players,
        squad_members=squad_members,
        squad_id=home_squad_id, 
        match_date=match_date
    )
    away_team = get_team_statistics_from_squad(
        statistics=statistics,
        players=players,
        squad_members=squad_members,
        squad_id=away_squad_id, 
        match_date=match_date
    )
    
    # Get sum of all members in each team
    sum_home_team = np.sum(home_team, axis=0)
    sum_away_team = np.sum(away_team, axis=0)
    
    # Concatenate
    sum_concated = (np.concatenate((sum_home_team, sum_away_team), axis=-1)).reshape((18, 1))
    
    # Normalize
    mean = mean_std[0]
    std = mean_std[0]
    
    sum_normalized = (sum_concated - mean) / std
    
    return sum_normalized

W1_end_index = 27 * 18
b1_end_index = W1_end_index + 27
W2_end_index = b1_end_index + (54 * 27)
b2_end_index = W2_end_index + 54
W3_end_index = b2_end_index + (121 * 54)
b3_end_index = W3_end_index + 121

def uncondense_parameters(
    theta: npt.NDArray
):
    """  
    Uncondense parameters
    
    Args:
        theta (ndarray): condensed parameters
        
    Returns:
        W1 (ndarray): uncondensed weights of first layer
        b1 (ndarray): uncondensed bias of first layer
        W2 (ndarray): uncondensed weights of second layer
        b2 (ndarray): uncondensed bias of second layer
        W3 (ndarray): uncondensed weights of third layer
        b3 (ndarray): uncondensed bias of third layer
    """
    W1 = np.array(theta[0:W1_end_index]).reshape((27, 18))
    b1 = np.array(theta[W1_end_index:b1_end_index]).reshape((27,1))
    W2 = np.array(theta[b1_end_index:W2_end_index]).reshape((54, 27))
    b2 = np.array(theta[W2_end_index:b2_end_index]).reshape((54, 1))
    W3 = np.array(theta[b2_end_index:W3_end_index]).reshape((121, 54))
    b3 = np.array(theta[W3_end_index:]).reshape((121, 1))
    
    return (W1, b1, W2, b2, W3, b3)

def interpret_probabilities(prob):
    """ 
    Function to interpret the returned probabilities by the model
    
    Args:
        prob (ndarray) - array with probabilities for different match results
        
    Returns:
        scores (str) - human readable version of prediction
    """
    predicted = np.argmax(prob)
    home_score = math.floor(predicted / 11)
    away_score = predicted % 11
    return f"{home_score}-{away_score}"