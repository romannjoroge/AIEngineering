import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

def load_original_data_with_trimmed_y():
    """  
    Import the original dataset with (m, 22, 9) array for training features and (m, 121) array with target labels
    """
    # Combine players in training set
    X = np.load("../../../data/duplicate_players_batch_6_2026-06-01 18:30:48.366255.npy")
    Y = np.load("../../../data/duplicate_results_batch_6_2026-06-01 18:30:48.366274.npy")

    # Trim target variable
    Y_trimmed = np.squeeze(Y, axis=-1)
    
    return (X, Y_trimmed)

def extract_features_from_old_player_vector(player: npt.NDArray):
    """  
    Many experiments require creating new player data based on player data from first version of extraction. The data contained in this version is:
        1. age
        2. forward
        3. middlefielder
        4. defense
        5. goalkeeper
        6. num_tournaments
        7. appearances
        8. goals
        9. average time per team
        
    This function returns each of these values from the given array using indexes above
    """
    age = player[0]
    forward = player[1]
    midfielder = player[2]
    defender = player[3]
    goalkeeper = player[4]
    num_tournamets = player[5]
    apperances = player[6]
    goals = player[7]
    average_time = player[8]
    
    return (age, forward, midfielder, defender, goalkeeper, num_tournamets, apperances, goals, average_time)

def collect_player_data_into_sum_concat_teams_form_from_old(
    Input: npt.NDArray,
    num_player_features: int,
    extract_player_vector
):
    """  
    Alot of experiments require transforming player data, arranging them into teams then combining team information using sum concat method
    that I previously saw worked well in training
    
    This functions encapsulates logic for combining players into teams, summing team then concating
    
    Args:
        Input (ndarray): a (m, 22, 9) vector with old details of each player in both teams
        num_player_features (scalar): number of features in each transformed player. This is needed when combining players to teams
        extract_player_vector (function): function that takes in old player details and returns vector with transformed player details
        
    Returns:
        Output (ndarray): array with outputs that have been sum concatenated
    """
    num_matchups = Input.shape[0] 
    Output = np.zeros((0, num_player_features * 2))
    
    # For each matchup
    for matchup in range(num_matchups):
        # Get players for each team in matchup
        home_team_players = np.zeros((0,num_player_features))
        away_team_players = np.zeros((0,num_player_features))
        
        # Combine offense defense for each team
        for i in range(22):
            player = extract_player_vector(raw_player=Input[matchup,i,:])
            if (i < 11):
                home_team_players = np.append(home_team_players, player, axis=0)
            else:
                away_team_players = np.append(away_team_players, player, axis=0)

        # Sum of home team and away team
        sum_home_team = np.sum(home_team_players, axis=0)
        sum_away_team = np.sum(away_team_players, axis=0)

        # Concat. I reshape so that output can be split with test train split in later steps
        sum_concated = (np.concatenate((sum_home_team, sum_away_team), axis=-1)).reshape((1, 16))

        # Add to Output data. 
        Output = np.append(Output, sum_concated, axis=0)
        
    return Output

def split_and_normalize_dataset(
    X: npt.NDArray,
    Y: npt.NDArray,
    training_set_ratio: float,
    cv_test_set_ratio: float,
    random_state: int
):
    """ 
    I often need to get a train, cv, test split of some data. This function does this when given a (m, n) dataset with training features
    and (m,n) target labels
    
    Args:
        X (ndarray): Training features
        Y (ndarray): Target output
        training_set_ratio (scalar): ratio of training set to entire dataset
        cv_test_set_ration (scalar): ration of cv_set to dataset - training set
        random_state (scalar): random value that will be used when splitting
        
    Returns:
        X_train (ndarray): a (n, mtrain) array with training set
        X_cv (ndarry): a (n, mcv) array with cross validation set
        X_test (ndarray): a (n, mtest) array with test set
        Y_train (ndarray): a (n,mtrain) array with training set labels
        Y_cv (ndarray): a (n,mcv) array with cv set labels
        Y_test (ndarray): a (n, mtest) array with test set labels
    """
    # Split data into train, cv and test
    X_train, X_mid, Y_train, Y_mid = train_test_split(X, Y, test_size=1-training_set_ratio, random_state=random_state)
    X_cv, X_test, Y_cv, Y_test = train_test_split(X_mid, Y_mid, test_size=cv_test_set_ratio, random_state=random_state)

    # Transpose target outputs and features to match what my network is working with
    X_train = X_train.T
    X_cv = X_cv.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_cv = Y_cv.T
    Y_test = Y_test.T

    # Normalize data
    mean = np.mean(X_train, axis=(1,0), keepdims=True)
    std = np.std(X_train, axis=(1,0), keepdims=True)
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_cv_scaled = (X_cv - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return (X_train_scaled, X_cv_scaled, X_test_scaled, Y_train, Y_cv, Y_test)