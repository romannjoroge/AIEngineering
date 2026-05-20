import numpy as np

def load_data():
    """
    Returns training and output labels for the decision tree example:
    
    Returns:
        X_train (ndarray): 10 X 3 matrix with training examples
               : Features include age, likes dogs, likes gravity
        y_train (ndarray): 12, matrix with output labels, showing if example is going to be an astronaut
    """
    X_train = np.array([
        [24, 0, 0],
        [30, 1, 1],
        [36, 0, 1],
        [36, 0, 0],
        [42, 0, 0],
        [44, 1, 1],
        [46, 1, 0],
        [47, 1, 1],
        [47, 0, 1],
        [51, 1, 1],
    ])
    y_train = np.array([0,1,1,0,0,1,0,1,0,1])
        
    return (X_train, y_train)
    