import numpy as np
import numpy.typing as npt
import math

w_1_end_index = 9
b_1_end_index = 10
w_2_end_index = 2672

# Convert concatenated params to individual parameters
def condensed_to_params(
    theta: npt.NDArray
):
    """ 
    Function to convert condensed params or derivatives back into their component pararams or derivatives
    
    Args:
        theta (ndarray): a (2793,) array with condensed params or derivatives
        
    Returns:
        w_1 (ndarray): a (1,9) array with first weights / derivative of first weights
        b_1 (scalar): bias / derivative of bias of first layer
        w_2 (ndarray): a (121, 22) array with weights / derivative of weights of second layer
        b_2 (ndarray): a (121, 1) array with bias / derivative of bias of second layer
    """
    w_1 = np.array(theta[0:w_1_end_index]).reshape((1, 9))
    b_1 = theta[b_1_end_index - 1]
    w_2 = np.array(theta[b_1_end_index:w_2_end_index]).reshape((121, 22))
    b_2 = np.array(theta[w_2_end_index:]).reshape((121,1))
    
    return (w_1, b_1, w_2, b_2)

# Concating Wp, bp, Wt and bt into theta
def params_to_condensed(
    w_p: npt.NDArray,
    b_p: float,
    w_t: npt.NDArray,
    b_t: npt.NDArray
) -> np.ndarray[tuple[2793], np.dtype[np.float64]]:
    """
    Function to condense model parameters into a single vector
    
    Args:
        w_p (ndarray) - (1, 9) array with weights of player portion of network
        b_p (scalar) - bias for player network
        w_t (ndarray) - (121, 22) array with weights of team portion of network
        b_t (ndarray) - (121, 1) array with bias of team portion of network
        
    Returns:
        theta (ndarray) - (2793,) condensed vector
    """
    theta = []
    # Concatenating w_p
    theta.extend(w_p.flatten(order='C'))
    # Concatenating b_p
    theta.append(np.float64(b_p))
    # Concatenating w_t
    theta.extend(w_t.flatten(order='C'))
    # Concatenating b_t
    theta.extend(b_t.flatten(order='C'))
    return np.array(theta)

# Non-vectorized per example forward pass
def f_x(
    x_i: npt.NDArray,
    theta: npt.NDArray
):
    """ 
    Function to get prediction of model from input and concatenated parameters
    
    Args:
        x_i (ndarray): a (22, 9) array with training features
        theta (ndarray): a (2793,) array with concatenated parameters
        
    Returns:
        y_pred_i (ndarray): a (121, 1) array with predicted game outcomes
        z_1_i (ndarray): cached z_2_i value
        a_1_i (ndarray): cached a_1_i value
    """
    w_1, b_1, w_2, b_2 = condensed_to_params(theta=theta)
    
    z_1_i = np.matmul(x_i, w_1.T) + b_1
    a_1_i = np.maximum(0, z_1_i)
    z_2_i = np.matmul(w_2, a_1_i) + b_2
    shifted_logits = z_2_i - np.max(z_2_i, axis=0, keepdims=True)
    e_zi = np.exp(shifted_logits)
    y_pred_i = e_zi / np.sum(e_zi, axis=0, keepdims=True)
    
    return (y_pred_i, z_1_i, a_1_i)

# Vectorized version of forward pass
def predict(
    X: npt.NDArray,
    theta: npt.NDArray
):
    """ 
    Vectorized implementation of forward propagation
    
    Args:
        X (ndarray): a (m, 22, 9) array with m training examples
        theta (ndarray): a (2793,) array with concatenated paramaters
        
    Returns:
        y_pred (ndarray): a (m, 121, 1) array with predicted game outcomes
        Z_1 (ndarray): cached z_1 value
        A_1 (ndarray): cached a_1 value
    """
    w_1, b_1, w_2, b_2 = condensed_to_params(theta=theta)
    
    Z_1 = np.matmul(X, w_1.T) + b_1
    A_1 = np.maximum(0, Z_1)
    Z_2 = np.matmul(w_2, A_1) + b_2
    
    shifted_logits = Z_2 - np.max(Z_2, axis=1, keepdims=True)
    e_z = np.exp(shifted_logits)
    Y_pred = e_z / np.sum(e_z, axis=1, keepdims=True)
    
    return (Y_pred, Z_1, A_1)

def J(
    X: npt.NDArray,
    Y: npt.NDArray,
    theta: npt.NDArray
):
    """ 
    Return the cost J given concatenated parameters theta
    
    Args:
        X (ndarray) : a (m, 22, 9) with m training examples
        Y (ndarray) : a (m, 121, 1) with target labels
        theta (ndarray) : a (2793,) array with concatenated parameters
        
    Returns:
        J (scalar): cost
    """
    m = X.shape[0]
    Y_pred, _, _ = predict(X=X, theta=theta)
    J = -np.sum(Y * np.log(Y_pred + 1e-15)) / m
    return J

def backprop(
    X: npt.NDArray,
    Y: npt.NDArray,
    theta: npt.NDArray
):
    """ 
    Get derivatives that will be used by gradient descent to minimize cost function
    
    Args:
        X (ndarray): a (m, 22, 9) array with m training examples
        Y (ndarray): a (m, 121, 1) array with m training output classes
        theta (ndarray): a (2793,) array with concatenated parameters
        
    Returns:
        dtheta (ndarray): a (2793,) array with concatenated derivatives
    """
    m = X.shape[0]
    Y_pred, Z_1, A_1 = predict(
        X=X,
        theta=theta
    )
    _, _, w_2, _ = condensed_to_params(theta=theta)
    
    dZ_2 = Y_pred - Y
    dw_2 = np.mean(np.matmul(dZ_2, np.transpose(A_1, (0, 2, 1))), axis=0)
    db_2 = np.mean(dZ_2, axis=0)
    da_1 = np.matmul(w_2.T, dZ_2)
    dZ_1 = np.where(Z_1 < 0, 0, da_1)
    dw_1 = np.mean(np.matmul(np.transpose(X, (0, 2, 1)), dZ_1).transpose(0, 2, 1), axis=0)
    db_1 = np.mean(np.sum(dZ_1, axis=1)[:, 0])

    return params_to_condensed(
        w_p=dw_1,
        b_p=db_1,
        w_t=dw_2,
        b_t=db_2
    )
    
def gradient_descent(
    X: npt.NDArray, 
    theta: npt.NDArray,
    Y: npt.NDArray,
    alpha: float,
    num_iters: int,
    J_n: int = 10
):
    """ 
    This function will minimize the cost of the model using gradient descent and return found weights
    
    Args:
        X (ndarray) - a (m, 22, 9) array serving as input to model
        theta (ndarray) - a (2739,) array with contatenated parameters
        Y (ndarray) - a (m, 2, 1) array with target outputs for model
        alpha (scalar) - learning rate
        num_iters (scalar) - number of times to run gradient descent
        J_n (scalar) - interval to store training example
        
    Return:
        w_p (ndarray) - a (1, 9) array that minimizes cost
        b_p (scalar) - bias that minimized cost
        w_t (ndarray) - a (2, 22) array that minimizes cost
        b_t (scalar) - bias that minimizes cost
        Js (list) - list of training costs and epoch at different iterations of training
    """
    Js = []
    
    for epoch in range(num_iters):
        # Get derivatives
        dtheta = backprop(
            X=X,
            Y=Y,
            theta=theta
        )
        w_1, b_1, w_2, b_2 = condensed_to_params(theta=theta)
        dw_1, db_1, dw_2, db_2 = condensed_to_params(theta=dtheta)
        
        # Update weights
        w_1 = w_1 - (alpha * dw_1)
        w_2 = w_2 - (alpha * dw_2)
        b_1 = b_1 - (alpha * db_1)
        b_2 = b_2 - (alpha * db_2)
        
        theta = params_to_condensed(w_p=w_1,b_p=b_1,w_t=w_2,b_t=b_2)
        cost = J(X=X,Y=Y,theta=theta)
        
        # Print result
        print(f"Cost at epoch {epoch}/{num_iters} = {cost}")
        
        if epoch % J_n == 0 or epoch == num_iters - 1:
            Js.append((epoch, cost))
        
    w_1, b_1, w_2, b_2 = condensed_to_params(theta=theta)
    return (w_1, w_2, b_1, b_2, Js)

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

def accuracy_score(y_pred, y_label):
    """ 
    Return the ratio of correct responses across the entire test set.add
    
    Args:
        y_pred (ndarray): (m, 121, 1) array with predictions for each class from the model from the model
        y_label(ndarray): (m, 121, 1) array with the actual labels
        
    Returns:
        accuracy (scalar): ratio of correct responses to all responses
    """
    predictions = np.argmax(y_pred, axis=1)
    actual = np.argmax(y_label, axis=1)
    num_correct = np.sum(predictions == actual)
    return num_correct / y_pred.shape[0]

class PredictionRecall:
    def __init__(self, t_p: int, f_p: int, f_n:int, total: int, label: str):
        """ 
        Return object with prediction recall values
        
        Args:
            label (str): Name of the class
            t_p (int): Total true positive for the given label
            f_p (int): Total false positives for given label
            total (int): Total items with given label
            f_n (int): Total false negatives for given label
        """
        self.total = total
        self.t_p = t_p
        self.f_p = f_p
        self.f_n = f_n
        self.label = label
        
    def __repr__(self):
        return f"{self.label} with t_p: {self.t_p}, f_p: {self.f_p}, f_n: {self.f_n} and total: {self.total}"

def get_precision_recall(y_pred, y_label) -> list[PredictionRecall]:
    """ 
    Returns the precision and recall of the model based on the predictions it has made
    
    Args:
        y_pred (ndarray) - a (m, 121, 1) array with the predictions of the model
        y_label (ndarray) - a (m, 121, 1) array with the actual values
        
    Returns:
        prediction_recall (list) - list of prediction recall values
    """
    predictions = [interpret_probabilities(prob) for prob in y_pred]
    actual = [interpret_probabilities(prob) for prob in y_label]
    all_items = predictions.copy()
    all_items.extend(actual)
    labels = set(all_items)
    
    m = len(actual)
    precisions = []
    
    for label in labels:
        total = 0
        false_positive = 0
        false_negative = 0
        true_positive = 0
        
        for i in range(m):
            entry = actual[i]
            pred = predictions[i]
            
            if entry == label:
                total += 1
                
            if entry == label and pred == entry:
                true_positive += 1
            elif entry != label and pred == label:
                false_positive += 1
        false_negative = total - true_positive
        
        precisions.append(PredictionRecall(
            t_p=true_positive,
            f_p=false_positive,
            f_n=false_negative,
            total=total,
            label=label
        ))
    
    return precisions
    
def precision_recall(y_pred, y_label):
    """ 
        Print precision recall for model predictions
        
        Args:
            y_pred (ndarray): a (m, 121, 1) array with the predictions of the model
            y_label (ndarray): a (m, 121, 1) array with labels of the model
    """ 
    precisions = get_precision_recall(y_pred=y_pred, y_label=y_label)
    
    # Order precisions by totals
    n = len(precisions)
    for i in range(n):
        swapped = False
        
        for j in range(0, n-i-1):
            # Traverse array from 0 to n - i - 1
            # Swap if the element found is lesser than next element
            if precisions[j].total < precisions[j+1].total:
                precisions[j], precisions[j + 1] = precisions[j + 1], precisions[j]
                swapped = True
        
        if (swapped == False):
            break
    
    
    print(f"    Precision and recall")
    for prec in precisions:
        precision = 0 if prec.f_p + prec.t_p == 0 else (prec.t_p / (prec.f_p + prec.t_p))
        recall = 0 if prec.t_p + prec.f_n == 0 else (prec.t_p) / (prec.t_p + prec.f_n)
        p_1 = 0 if precision == 0 else 1 / precision
        r_1 = 0 if recall == 0 else 1 / recall
        f1_score = 1 / 0.5 * (p_1 + r_1)
        n_predictions = prec.t_p + prec.f_p
        print(f"{prec.label.center(5)}  TP:{prec.t_p}  FP:{prec.f_p}  FN:{prec.f_n}: Prec:{precision:.4f}  Rec:{recall:.4f}  F1:{f1_score:.4f}  #Pred: {n_predictions}  Sample:{prec.total}")
        
def win_prediction_accuracy(y_pred, y_label):
    """ 
    Returns the ratio of accuratey predicted win / loss outcomes
    
    Args:
        y_pred (ndarray) - a (m, 121, 1) with model's predictions
        y_label (ndarray) - a (m, 121, 1) with actual labels
        
    Returns
        win_loss_accuracy (scalar) - ratio of correctly predicted wins / losses
    """
    win = 0
    draw = 1
    loss = 2
    
    predictions = np.argmax(y_pred, axis=1)
    predicted_wins = []
    actual = np.argmax(y_label, axis=1)
    actual_wins = []
    m = len(actual)
    
    for i in range(m):
        predicted_outcome = predictions[i][0]
        predicted_home = math.floor(predicted_outcome / 11)
        predicted_away = predicted_outcome % 11
        predicted_label = win if predicted_home > predicted_away else draw if predicted_home == predicted_away else loss
        predicted_wins.append(predicted_label)
        
        actual_outcome = actual[i][0]
        actual_home = math.floor(actual_outcome / 11)
        actual_away = actual_outcome % 11
        actual_label = win if actual_home > actual_away else draw if actual_home == actual_away else loss
        actual_wins.append(actual_label)
    
    return sum([predicted_wins[i] == actual_wins[i] for i in range(m)]) / m