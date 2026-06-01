import numpy as np
import numpy.typing as npt

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
    
