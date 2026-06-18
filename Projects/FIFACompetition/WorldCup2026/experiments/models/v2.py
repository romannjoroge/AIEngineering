import numpy as np
import numpy.typing as npt

class ModelV2:
    """ 
    Version of model that uses 2 layers
    """
    def __init__(
        self,
        W1_shape: tuple[int, int],
        b1_shape: tuple[int, int],
        W2_shape: tuple[int, int],
        b2_shape: tuple[int, int],
        seed: int
    ):
        """ 
        This function defines the shape of the parameters used by the model. From this it can initialize weights
        
        Args:
            W1_shape (tuple): tuple with shape of weights for first layer
            b1_shape (tuple): tuple with shape of biases for first layer
            W2_shape (tuple): tuple with shape of weights for second layer
            b2_shape (tuple): tuple with shapes of biases for second layer
            seed (scalar): seed used to generate random numbers
        """
        assert W1_shape[0] == b1_shape[0], "Weights and biases for first layer should have the same number of units"
        assert W2_shape[0] == b2_shape[0], "Weights and biases for second layer should have same number of units"
        assert W2_shape[1] == W1_shape[0], "Weights of second layer should accept same number of inputs as units of first layer"
        
        self.W1_shape = W1_shape
        self.b1_shape = b1_shape
        self.W2_shape = W2_shape
        self.b2_shape = b2_shape
        self.rng = np.random.default_rng(seed=seed)
        
    def initialize_weights(self):
        """ 
        Initialize weights with the indicated shape
        
        Return:
            W1 (ndarray): initial weights for first layer
            b1 (ndarray): initial bias for first layer
            W2 (ndarray): initial weights for second layer
            b2 (ndarray): initial bias for second layer
        """
        initial_W1 = self.rng.random(self.W1_shape) * np.sqrt(1 / self.W1_shape[1])
        initial_b1 = np.zeros(self.b1_shape)
        initial_W2 = self.rng.random(self.W2_shape) * np.sqrt(1 / self.W2_shape[1])
        initial_b2 = np.zeros(self.b2_shape)
        
        return (initial_W1, initial_b1, initial_W2, initial_b2)
    
    # Vectorized version of forward pass
    def f_x(
        X: npt.NDArray,
        W1: npt.NDArray,
        b1: npt.NDArray,
        W2: npt.NDArray,
        b2: npt.NDArray
    ):
        """ 
        Vectorized implementation of forward propagation
        
        Args:
            X (ndarray): array with m training examples
            W1 (ndarray): array with weights of first layer
            b1 (ndarray): array with biases of first layer
            W2 (ndarray): array with weights of second layer
            b2 (ndarray): array with biases of second layer
            
        Returns:
            y_pred (ndarray): a (121, m) array with predicted game outcomes
            Z_1 (ndarray): cached z_1 value
            A_1 (ndarray): cached a_1 value
        """
        print(W1.shape, X.shape, b1.shape)
        Z_1 = np.matmul(W1, X) + b1
        print(Z_1.shape)
        A_1 = np.maximum(0, Z_1)
        print(A_1.shape, W2.shape, b2.shape)
        Z_2 = np.matmul(W2, A_1) + b2
        print(Z_2.shape)
        
        shifted_logits = Z_2 - np.max(Z_2, axis=1, keepdims=True)
        e_z = np.exp(shifted_logits)
        Y_pred = e_z / np.sum(e_z, axis=1, keepdims=True)
        print(Y_pred.shape)
        
        return (Y_pred, Z_1, A_1)

    def J(
        Y_pred: npt.NDArray,
        Y: npt.NDArray,
    ):
        """ 
        Return the cost J given concatenated parameters theta
        
        Args:
            Y_pred (ndarray): a (121, m) array with predicted labels
            Y (ndarray) : a (121, m) with target labels
            
        Returns:
            J (scalar): cost
        """
        m = Y.shape[1]
        epsilon = 1e-15
        pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
        step_1 = Y * np.log(pred_clipped)
        example_losses = -np.sum(step_1, axis=0)
        average_loss = np.sum(example_losses) / m
        
        return average_loss

    def backprop(
        X: npt.NDArray,
        Y: npt.NDArray,
        W1: npt.NDArray,
        b1: npt.NDArray,
        W2: npt.NDArray,
        b2: npt.NDArray
    ):
        """ 
        Get derivatives that will be used by gradient descent to minimize cost function
        
        Args:
            X (ndarray): array with m training examples
            Y (ndarray): array with m training output classes
            W1 (ndarray): array with weights of first layer
            b1 (ndarray): array with bias of first layer
            W2 (ndarray): array with weights of second layer
            b2 (ndaray): array with bias of second layer
            
        Returns:
            dW1 (ndarray): derivative of J with respect to W1
            db1 (ndarray): derivative of J with respect to b1
            dW2 (ndarray): derivative of J with respect to W2
            db2 (ndarray): derivative of J with respect to b2
        """
        print("Starting backprop")
        m = X.shape[1]
        Y_pred, Z1, A1 = ModelV2.f_x(X=X, W1=W1, W2=W2, b1=b1, b2=b2)
        print("Got prediction z1 and a1 in backprop")
        
        
        dZ2 = Y_pred - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1) / m
        dA1 = np.matmul(W2.T, dZ2)
        dZ1 = np.where(Z1 < 0, np.zeros(Z1.shape), dA1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1) / m
        
        print("finished backprop")
        return (dW1, db1, dW2, db2)
        
    def train(
        self,
        X: npt.NDArray, 
        W1: npt.NDArray,
        b1: npt.NDArray,
        W2: npt.NDArray,
        b2: npt.NDArray,
        Y: npt.NDArray,
        alpha: float,
        num_iters: int,
        J_n: int = 10
    ):
        """ 
        This function will minimize the cost of the model using gradient descent and return found weights
        
        Args:
            X (ndarray) - a (m, 22, 9) array serving as input to model
            W1 (ndarray): array with weights of first layer
            b1 (ndarray): array with bias of first layer
            W2 (ndarray): array with weights of second layer
            b2 (ndaray): array with bias of second layer
            Y (ndarray) - a (m, 2, 1) array with target outputs for model
            alpha (scalar) - learning rate
            num_iters (scalar) - number of times to run gradient descent
            J_n (scalar) - interval to store training example
            
        Return:
            W1 (ndarray) - updated weights of first layer
            W2 (scalar) - updated weights of second layer
            b1 (ndarray) - updated bias of first layer
            b2 (scalar) - updated bias of second layer
            Js (list) - list of training costs and epoch at different iterations of training
        """
        Js = []
        
        for epoch in range(num_iters):
            dW1, db1, dW2, db2 = ModelV2.backprop(X=X, Y=Y, W1=W1, b1=b1, W2=W2, b2=b2)
            
            # Update weights
            W1 = W1 - (alpha * dW1)
            W2 = W2 - (alpha * dW2)
            b1 = b1 - (alpha * db1)
            b2 = b2 - (alpha * db2)
            
            
            Y_pred, _, _ = ModelV2.f_x(X=X, W1=W1, b1=b1, W2=W2, b2=b2)
            print(f"f_x result shape {Y_pred.shape}")
            cost = ModelV2.J(Y_pred=Y_pred, Y=Y)
            
            # Print result
            print(f"Cost at epoch {epoch}/{num_iters} = {cost}")
            
            if epoch % J_n == 0 or epoch == num_iters - 1:
                Js.append((epoch, cost))
            
        return (W1, W2, b1, b2, Js)