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

        self.W1_end_index = W1_shape[0] * W1_shape[1]
        self.b1_end_index = self.W1_end_index + (b1_shape[0] * b1_shape[1])
        self.W2_end_index = self.b1_end_index + (W2_shape[0] * W2_shape[1])
        
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
        Z_1 = np.matmul(W1, X) + b1
        A_1 = np.maximum(0, Z_1)
        Z_2 = np.matmul(W2, A_1) + b2
        
        shifted_logits = Z_2 - np.max(Z_2, axis=0, keepdims=True)
        e_z = np.exp(shifted_logits)
        Y_pred = e_z / np.sum(e_z, axis=0, keepdims=True)
        
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
        m = X.shape[1]
        Y_pred, Z1, A1 = ModelV2.f_x(X=X, W1=W1, W2=W2, b1=b1, b2=b2)
        
        dZ2 = Y_pred - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = (np.sum(dZ2, axis=1) / m).reshape(-1, 1)
        dA1 = np.matmul(W2.T, dZ2)
        dZ1 = np.where(Z1 < 0, np.zeros(Z1.shape), dA1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = (np.sum(dZ1, axis=1) / m).reshape(-1, 1)
        
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
            cost = ModelV2.J(Y_pred=Y_pred, Y=Y)
    
            # Print result
            print(f"Cost at epoch {epoch}/{num_iters} = {cost}")
            
            if epoch % J_n == 0 or epoch == num_iters - 1:
                Js.append((epoch, cost))
            
        return (W1, W2, b1, b2, Js)

    def _concatenate_parameters(
        W1: npt.NDArray,
        b1: npt.NDArray,
        W2: npt.NDArray,
        b2: npt.NDArray
    ): 
        """ 
        Function to concatenate paramaters into a single array

        Args:
            W1 (ndarray): weights of the first layer
            b1 (ndarray): bias of the first layer
            W2 (ndarray): weights of the second layer
            b2 (ndarray): bias of the second layer

        Returns:
            theta (ndarray): single array with concatenated weights
        """
        concatenated = []
        
        concatenated.extend(W1.flatten(order='C'))
        concatenated.extend(b1.flatten(order='C'))
        concatenated.extend(W2.flatten(order='C'))
        concatenated.extend(b2.flatten(order='C'))

        return np.array(concatenated)

    def _unconcatenate_parameters(
        self,
        theta: npt.NDArray
    ): 
        """ 
        Function to unconcatenate parameters

        Args:
            theta (ndarray): concatenated weights 

        Returns:
            W1 (ndarray): weights of first layer
            b1 (ndarray): bias of first layer
            W2 (ndarray): weights of second layer
            b2 (ndarray): bias of second layer
        """

        W1 = np.array(theta[0:self.W1_end_index]).reshape(self.W1_shape)
        b1 = np.array(theta[self.W1_end_index:self.b1_end_index]).reshape(self.b1_shape)
        W2 = np.array(theta[self.b1_end_index:self.W2_end_index]).reshape(self.W2_shape)
        b2 = np.array(theta[self.W2_end_index:]).reshape(self.b2_shape)

        return (W1, b1, W2, b2)

    def gradient_checking(
        self,
        X: npt.NDArray,
        Y: npt.NDArray,
        W1: npt.NDArray,
        b1: npt.NDArray,
        W2: npt.NDArray,
        b2: npt.NDArray,
    ):
        """ 
        Confirm the implementation of backprop using gradient checking
        """
        epsilon = 1e-7

        # Concatenate parameters into a single array
        theta = ModelV2._concatenate_parameters(W1=W1, b1=b1, W2=W2, b2=b2)
        dW1, db1, dW2, db2 = ModelV2.backprop(X=X, Y=Y, W1=W1, b1=b1, W2=W2, b2=b2)
        dtheta = ModelV2._concatenate_parameters(W1=dW1, b1=db1, W2=dW2, b2=db2)
        dtheta_approx = np.zeros(theta.shape)

        num_parameters = theta.shape[0]
        # For each parameter
        for param_i in range(num_parameters):
            # Get cost of adding parameter by small number
            theta_right = theta.copy()
            theta_right[param_i] += epsilon
            W1_right, b1_right, W2_right, b2_right = self._unconcatenate_parameters(theta=theta_right)
            Y_pred_right, _, _ = ModelV2.f_x(X=X, W1=W1_right, b1=b1_right, W2=W2_right, b2=b2_right)
            J_right = ModelV2.J(Y_pred=Y_pred_right, Y=Y)

            # Get cost of subtracting parameter by small parameter
            theta_left = theta.copy()
            theta_left[param_i] -= epsilon
            W1_left, b1_left, W2_left, b2_left = self._unconcatenate_parameters(theta=theta_left)
            Y_pred_left, _, _ = ModelV2.f_x(X=X, W1=W1_left, b1=b1_left, W2=W2_left, b2=b2_left)
            J_left = ModelV2.J(Y_pred=Y_pred_left, Y=Y)

            # Get approximated derivative by subtracting both costs and dividing by 2 * small number
            dtheta_approx[param_i] = (J_right - J_left) / (2 * epsilon)

        # Get similarity of approximate derivatives and the ones calculated by backprop
        distance = np.linalg.norm(dtheta_approx - dtheta)
        similarity_check = distance / (np.linalg.norm(dtheta_approx) + np.linalg.norm(dtheta))
        
        # Interpreting results
        print(f"The difference between dtheta_approx and dtheta is {similarity_check}")
        
        # Get accuracy of backprop for each parameter
        W1_problem = 0
        b1_problem = 0
        W2_problem = 0
        b2_problem = 0

        for param_i in range(num_parameters):
            # Get similarity between calculated derivative and approximate
            dtheta_value = dtheta[param_i]
            dtheta_approx_value = dtheta_approx[param_i]
            difference = (dtheta_value - dtheta_approx_value) ** 2
            similarity = difference / (dtheta_value ** 2 + dtheta_approx_value ** 2 + 1e-8) # Adding by small value to prevent scenario of dividing by zero

            if param_i < self.W1_end_index:
                W1_problem += similarity
            elif self.W1_end_index <= param_i < self.b1_end_index:
                b1_problem += similarity
            elif self.b1_end_index <= param_i < self.W2_end_index:
                W2_problem += similarity
            elif self.W2_end_index <= param_i < num_parameters:
                b2_problem += similarity
            else:
                print("Don't know how to handle this value")

        # Print similarities
        print(f"Problem for W1 is {W1_problem}")
        print(f"Problem for b1 is {b1_problem}")
        print(f"Problem for W2 is {W2_problem}")
        print(f"Problem for b2 is {b2_problem}")

        # Print five values of each parameter
        for i in range(0, 5):
            print(f"Calcuated value of dW1 is {dtheta[i]} while approximate is {dtheta_approx[i]}")

        for i in range(self.W1_end_index, self.W1_end_index+5):
            print(f"Calcuated value of db1 is {dtheta[i]} while approximate is {dtheta_approx[i]}")

        for i in range(self.b1_end_index, self.b1_end_index+5):
            print(f"Calcuated value of dW2 is {dtheta[i]} while approximate is {dtheta_approx[i]}")

        for i in range(self.W2_end_index, self.W2_end_index+5):
            print(f"Calcuated value of db2 is {dtheta[i]} while approximate is {dtheta_approx[i]}")
            

