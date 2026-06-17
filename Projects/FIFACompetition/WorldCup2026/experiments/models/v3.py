import numpy as np
import numpy.typing as npt
import math

class ModelV3:
    """ 
    This architecture has 3 layers with the final layer having 121 units to predict one of 121 different outputs
    """
    def __init__(
        self,
        W1_shape: tuple[int, int],
        b1_shape: tuple[int, int],
        W2_shape: tuple[int, int],
        b2_shape: tuple[int, int],
        W3_shape: tuple[int, int],
        b3_shape: tuple[int, int],
        seed: int
    ):
        """  
        This initializes the shapes of the different layers in the network
        
        Args:  
            W1_shape (tuple): shape of weights of first layer
            b1_shape (tuple): shape of bias of first layer
            W2_shape (tuple): shape of weights of second layer
            b2_shape (tuple): shape of bias of second layer
            W3_shape (tuple): shape of weights of third layer
            b3_shape (tuple): shape of bias of third layer
            seed (scalar): random state initializer
        """
        assert b1_shape[0] == W1_shape[0], "Units in weight and bias of first layer must be the same"
        assert b2_shape[0] == W2_shape[0], "Units in weight and bias of second layer must be the same"
        assert b3_shape[0] == W3_shape[0], "Units in weight and bias of third layer must be the same"
        assert W2_shape[1] == W1_shape[0], "Number of weights for each unit in second layer must be same as number of units in first layer"
        assert W3_shape[1] == W2_shape[0], "Number of weights for each unit in third layer must be same as number of units in second layer"
        
        self.W1_shape = W1_shape
        self.b1_shape = b1_shape
        self.W2_shape = W2_shape
        self.b2_shape = b2_shape
        self.W3_shape = W3_shape
        self.b3_shape = b3_shape
        self.rng = np.random.default_rng(seed=seed)
        
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
    
    def J(
        Y_pred: npt.NDArray,
        Y: npt.NDArray,
        lambd: float,
        W1: npt.NDArray,
        W2: npt.NDArray,
        W3: npt.NDArray
    ):
        """ 
        Function to get cost of the model
        
        Args:
            Y_pred (ndarray): a (121, m) array with predicted labels
            Y (ndarray): a (121, m) array with target labels for training examples
            lambd (scalar): regularization parameter
            W1 (ndarray): weights of the first layer
            W2 (ndarray): weights of second layer
            W3 (ndarray): weights of third layer
            
        Returns:
            cost (scalar): Training cost of model
        """
        m = Y.shape[1]
        epsilon = 1e-15
        pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
        step_1 = Y * np.log(pred_clipped)
        example_losses = -np.sum(step_1, axis=0)
        average_loss = np.sum(example_losses) / m
        
        # Regularization
        reg_layer_1 = np.sum(W1 ** 2)
        reg_layer_2 = np.sum(W2 ** 2)
        reg_layer_3 = np.sum(W3 ** 2)
        reg = (lambd / (2 * m)) * (reg_layer_1 + reg_layer_2 + reg_layer_3)
        return average_loss + reg
    
    def initialize_weights(self):
        """  
        Initialize random weights that will be used by the model
        
        These weights are meant to reduce effects of vanishing / exploding gradients
        
        Returns:
            W1 (ndarray): initial weights of first layer
            b1 (ndarray): initial bias of first layer
            W2 (ndarray): initial weights of second layer
            b2 (ndarray): initial bias of second layer
            W3 (ndarray): initial weights of third layer
            b3 (ndarray): initial bias of third layer
        """
        initial_W1 = self.rng.random(self.W1_shape) * np.sqrt(1 / self.W1_shape[1])
        initial_B1 = np.zeros(self.b1_shape)
        initial_W2 = self.rng.random(self.W2_shape) * np.sqrt(1 / self.W2_shape[1])
        initial_B2 = np.zeros(self.b2_shape)
        initial_W3 = self.rng.random(self.W3_shape) * np.sqrt(1 / self.W3_shape[1])
        initial_B3 = np.zeros(self.b3_shape)
        
        return (initial_W1, initial_B1, initial_W2, initial_B2, initial_W3, initial_B3)
    
    def backprop(
        X: npt.NDArray,
        Y: npt.NDArray,
        lambd: float,
        W1: npt.NDArray,
        b1: npt.NDArray,
        W2: npt.NDArray,
        b2: npt.NDArray,
        W3: npt.NDArray,
        b3: npt.NDArray
    ):
        """ 
        Get derivative of parameters that will be used to optimize the model
        
        Args:
            X (ndarray): a (18,m) array with m training examples
            Y (ndarray): a (121,m) array with m target labels
            lambd (scalar): regularization parameter
            W1 (ndarray): weights of first layer
            b1 (ndarray): bias of first layer
            W2 (ndarray): weights of second layer
            b2 (ndarray): bias of second layer
            W3 (ndarray): weights of third layer
            b3 (ndarray): bias of third layer
            
        Returns:
            dW1 (ndarray): derivative of first layer's weights
            db1 (ndarray): derivative of first layer's bias
            dW2 (ndarray): derivative of second layer's weights
            db2 (ndarray): derivative of second layer's bias
            dW3 (ndarray): derivative of third layer's weights
            db3 (ndarray): derivative of third layer's bias
        """
        m = X.shape[1]
        Y_pred, A1, Z1, A2, Z2 = ModelV3.f_x(X=X, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
        
        dZ3 = Y_pred - Y
        dW3 = (np.matmul(dZ3, A2.T) / m) + ((lambd / m) * W3)
        db3 = (np.sum(dZ3, axis=1) / m).reshape(-1, 1)
        dA2 = np.matmul(W3.T, dZ3)
        dZ2_zeros = np.zeros((54, m))
        dZ2 = np.where(Z2 < 0, dZ2_zeros, dA2)
        dW2 = (np.matmul(dZ2, A1.T) / m) + ((lambd / m) * W2)
        db2 = (np.sum(dZ2, axis=1) / m).reshape(-1, 1)
        dA1 = np.matmul(W2.T, dZ2)
        dZ1_zeros = np.zeros((27, m))
        dZ1 = np.where(Z1 < 0, dZ1_zeros, dA1)
        dW1 = (np.matmul(dZ1, X.T) / m) + ((lambd / m) * W1)
        db1 = (np.sum(dZ1, axis=1) / m).reshape(-1, 1)
        
        return (dW1, db1, dW2, db2, dW3, db3)
    
    def train(
        self,
        X: npt.NDArray,
        Y: npt.NDArray,
        W1: npt.NDArray,
        b1: npt.NDArray,
        W2: npt.NDArray,
        b2: npt.NDArray,
        W3: npt.NDArray,
        b3: npt.NDArray,
        lambd: float,
        beta_m: float,
        beta_r: float,
        alpha: float,
        batch_size: int,
        num_epochs: int
    ):
        """ 
        Adam cost optmizer algorithm
        
        Args:
            X (ndarray): a (18,m) array with m training examples
            Y (ndarray): a (121,m) array with m training examples
            W1 (ndarray): weights of the first layer
            b1 (ndarray): bias of the first layer
            W2 (ndarray): weights of second layer
            b2 (ndarray): bias of second layer
            W3 (ndarray): weights of third layer
            b3 (ndarray): bias of third layer
            lambd (scalar): regularization parameter
            beta_m (scalar): beta value for momentum EMA
            beta_r (scalar): beta value for RMSprop EMA
            alpha (scalar): learning rate
            batch_size (scalar): number of examples in a batch
            num_epochs (scalar): number of times to go through entire dataset
            
        Return:
            W1 (ndarray): updated w1
            b1 (ndarray): updated b1
            W2 (ndarray): updated w2
            b2 (ndarray): updated b2
            W3 (ndarray): updated W3
            b3 (ndarray): updated b3
            Js (ndarray): list of iterations and costs
        """
        m = X.shape[1]
        num_batches = math.ceil(m / batch_size)
        Y_pred, _, _, _, _ = ModelV3.f_x(X=X, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
        cost = ModelV3.J(Y_pred=Y_pred, Y=Y, lambd=lambd, W1=W1, W2=W2, W3=W3)
        Js = [(0, cost)]
        epsilon = 1e-8
        
        sdW1 = np.zeros(self.W1_shape)
        sdb1 = np.zeros(self.b1_shape)
        sdW2 = np.zeros(self.W2_shape)
        sdb2 = np.zeros(self.b2_shape)
        sdW3 = np.zeros(self.W3_shape)
        sdb3 = np.zeros(self.b3_shape)
        vdW1 = np.zeros(self.W1_shape)
        vdb1 = np.zeros(self.b1_shape)
        vdW2 = np.zeros(self.W2_shape)
        vdb2 = np.zeros(self.b2_shape)
        vdW3 = np.zeros(self.W3_shape)
        vdb3 = np.zeros(self.b3_shape)
        
        for epoch in range(num_epochs):
            for t in range(num_batches):
                # Get Xt, Yt
                start = t * batch_size
                if t == num_batches - 1:
                    Xt = X[:, start:]
                    Yt = Y[:, start:]
                else:
                    end = start + batch_size
                    Xt = X[:, start:end]
                    Yt = Y[:, start:end]
                
                # Get derivatives with Xt, Yt
                dw1, db1, dw2, db2, dw3, db3 = ModelV3.backprop(X=Xt, Y=Yt, lambd=lambd, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
                
                # Get sdw, vdw,
                iteration = (epoch * num_batches) + t + 1

                sdW1 = (beta_r * sdW1) + ((1 - beta_r) * dw1 ** 2)
                sdb1 = (beta_r * sdb1) + ((1 - beta_r) * db1 ** 2)
                sdW2 = (beta_r * sdW2) + ((1 - beta_r) * dw2 ** 2)
                sdb2 = (beta_r * sdb2) + ((1 - beta_r) * db2 ** 2)
                sdW3 = (beta_r * sdW3) + ((1 - beta_r) * dw3 ** 2)
                sdb3 = (beta_r * sdb3) + ((1 - beta_r) * db3 ** 2)
                vdW1 = (beta_m * vdW1) + ((1 - beta_m) * dw1)
                vdb1 = (beta_m * vdb1) + ((1 - beta_m) * db1)
                vdW2 = (beta_m * vdW2) + ((1 - beta_m) * dw2)
                vdb2 = (beta_m * vdb2) + ((1 - beta_m) * db2)
                vdW3 = (beta_m * vdW3) + ((1 - beta_m) * dw3)
                vdb3 = (beta_m * vdb3) + ((1 - beta_m) * db3)
                
                sdw1_corrected = sdW1 / (1 - (beta_r ** iteration))
                sdb1_corrected = sdb1 / (1 - (beta_r ** iteration))
                sdw2_corrected = sdW2 / (1 - (beta_r ** iteration))
                sdb2_corrected = sdb2 / (1 - (beta_r ** iteration))
                sdw3_corrected = sdW3 / (1 - (beta_r ** iteration))
                sdb3_corrected = sdb3 / (1 - (beta_r ** iteration))
                vdw1_corrected = vdW1 / (1 - (beta_m ** iteration))
                vdb1_corrected = vdb1 / (1 - (beta_m ** iteration))
                vdw2_corrected = vdW2 / (1 - (beta_m ** iteration))
                vdb2_corrected = vdb2 / (1 - (beta_m ** iteration))
                vdw3_corrected = vdW3 / (1 - (beta_m ** iteration))
                vdb3_corrected = vdb3 / (1 - (beta_m ** iteration))
                
                # Update parameters
                W1 = W1 - (alpha * (vdw1_corrected / np.sqrt(sdw1_corrected + epsilon)))
                b1 = b1 - (alpha* (vdb1_corrected / np.sqrt(sdb1_corrected + epsilon)))
                W2 = W2 - (alpha * (vdw2_corrected / np.sqrt(sdw2_corrected + epsilon)))
                b2 = b2 - (alpha* (vdb2_corrected / np.sqrt(sdb2_corrected + epsilon)))
                W3 = W3 - (alpha * (vdw3_corrected / np.sqrt(sdw3_corrected + epsilon)))
                b3 = b3 - (alpha* (vdb3_corrected / np.sqrt(sdb3_corrected + epsilon)))
                
                # Get cost for Xt, Yt
                Y_pred_t, _, _, _, _ = ModelV3.f_x(X=Xt, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
                cost = ModelV3.J(Y=Yt, Y_pred=Y_pred_t, lambd=lambd, W1=W1, W2=W2, W3=W3)
                
                # Store cost and iteration
                Js.append((iteration, cost))
                print(f"Cost at epoch {epoch + 1} / {num_epochs} in batch {t + 1} / {num_batches}: {cost}")
                
        return (W1, b1, W2, b2, W3, b3, Js)
        
        