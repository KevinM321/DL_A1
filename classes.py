import numpy as np
import scipy
import random
from matplotlib import pyplot as plt
import scipy.special


# -----------------------------
# Custom Dataset and Dataloader
# -----------------------------
class DataSet:
    """
    Custom Dataset class for loading and normalizing datasets
    """
    def __init__(self, data, labels):
        if isinstance(data, np.ndarray): # if given data
            self.data = data
            self.labels = labels
        else: # if given filename
            self.data = np.load(data)
            self.labels = np.load(labels).squeeze()

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.data[idx], self.labels[idx]

    # calculate and return the mean and standard deviation of the stored dataset
    def get_mean_std(self):
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0) + 1e-9
        return mean, std

    # normalise the stored dataset using the given mean and standard deviation
    def normalise(self, mean: float, std: float):
        self.data = (self.data - mean) / std


class DataLoader:
    """
    Custom DataLoader supporting mini-batch
    """
    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size if batch_size <= len(dataset) else len(dataset)
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        # shuffle all items in dataset at start of loop
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0
        return self
    
    def __next__(self) -> list:
        if self.current_idx >= len(self.indices):
            raise StopIteration

        # select mini-batch item indices based on batch size
        batch_end_idx = self.current_idx + self.batch_size
        batch_indices = self.indices[self.current_idx:batch_end_idx]
        self.current_idx = batch_end_idx

        batch = [self.dataset[i] for i in batch_indices] # create batch using selected indices
        X_batch, y_batch = zip(*batch)
        return np.array(X_batch), np.array(y_batch)
       

# -----------------------------
# MLP Layers
# -----------------------------

class RELU:
    """
    ReLU activation function
    """
    def __init__(self):
        self.input = None
        self.have_params = False
        self.train = True

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.mask = (input > 0).astype(np.float32)  # store as float
        return input * self.mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask
    

class GELU:
    """
    GELU activation function using exact error function
    """
    def __init__(self):
        self.input = None
        self.have_params = False
        self.train = True

    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input

        # calculates input transformation using gaussian error function
        input = input * 0.5 * (1 + scipy.special.erf(input / np.sqrt(2)))
        return input        

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # derivatives are calculated with respect to the exact error function
        phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.pow(self.input, 2))
        Phi = 0.5 * (1 + scipy.special.erf(self.input / np.sqrt(2)))
        return grad * (self.input * phi + Phi)

class GELU2:
    """
    GELU activation function approximated using tanh
    """
    def __init__(self):
        self.input = None
        self.have_params = False
        self.train = True

    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input

        # transformation approximated using tanh as given in original paper
        self.tanh = np.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * np.power(input, 3)))
        return 0.5 * input * (1 + self.tanh) 

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # derivatives of forward transformation function
        tanh_deriv = 1 - np.power(self.tanh, 2)
        inner_deriv = 1 + 3 * 0.044715 * np.power(self.input, 2)

        part_1 = 0.5 * (1 + self.tanh)
        part_2 = 0.5 * self.input * tanh_deriv * np.sqrt(2 / np.pi) * inner_deriv
        
        return grad * (part_1 + part_2)


class Linear:
    """
    Linear layer
    """
    def __init__(self, in_features : int, out_features : int):
        self.input = None
        self.have_params = True
        self.train = True

        # kaiming initialisation of weights
        self.W = np.random.uniform(
                low = -np.sqrt(6. / (in_features + out_features)),
                high = np.sqrt(6. / (in_features + out_features)),
                size = (in_features, out_features)
        )

        # initialise biases, gradients of weights and biases
        self.b = np.zeros(out_features, )
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.dot(input, self.W) + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        input = self.input

        # update gradients
        self.grad_W[...] = np.dot(input.T, grad)
        self.grad_b[...] = np.sum(grad, axis=0)

        return np.dot(grad, self.W.T)

    def params(self) -> list:
        return [
            {'param': self.W, 'grad': self.grad_W},
            {'param': self.b, 'grad': self.grad_b}
        ]

class Dropout:
    """
    Dropout layer used for regularization 
    """
    def __init__(self, chance: float = 0.5):
        self.have_params = False
        self.chance = chance
        self.mask = None
        self.train = True

    def __call__(self, input):
        return self.forward(input)

    def set_train(self, train: bool):
        self.train = train
           
    def forward(self, input: np.ndarray) -> np.ndarray:
        if self.train:
            # generate random mask to drop activations
            self.mask = np.random.rand(*input.shape) > self.chance
            self.mask.astype(np.float32)
            return input * self.mask / (1 - self.chance)
        else:
            return input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask
    

class BatchNorm:
    """
    Batch Normalization layer
    """
    def __init__(self, in_features: int, momentum: float=0.9, eps: float=1e-5):
        self.in_features = in_features
        self.momentum = momentum
        self.eps = eps
        self.train = True
        self.have_params = True 

        # initialise trainable parameters and gradients
        self.gamma = np.ones((1, in_features))
        self.grad_gamma = np.zeros_like(self.gamma)

        self.beta = np.zeros((1, in_features))
        self.grad_beta = np.zeros_like(self.beta)

        self.running_mean = np.zeros((1, in_features))
        self.running_var = np.ones((1, in_features))

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        
        if self.train:
            self.batch_mean = np.mean(input, axis=0, keepdims=True)
            self.batch_var = np.var(input, axis=0, keepdims=True)

            # calculate and store running mean and variance of batches
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var

            # calculate and update normalisation
            self.norm = (input - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
        else:
            self.norm = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.norm + self.beta
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # calculation of derivatives according to original paper
        m = grad.shape[0]
        
        var_inverse = 1 / np.sqrt(self.batch_var + self.eps)
        input_norm = self.input - self.batch_mean

        norm_deriv = grad * self.gamma
        batch_var_deriv = np.sum(norm_deriv * input_norm * -0.5 * np.power(var_inverse, 3), axis=0, keepdims=True)

        mean_deriv_t1 = np.sum(norm_deriv * -var_inverse, axis=0, keepdims=True)
        mean_deriv_t2 =  batch_var_deriv * np.mean(-2 * input_norm, axis=0, keepdims=True)
        batch_mean_deriv = mean_deriv_t1 + mean_deriv_t2

        input_deriv = norm_deriv * var_inverse + batch_var_deriv * (2 * input_norm / m) + batch_mean_deriv / m

        # update gradients
        self.grad_gamma[...] = np.sum(grad * self.norm, axis=0, keepdims=True)
        self.grad_beta[...] = np.sum(grad, axis=0, keepdims=True)

        return input_deriv

    def params(self) -> list:
        return [
            {'param': self.gamma, 'grad': self.grad_gamma},
            {'param': self.beta, 'grad': self.grad_beta}
        ]


class Softmax:
    """
    Softmax layer for output normalization
    """
    def __init__(self):
        self.output = None
        self.have_params = False
        self.train = True

    def __call__(self, logits):
        return self.forward(logits)

    def forward(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.output * (1 - self.output)


class CrossEntropyLoss():
    """
    Cross entropy loss for multi-class classification without softmax
    """
    def __init__(self):
        self.preds = None
        self.labels = None

    def __call__(self, preds, labels):
        return self.forward(preds, labels)

    def forward(self, preds: np.ndarray, labels: int) -> float:
        self.preds = preds
        self.labels = labels

        # change single integer label to onehot encoding
        labels_onehot = np.zeros_like(preds)
        labels_onehot[np.arange(len(labels)), labels] = 1
        self.labels = labels_onehot
        eps = 1e-9 # include epsilon to avoid log(0)

        loss = -np.mean(np.sum(labels_onehot * np.log(np.clip(preds, eps, 1-eps)), axis = -1))
        return loss

    def backward(self) -> np.ndarray:
        return self.preds - self.labels 


class SGD:
    """
    Stochastic Gradient Descent optimizer
    """
    def __init__(self, params: list, lr: float=0.01, momentum: float=0.9, weight_decay: float=0.01):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

        self.momentum = momentum
        self.velocities = [np.zeros_like(p['param']) for p in self.params] 

    # store reference to model parameters
    def set_params(self, params: list):
        self.params = params

    def step(self):
        for i, p in enumerate(self.params):
            grad = p['grad']
            if self.weight_decay != 0: # apply weight decay (L2 regularization)
                grad += self.weight_decay * p['param'] 

            # apply momentum and update velocities
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            p['param'] += self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            p['grad'].fill(0.0) 
    

class Adam:
    """
    Adam optimizer
    """
    def __init__(self, params: list, lr: float=0.001, beta1: float=0.9, beta2: float=0.99, eps: float=1e-10, weight_decay: float=0.01):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        if not params:
            return
        self.m = [np.zeros_like(p['param']) for p in params]
        self.v = [np.zeros_like(p['param']) for p in params]

    # store reference to model parameters
    def set_params(self, params: list):
        self.params = params
        self.m = [np.zeros_like(p['param']) for p in params]
        self.v = [np.zeros_like(p['param']) for p in params]

    def step(self):
        self.t += 1

        for i, p in enumerate(self.params):
            grad = p['grad']

            if self.weight_decay != 0: # apply weight decay (L2 regularization)
                grad += self.weight_decay * p['param']

            # update parameters of model according to original paper
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.power(grad, 2)

            m_hat = self.m[i] / (1 - np.power(self.beta1, self.t))
            v_hat = self.v[i] / (1 - np.power(self.beta2, self.t))

            p['param'] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p['grad'].fill(0.0)


class Model:
    """
    Model class for better modularity, similar to Pytorch Sequential class
    """
    def __init__(self, model):
        self.model = model

    def forward(self, input: np.ndarray) -> np.ndarray:
        # transform input layer by layer
        for layer in self.model:
            input = layer(input)
        return input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # backpropagate gradient layer by layer
        for layer in self.model[::-1]:
            grad = layer.backward(grad)

    # return trainable parameters from layers in the model that have it
    def get_params(self) -> list:
        params = []
        for layer in self.model:
            if layer.have_params:
                params += layer.params()
        return params

    # set layers to training mode
    def set_train(self):
        for layer in self.model:
                layer.train = True
    
    # set layers to inference mode
    def set_test(self):
        for layer in self.model:
                layer.train = False