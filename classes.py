import numpy as np
import scipy
import os
import random
from matplotlib import pyplot as plt
import scipy.special


""" Custom DataSet and DataLoader classes"""
class DataSet:
    def __init__(self, data, labels):
        if isinstance(data, np.ndarray):
            self.data = data
            self.labels = labels
        else:
            self.data = np.load(data)
            self.labels = np.load(labels).squeeze()

    def __len__(self) -> int:
        return len(self.data)
    
    def _shape(self) -> tuple:
        return self.data.shape[0], self.data.shape[1], self.labels.shape[1]
        
    def __getitem__(self, idx) -> tuple:
        return self.data[idx], self.labels[idx]


class DataLoader:
    def __init__(self, dataset, batch_size : int = 32, shuffle : bool = True):
        self.dataset = dataset
        self.batch_size = batch_size if batch_size <= len(dataset) else len(dataset)
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0
        return self
    
    def __next__(self) -> list:
        if self.current_idx >= len(self.indices):
            raise StopIteration

        batch_end_idx = self.current_idx + self.batch_size
        batch_indices = self.indices[self.current_idx:batch_end_idx]
        self.current_idx = batch_end_idx

        batch = [self.dataset[i] for i in batch_indices]
        X_batch, y_batch = zip(*batch)
        return np.array(X_batch), np.array(y_batch)
       

""" Activation Function Classes"""
class RELU:
    def __init__(self):
        self.input = None
        self.have_params = False
        self.train = True

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.input = input
        self.mask = (input > 0).astype(np.float32)  # store as float
        return input * self.mask

    def backward(self, grad):
        return grad * self.mask
    

# exact
class GELU:
    def __init__(self):
        self.input = None
        self.have_params = False
        self.train = True

    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        self.input = input
        input = input * 0.5 * (1 + scipy.special.erf(input / np.sqrt(2)))
        return input        

    def backward(self, grad):
        phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.pow(self.input, 2))
        Phi = 0.5 * (1 + scipy.special.erf(self.input / np.sqrt(2)))
        return grad * (self.input * phi + Phi)

# approximated
class GELU2:
    def __init__(self):
        self.input = None
        self.have_params = False
        self.train = True

    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        self.input = input
        self.tanh = np.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * np.power(input, 3)))
        return 0.5 * input * (1 + self.tanh)     

    def backward(self, grad):
        tanh_deriv = 1 - np.power(self.tanh, 2)
        inner_deriv = 1 + 3 * 0.044715 * np.power(self.input, 2)

        part_1 = 0.5 * (1 + self.tanh)
        part_2 = 0.5 * self.input * tanh_deriv * np.sqrt(2 / np.pi) * inner_deriv
        
        return grad * (part_1 + part_2)


class Linear:
    def __init__(self, in_features : int, out_features : int):
        self.input = None
        self.have_params = True
        self.train = True

        self.W = np.random.uniform(
                low = -np.sqrt(6. / (in_features + out_features)),
                high = np.sqrt(6. / (in_features + out_features)),
                size = (in_features, out_features)
        )
        self.b = np.zeros(out_features, )

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.input = input
        output = np.dot(input, self.W) + self.b
        return output

    def backward(self, grad):
        input = self.input
        self.grad_W[...] = np.dot(input.T, grad)
        self.grad_b[...] = np.sum(grad, axis=0)
        return np.dot(grad, self.W.T)

    def params(self):
        return [
            {'param': self.W, 'grad': self.grad_W},
            {'param': self.b, 'grad': self.grad_b}
        ]

class Dropout:
    def __init__(self, chance : float = 0.5):
        self.have_params = False
        self.chance = chance
        self.mask = None
        self.train = True

    def __call__(self, input):
        return self.forward(input)

    def set_train(self, train : bool):
        self.train = train
           
    def forward(self, input):
        if self.train:
            self.mask = np.random.rand(*input.shape) > self.chance
            self.mask.astype(np.float32)
            return input * self.mask / (1 - self.chance)
        else:
            return input

    def backward(self, grad):
        return grad * self.mask
    

class BatchNorm:
    def __init__(self, in_features, momentum=0.9, eps=1e-5):
        self.in_features = in_features
        self.momentum = momentum
        self.eps = eps
        self.train = True
        self.have_params = True        

        self.gamma = np.ones((1, in_features))
        self.grad_gamma = np.zeros_like(self.gamma)

        self.beta = np.zeros((1, in_features))
        self.grad_beta = np.zeros_like(self.beta)

        self.running_mean = np.zeros((1, in_features))
        self.running_var = np.ones((1, in_features))

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.input = input
        
        if self.train:
            self.batch_mean = np.mean(input, axis=0, keepdims=True)
            self.batch_var = np.var(input, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var

            self.norm = (input - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
        else:
            self.norm = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.norm + self.beta
        
    def backward(self, grad):
        m = grad.shape[0]
        
        var_inverse = 1 / np.sqrt(self.batch_var + self.eps)
        input_norm = self.input - self.batch_mean

        norm_deriv = grad * self.gamma
        batch_var_deriv = np.sum(norm_deriv * input_norm * -0.5 * np.power(var_inverse, 3), axis=0, keepdims=True)

        mean_deriv_t1 = np.sum(norm_deriv * -var_inverse, axis=0, keepdims=True)
        mean_deriv_t2 =  batch_var_deriv * np.mean(-2 * input_norm, axis=0, keepdims=True)
        batch_mean_deriv = mean_deriv_t1 + mean_deriv_t2

        input_deriv = norm_deriv * var_inverse + batch_var_deriv * (2 * input_norm / m) + batch_mean_deriv / m

        self.grad_gamma = np.sum(grad * self.norm, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad, axis=0, keepdims=True)

        return input_deriv

    def params(self):
        return [
            {'param': self.gamma, 'grad': self.grad_gamma},
            {'param': self.beta, 'grad': self.grad_beta}
        ]


class Softmax:
    def __init__(self):
        self.output = None
        self.have_params = False
        self.train = True

    def __call__(self, logits):
        return self.forward(logits)

    def forward(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return self.output

    def backward(self, grad):
        return grad * self.output * (1 - self.output)


class CrossEntropyLoss():
    def __init__(self):
        self.preds = None
        self.labels = None

    def __call__(self, preds, labels):
        return self.forward(preds, labels)

    def forward(self, preds, labels):
        self.preds = preds
        self.labels = labels
        labels_onehot = np.zeros_like(preds)
        labels_onehot[np.arange(len(labels)), labels] = 1
        self.labels = labels_onehot
        eps = 1e-9

        loss = -np.mean(np.sum(labels_onehot * np.log(np.clip(preds, eps, 1-eps)), axis = -1))
        return loss

    def backward(self):
        return self.preds - self.labels 


class MeanSquareErrorLoss:
    def __init__(self):
        self.preds = None
        self.targets = None

    def __call__(self, preds, targets):
        return self.forward(preds, targets)

    def forward(self, preds, targets):
        self.preds = preds
        self.targets = targets
        loss = np.mean(np.power(preds - targets, 2))
        return loss

    def backward(self):
        return 2 * (self.preds - self.targets) / self.preds.shape[0]


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.1):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

        self.momentum = momentum
        self.velocities = [np.zeros_like(p['param']) for p in self.params]     

    def step(self):
        for i, p in enumerate(self.params):
            grad = p['grad']
            if self.weight_decay != 0:
                grad += self.weight_decay * p['param'] 

            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            p['param'] += self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            p['grad'].fill(0.0)    
    

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.99, eps=1e-10, weight_decay=0.1):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [np.zeros_like(p['param']) for p in params]
        self.v = [np.zeros_like(p['param']) for p in params]
        self.t = 0

    def step(self):
        self.t += 1

        for i, p in enumerate(self.params):
            grad = p['grad']

            if self.weight_decay != 0:
                grad += self.weight_decay * p['param']

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.power(grad, 2)

            m_hat = self.m[i] / (1 - np.power(self.beta1, self.t))
            v_hat = self.v[i] / (1 - np.power(self.beta2, self.t))

            p['param'] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p['grad'].fill(0.0)


class Model:
    def __init__(self, model):
        self.model = model

    def forward(self, input):
        for layer in self.model:
            input = layer(input)
        return input

    def backward(self, grad):
        for layer in self.model[::-1]:
            grad = layer.backward(grad)

    def get_params(self):
        params = []
        for layer in self.model:
            if layer.have_params:
                params += layer.params()
        return params

    def set_train(self):
        for layer in self.model:
            if not layer.train:
                layer.train = True
    
    def set_test(self):
        for layer in self.model:
            if layer.train:
                layer.train = False