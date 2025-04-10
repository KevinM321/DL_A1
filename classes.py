import numpy as np
import scipy
import os
import random
from matplotlib import pyplot as plt


""" Custom DataSet and DataLoader classes"""
class DataSet:
    def __init__(self, data_file : str, label_file : str):
        self.data = np.load(data_file)
        self.labels = np.load(label_file)
    
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
        if batch_end_idx >= len(self.dataset):
            batch_end_idx = len(self.dataset) - 1

        batch_indices = self.indices[self.current_idx : batch_end_idx]
        batch = [self.dataset[i] for i in batch_indices]
        self.current_idx = batch_end_idx

        return batch


""" Activation Function Classes"""
class RELU:
    def __init__(self):
        self.input = None
        self.have_params = False

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad):
        return grad * (self.input > 0)
    

class GELU:
    def __init__(self):
        self.input = None
        self.have_params = False

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        self.input = x
        return x * 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))

    def backward(self, grad_x):
        x = self.input
        phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.pow(x, 2))
        Phi = 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))
        return grad_x * (x * phi + Phi)


class Linear:
    def __init__(self, in_features : int, out_features : int):
        self.input = None
        self.have_params = True

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
        # grad_input = np.dot(grad_output, self.W)
        # self.grad_W = np.dot(grad_output.T, self.input)
        # self.grad_b = np.sum(grad_output, axis=0)
        input = np.expand_dims(self.input, axis=0)
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


class Softmax:
    def __init__(self):
        self.output = None
        self.have_params = False

    def __call__(self, logits):
        return self.forward(logits)

    def forward(self, logits):
        output = np.exp(logits - np.max(logits))
        self.output = output / output.sum(axis=0)
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
        print(preds, labels)
        self.preds = preds
        self.labels = labels
        labels_onehot = np.zeros_like(preds)
        labels_onehot[np.arange(len(labels)), labels] = 1
        self.labels = labels_onehot
        print(labels_onehot)

        # loss = -1/len(labels) * np.sum(np.sum(labels * np.log(preds)))
        loss = -np.mean(np.sum(labels_onehot * np.log(preds), axis = -1))
        return loss

    def backward(self):
        return self.preds - self.labels 


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
        print(input.shape)
        for layer in self.model:
            input = layer(input)
            print(type(layer).__name__, input)
        return input

    def backward(self, grad):
        print(grad.shape, grad)
        for layer in self.model[::-1]:
            grad = layer.backward(grad)
            print(type(layer).__name__, grad)

    def get_params(self):
        params = []
        for layer in self.model:
            if layer.have_params:
                params += layer.params()
        return params