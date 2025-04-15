import matplotlib.pyplot as plt
from experiment_functions import *


# train optimal model and store figure and results
def optimal_model(train_loader, test_dataset, seed=100):
    model = Model([
                Linear(128, 192),
                BatchNorm(192),
                GELU2(),
                Dropout(0.15),

                Linear(192, 64),
                BatchNorm(64),
                GELU2(),
                Dropout(0.1),

                Linear(64, 10),
                Softmax()
                ])
    lr = 0.0001
    loss_fn = CrossEntropyLoss()
    optim = Adam(model.get_params(), lr, weight_decay=0.025)

    final_data, results = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)
    plot_helper([("Optimal Model", results)], 'Loss', 'Loss', '', 'optimal_model_loss', True, True, False)
    plot_helper([("Optimal Model", results)], 'Accuracy', 'Accuracy (%)', '', 'optimal_model_acc', True, True, False)
    result_store_helper([("Optimal Model", final_data)], 'optimal_model')


# tests models of different depth with same number of hidden dimensions
def depth_experiments(train_loader, test_dataset, seed=100):
    lr = 0.0001
    loss_fn = CrossEntropyLoss()
    optim = Adam(None, lr, weight_decay = 0.025)
    models = [
        Model([ # one layer
            Linear(128, 192),
            GELU2(),
            
            Linear(192, 10),
            Softmax()
        ]), 

        Model([ # two layers
            Linear(128, 192),
            GELU2(),
            
            Linear(192, 192),
            GELU2(),
            
            Linear(192, 10),
            Softmax()]),

        Model([ # three layers
            Linear(128, 192),
            GELU2(),

            Linear(192, 192),
            GELU2(),

            Linear(192, 192),
            GELU2(),

            Linear(192, 10),
            Softmax()
            ])
    ]

    names = ['One Hidden Layer', 'Two Hidden Layers', 'Three Hidden Layers']
    results = []
    final_datas = []

    i = 0
    for model in models:
        final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)
        results.append((names[i], result))
        final_datas.append((names[i], final_data))
        i += 1
    plot_helper(results, 'Loss', "Loss", '', 'depth_experiments_loss', False, True, True)
    plot_helper(results, "Accuracy", "Accuracy (%)", '', 'depth_experiments_acc', False, True, True)
    result_store_helper(final_datas, 'depth_experiments')


# tests one layer model with different number of neurons
def width_experiments(train_loader, test_dataset, seed=100):
    lr = 0.0001
    loss_fn = CrossEntropyLoss()
    optim = Adam(None, lr, weight_decay = 0.025)
    models = [
        Model([ # 32 neurons
            Linear(128, 32),
            GELU2(),
            
            Linear(32, 10),
            Softmax()
        ]), 

        Model([ # 64 neurons
            Linear(128, 64),
            GELU2(),

            Linear(64, 10),
            Softmax()
        ]),

        Model([ # 128 neurons
            Linear(128, 128),
            GELU2(),

            Linear(128, 10),
            Softmax()
        ]),

        Model([ # 256 neurons
            Linear(128, 256),
            GELU2(),

            Linear(256, 10),
            Softmax()
        ])
    ]

    names = ['32', '64', '128', '256']
    results = []
    final_datas = []

    i = 0
    for model in models:
        final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)
        results.append((names[i], result))
        final_datas.append((names[i], final_data))
        i += 1
    plot_helper(results, 'Loss', "Loss", '', 'width_experiments_loss', False, True, True)
    plot_helper(results, "Accuracy", "Accuracy (%)", '', 'width_experiments_acc', False, True, True)
    result_store_helper(final_datas, 'width_experiments')


# tests effect of different dropout probability
def dropout_experiments(train_loader, test_dataset, seed=100):
    lr = 0.0001
    loss_fn = CrossEntropyLoss()
    optim = Adam(None, lr, weight_decay = 0.025)
    models = [
        Model([ # 0.1 dropout 
            Linear(128, 192),
            GELU2(),
            Dropout(0.1),

            Linear(192, 10),
            Softmax()
        ]),

        Model([ # 0.3 dropout 
            Linear(128, 192),
            GELU2(),
            Dropout(0.3),

            Linear(192, 10),
            Softmax()
        ]),

        Model([ # 0.5 dropout 
            Linear(128, 192),
            GELU2(),
            Dropout(0.5),

            Linear(192, 10),
            Softmax()
        ]),

        Model([ # 0.7 dropout 
            Linear(128, 192),
            GELU2(),
            Dropout(0.7),

            Linear(192, 10),
            Softmax()
        ])
    ]

    names = ['0.1', '0.3', '0.5', '0.7']
    results = []
    final_datas = []

    i = 0
    for model in models:
        final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)
        results.append((names[i], result))
        final_datas.append((names[i], final_data))
        i += 1
    plot_helper(results, 'Loss', "Loss", '', 'dropout_experiments_loss', False, True, True)
    plot_helper(results, "Accuracy", "Accuracy (%)", '', 'dropout_experiments_acc', False, True, True)
    result_store_helper(final_datas, 'dropout_experiments')


# tests effect of different activation function
def activation_experiments(train_loader, test_dataset, seed=100):
    lr = 0.0001
    loss_fn = CrossEntropyLoss()
    optim = Adam(None, lr, weight_decay = 0.025)
    models = [
        Model([ # ReLU activation
            Linear(128, 192),
            RELU(),

            Linear(192, 10),
            Softmax()
        ]),

        Model([ # exact GELU activation
            Linear(128, 192),
            GELU(),

            Linear(192, 10),
            Softmax()
        ]),

        Model([ # approximated GELU activation
            Linear(128, 192),
            GELU2(),

            Linear(192, 10),
            Softmax()
        ])
    ]

    names = ['ReLU', 'Exact GELU', 'Approximated GELU']
    results = []
    final_datas = []

    i = 0
    for model in models:
        final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)
        results.append((names[i], result))
        final_datas.append((names[i], final_data))
        i += 1
    plot_helper(results, 'Loss', "Loss", '', 'activation_experiments_loss', False, True, True)
    plot_helper(results, "Accuracy", "Accuracy (%)", '', 'activation_experiments_acc', False, True, True)
    result_store_helper(final_datas, 'activation_experiments')


# tests effect of using different optimizers
def optim_experiments(train_loader, test_dataset, seed=100):
    # same hyperparameters to be used for both SGD and Adam
    lr = 0.0001
    weight_decay = 0.02
    loss_fn = CrossEntropyLoss()
    
    results = []
    final_datas = []

    set_seed(seed)
    model = Model([
        Linear(128, 192),
        GELU2(),

        Linear(192, 10),
        Softmax(),
    ])


    # test run using SGD optimizer without momentum
    final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, SGD(None, lr, momentum=0.0, weight_decay=weight_decay))
    results.append(('SGD w/ Momentum', result))
    final_datas.append(('SGD w/ Momentum', final_data))

    set_seed(seed)
    model = Model([
        Linear(128, 192),
        GELU2(),

        Linear(192, 10),
        Softmax(),
    ])

    # test run using SGD optimizer with momentum
    final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, SGD(None, lr, momentum=0.9, weight_decay=weight_decay))
    results.append(('SGD w Momentum', result))
    final_datas.append(('SGD w Momentum', final_data))

    set_seed(seed)
    model = Model([
        Linear(128, 192),
        GELU2(),

        Linear(192, 10),
        Softmax(),
    ])

    # test run using Adam optimizer
    final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, Adam(None, lr=lr, weight_decay=weight_decay))
    results.append(('Adam', result))
    final_datas.append(('Adam', final_data))

    plot_helper(results, 'Loss', "Loss", '', 'optim_experiments_loss', False, True, True)
    plot_helper(results, "Accuracy", "Accuracy (%)", '', 'optim_experiments_acc', False, True, True)
    result_store_helper(final_datas, 'optim_experiments')


# test batch size effects on training of a simple model
def batch_size_experiments(train_dataset, test_dataset, seed=100):
    b32_loader = DataLoader(train_dataset, 32)
    b64_loader = DataLoader(train_dataset, 64)
    b128_loader = DataLoader(train_dataset, 128)

    lr = 0.0001
    loss_fn = CrossEntropyLoss()
    optim = Adam(None, lr, weight_decay = 0.025)
    
    loaders = [b32_loader, b64_loader, b128_loader]
    names = ["32", "64", "128"]
    results = []
    final_datas = []

    i = 0
    for loader in loaders:
        set_seed(seed) 
        # simple model with batchnorm
        model = Model([
                Linear(128, 192),
                BatchNorm(192),
                GELU2(),

                Linear(192, 10),
                Softmax()
                ])
        final_data, result = model_train_test_run(seed, loader, test_dataset, model, loss_fn, optim)
        results.append((names[i], result))
        final_datas.append((names[i], final_data))
        i += 1
       
    plot_helper(results, 'Loss', "Loss", '', 'batch_size_experiments_loss', False, True, True)
    plot_helper(results, "Accuracy", "Accuracy (%)", '', 'batch_size_experiments_acc', False, True, True)
    result_store_helper(final_datas, 'batch_size_experiments')


# test weight decay (L2 regularization) effect on training
def L2_experiments(train_loader, test_dataset, seed):
    lr = 0.0001
    loss_fn = CrossEntropyLoss()
    optim = Adam(None, lr)
    

    names = ["0.001", "0.01", "0.1"]
    results = []
    final_datas = []

    for name in names:
        set_seed(seed) 
        # simple model with batchnorm
        model = Model([
                Linear(128, 192),
                BatchNorm(192),
                GELU2(),

                Linear(192, 10),
                Softmax()
                ])

        # set weight decay of optimizer with new test value
        optim.weight_decay = float(name)
        final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)
        results.append((name, result))
        final_datas.append((name, final_data))
    plot_helper(results, 'Loss', "Loss", '', 'L2_experiments_loss', False, True, True)
    plot_helper(results, "Accuracy", "Accuracy (%)", '', 'L2_experiments_acc', False, True, True)
    result_store_helper(final_datas, 'L2_experiments')


# tests effect of different learning rate
def lr_experiments(train_loader, test_dataset, seed):
    loss_fn = CrossEntropyLoss()
    optim = Adam(None, weight_decay=0.025)
    

    names = ["0.0001", "0.001", "0.01"]
    results = []
    final_datas = []

    for name in names:
        set_seed(seed) 
        # simple model with batchnorm
        model = Model([
                Linear(128, 192),
                BatchNorm(192),
                GELU2(),

                Linear(192, 10),
                Softmax()
                ])
        
        # set learning rate of optimizer with new test value
        optim.lr = float(name)
        final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)
        results.append((name, result))
        final_datas.append((name, final_data))
    plot_helper(results, 'Loss', "Loss", '', 'lr_experiments_loss', False, True, True)
    plot_helper(results, "Accuracy", "Accuracy (%)", '', 'lr_experiments_acc', False, True, True)
    result_store_helper(final_datas, 'lr_experiments')


# run ablated models using same training hyperparameters as the optimal model
def ablation_experiments(train_loader, test_dataset, seed=100):
    lr = 0.0001
    loss_fn = CrossEntropyLoss()
    optim = Adam(None, lr, weight_decay=0.025)
    models = [
        Model([ # optimal model
            Linear(128, 192),
            BatchNorm(192),
            GELU2(),
            Dropout(0.15),

            Linear(192, 64),
            BatchNorm(64),
            GELU2(),
            Dropout(0.1),

            Linear(64, 10),
            Softmax()
        ]), 

        Model([ # no activation function
            Linear(128, 192),
            BatchNorm(192),
            Dropout(0.15),

            Linear(192, 64),
            BatchNorm(64),
            Dropout(0.1),

            Linear(64, 10),
            Softmax()
        ]),

        Model([ # no Batch normalisation
            Linear(128, 192),
            GELU2(),
            Dropout(0.15),

            Linear(192, 64),
            GELU2(),
            Dropout(0.1),

            Linear(64, 10),
            Softmax()
        ]),

        Model([ # no Dropout
            Linear(128, 192),
            BatchNorm(192),
            GELU2(),

            Linear(192, 64),
            BatchNorm(64),
            GELU2(),

            Linear(64, 10),
            Softmax()
        ]),

        Model([ # linear and activation only
            Linear(128, 192),
            GELU2(),
            
            Linear(192, 64),
            GELU2(),

            Linear(64, 10),
            Softmax()
        ])
    ]

    names = ['Optimal Model', 'No Activation', 'No Batch Normalisation', 'No Dropout', 'Linear and Activation Only']
    results = []
    final_datas = []

    i = 0
    for model in models:
        final_data, result = model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)
        results.append((names[i], result))
        final_datas.append((names[i], final_data))
        i += 1

    plot_helper(results, 'Loss', "Loss", '', 'ablation_experiments_loss', False, True, True)
    plot_helper(results, "Accuracy", "Accuracy (%)", '', 'ablation_experiments_acc', False, True, True)
    result_store_helper(final_datas, 'ablation_experiments')

if __name__ == '__main__':
    seed = 23
    train_dataset, test_dataset = prepare_data(seed)
    train_loader = DataLoader(train_dataset, batch_size=64)

    # architecture studies
    depth_experiments(train_loader, test_dataset, seed)
    width_experiments(train_loader, test_dataset, seed)
    dropout_experiments(train_loader, test_dataset, seed)
    activation_experiments(train_loader, test_dataset, seed)
    optim_experiments(train_loader, test_dataset, seed)

    # hyperparameter studies
    batch_size_experiments(train_dataset, test_dataset, seed)
    L2_experiments(train_loader, test_dataset, seed)
    lr_experiments(train_loader, test_dataset, seed)

    # best model and ablation studies
    optimal_model(train_loader, test_dataset, seed)
    ablation_experiments(train_loader, test_dataset, seed)
    