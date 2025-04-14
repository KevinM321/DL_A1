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


# run ablated models using same training hyperparameters as the optimal model
# and store figures and result data
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
    test_dataset, train_loader = prepare_data(seed)

    # depth_experiments(train_loader, test_dataset, seed)
    # width_experiments(train_loader, test_dataset, seed)
    # dropout_experiments(train_loader, test_dataset, seed)
    # activation_experiments(train_loader, test_dataset, seed)

    loss_func_experiments(train_loader, test_dataset, seed)
    momentum_experiments(train_loader, test_dataset, seed)
    batch_size_experiments(train_loader, test_dataset, seed)
    L2_experiments(train_loader, test_dataset, seed)
    lr_experiments(train_loader, test_dataset, seed)
    

    # optimal_model(train_loader, test_dataset, seed)
    # ablation_experiments(train_loader, test_dataset, seed)
    