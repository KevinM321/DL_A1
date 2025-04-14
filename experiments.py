import matplotlib.pyplot as plt
from experiment_functions import *

def optimal_model(train_loader, test_dataset, seed=100):
    model = Model([
                Linear(128, 192),
                BatchNorm(192),
                GELU2(),
                Dropout(0.15),

                Linear(192, 64),
                BatchNorm(192),
                GELU2(),
                Dropout(0.1),

                Linear(64, 10),
                Softmax()
                ])
    lr = 0.0001
    loss_fn = CrossEntropyLoss()
    optim = Adam(model.get_params(), lr, weight_decay=0.025)

    results = ['optimal_model', model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)]
    plot_helper()


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
            BatchNorm(192),
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

    results = []
    for model in models:
        results += model_train_test_run(seed, train_loader, test_dataset, model, loss_fn, optim)

    plt.figure(figsize=(10, 6))

    i = 1
    for run in results:
        plt.plot(run['test_l'], label='Test Loss' + str(i))
        i += 1

    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss of Ablated Models")
    plt.legend()
    # plt.grid(True)
    plt.show()


def main():
    seed = 37
    test_dataset, train_loader = prepare_data(seed)

    optimal_model(train_loader, test_dataset, seed)
    ablation_experiments(train_loader, test_dataset, seed)

main()