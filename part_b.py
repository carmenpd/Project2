import numpy as np
import matplotlib.pyplot as plt
from FFNN import *
from activation_functions import *
from cost_functions import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import seaborn as sns

def create_eta_lambda_heatmap(X_train, t_train, eta_vals, lmbd_vals, n_epochs, batches):
    train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
    train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals): 
            linear_regression.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

            scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
            scores = linear_regression.fit(X_train, t_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)

            pred_train = linear_regression.predict(X_train)
            train_mse[i, j] = mean_squared_error(pred_train, t_train)
            train_r2[i, j] = r2_score(t_train, pred_train)
            
            pred_test = linear_regression.predict(X_test)
            test_mse[i, j] = mean_squared_error(pred_test, t_test)
            test_r2[i, j] = r2_score(t_test , pred_test)

    # Plot MSE
    sns.set()
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(test_mse, annot=True, ax=ax, cmap="plasma")
    ax.set_title("MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    ax.set_yticklabels(eta_vals)
    ax.set_xticklabels(lmbd_vals)
    plt.show()

def plot_activation_func_comparison(X_train, X_test, target, n_epochs, batches, eta, lmbd):
    activation_func_dict = {
        'Sigmoid': {
            'func': sigmoid,
            'marker': '^',
            'color': 'green'
        },
        'RELU': {
            'func': RELU,
            'marker': 'v',
            'color': 'black'
        },
        'Leaky RELU': {
            'func': LRELU,
            'marker': '*',
            'color': 'orange'
        },
        'Identity': {
            'func': identity,
            'marker': '+',
            'color': 'gray'
        },
    }
    for key in activation_func_dict.keys():
        # activation_func = activation_func_dict[key]['func']
        func = activation_func_dict[key]['func']
        linear_regression = FFNN(layers, hidden_func=func, cost_func=cost_func, output_func=identity)    
        linear_regression.reset_weights()
        scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
        scores = linear_regression.fit(X_train, target, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)
        pred = linear_regression.predict(X_test)
        plt.scatter(X_test[:,0], pred, c=activation_func_dict[key]['color'], marker=activation_func_dict[key]['marker'], label=key, zorder=10)

    # Plot
    sns.set()
    plt.plot(X[:,0], y, 'ro', label='Data')
    plt.plot(X[:,0], y_true, 'b-', label='True')
    plt.legend()
    plt.show()

def compare_model_and_sklearn(X_train, X_test, target_train, target_test, true_train, true_test, layers, lmbd, eta, batches, n_epochs):
    M = int(X_train.shape[0] // batches)

    # Out model
    our_model = FFNN(layers, hidden_func=sigmoid, cost_func=cost_func, output_func=identity)
    our_model.reset_weights()
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
    scores = our_model.fit(X_train, target_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)

    # SciKit Learn model
    clf = MLPRegressor(hidden_layer_sizes=layers[1:-1], activation='logistic', solver='adam', 
                       alpha=lmbd, batch_size=M, learning_rate_init=eta, max_iter=400, 
                       shuffle=False, tol=0.001, verbose=False,
                       beta_1=0.9, beta_2=0.999, epsilon=10e-8)

    clf.fit(X_train, target_train.ravel())
    
    # Predict
    pred = our_model.predict(X_test)
    pred_sklearn = clf.predict(X_test)

    # Score
    score = mean_squared_error(pred, true_test)
    score_sklearn = mean_squared_error(pred_sklearn, true_test)
    print(f"\nOur model score: {score}")
    print(f"\nSciKit Learn score: {score_sklearn}")



    # Plot
    sns.set()
    plt.plot(X[:,0], y, 'ro', label='Data')
    plt.plot(X[:,0], y_true, 'b-', label='True')
    plt.scatter(X_test[:,0], pred, c='g', marker='v', label='Our model', zorder=10)
    plt.scatter(X_test[:,0], pred_sklearn, c='k', marker='^', label='SciKit Learn model', zorder=10)
    plt.legend()
    plt.show()


n = 100
x = np.linspace(-3, 3, n)
noise = np.random.normal(0, 1.0, n)
y_true = 0.2*x**4 - 1*x**3 - 0.25*x**2#  + 2*np.sin(x*np.pi) + 4*np.sin(x*np.pi*0.7 + 0.3)
# y_true = np.exp(-x**2)*np.sin(x*np.pi)
y = y_true + noise
X = np.column_stack((x.reshape(-1, 1), y.reshape(-1, 1)))

# Split and scale
X_train, X_test, t_train, t_test, true_train, true_test = train_test_split(X, y.reshape(-1, 1), y_true.reshape(-1, 1), test_size=0.2)

# Model
input_nodes = X_train.shape[1]
hidden_nodes_1 = 10
hidden_nodes_2 = 5
output_nodes = t_train.shape[1]
layers = (input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes)
n_epochs = 400
batches = 20
cost_func = CostOLS
activation_func = sigmoid
linear_regression = FFNN(layers, hidden_func=activation_func, cost_func=cost_func, output_func=identity)

# Train and create heatmap
eta_vals = np.logspace(-5, -2, 4)
lmbd_vals = np.logspace(-5, -1, 5)
# create_eta_lambda_heatmap(X_train, t_train, eta_vals, lmbd_vals, n_epochs, batches)

# Train with different activation functions
# plot_activation_func_comparison(X_train, X_test, t_train, n_epochs, batches, eta=0.01, lmbd=0.01)

# Compare our model with SciKit Learn
compare_model_and_sklearn(X_train, X_test, t_train, t_test, true_train, true_test, layers, lmbd=0.01, eta=0.01, batches=batches, n_epochs=n_epochs)