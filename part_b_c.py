import numpy as np
import matplotlib.pyplot as plt
from FFNN import *
from activation_functions import *
from cost_functions import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import pandas as pd

def output_func_compare(X_train, X_test, target_train, eta, lmbd, n_epochs, batches):
    # Compare output functions

    # Sigmoid
    sigmoid_model = FFNN(layers, hidden_func=sigmoid, cost_func=cost_func, output_func=sigmoid)
    sigmoid_model.reset_weights()
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
    scores = sigmoid_model.fit(X_train, target_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)
    pred_sigmoid = sigmoid_model.predict(X_test)

    # Identity
    identity_model = FFNN(layers, hidden_func=sigmoid, cost_func=cost_func, output_func=identity)
    identity_model.reset_weights()
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
    scores = identity_model.fit(X_train, target_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)
    pred_identity = identity_model.predict(X_test)

    # Metrics
    mse_sigmoid = mean_squared_error(true_test, pred_sigmoid)
    mse_identity = mean_squared_error(true_test, pred_identity)
    r2_sigmoid = r2_score(true_test, pred_sigmoid)
    r2_identity = r2_score(true_test, pred_identity)
    print(f"\nSigmoid score: \tMSE\t{mse_sigmoid:.4f}\t R^2\t{r2_sigmoid:.4f}")
    print(f"Identity score: \tMSE\t{mse_identity:.4f}\t R^2\t{r2_identity:.4f}")

    # Plot
    sns.set()
    plt.plot(X[:,0], y, 'ro', label='Data')
    plt.plot(X[:,0], y_true, 'b-', label='True')
    plt.scatter(X_test[:,0], pred_sigmoid, c='g', marker='v', label='Sigmoid', zorder=10)
    plt.scatter(X_test[:,0], pred_identity, c='k', marker='^', label='Identity', zorder=10)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

def create_eta_lambda_heatmap(X_train, X_test, t_train, eta_vals, lmbd_vals, n_epochs, batches):
    train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
    train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))

    linear_regression = FFNN(layers, hidden_func=sigmoid, cost_func=cost_func, output_func=identity)

    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals): 
            linear_regression.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

            scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
            scores = linear_regression.fit(X_train, t_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)

            pred_train = linear_regression.predict(X_train)
            train_mse[i, j] = mean_squared_error(true_train, pred_train)
            train_r2[i, j] = r2_score(true_train, pred_train)
            
            pred_test = linear_regression.predict(X_test)
            test_mse[i, j] = mean_squared_error(true_test, pred_test)
            test_r2[i, j] = r2_score(true_test , pred_test)

    # Plot MSE training
    sns.set()
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(train_mse, annot=True, ax=ax, cmap="plasma", fmt=".3f")
    ax.set_title("Training MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    ax.set_yticklabels(eta_vals)
    ax.set_xticklabels(lmbd_vals)
    plt.show()

    # Plot MSE test
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(test_mse, annot=True, ax=ax, cmap="plasma", fmt=".3f")
    ax.set_title("Test MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    ax.set_yticklabels(eta_vals)
    ax.set_xticklabels(lmbd_vals)
    plt.show()

    # Plot R2 training
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(train_r2, annot=True, ax=ax, cmap="plasma", fmt=".3f")
    ax.set_title("Training R2")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    ax.set_yticklabels(eta_vals)
    ax.set_xticklabels(lmbd_vals)
    plt.show()

    # Plot R2 test
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(test_r2, annot=True, ax=ax, cmap="plasma", fmt=".3f")
    ax.set_title("Test R2")
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
        'Hyperbolic tangent': {
            'func': tanh,
            'marker': '+',
            'color': 'gray'
        },
    }
    sns.set()
    for key in activation_func_dict.keys():
        # activation_func = activation_func_dict[key]['func']
        func = activation_func_dict[key]['func']
        linear_regression = FFNN(layers, hidden_func=func, cost_func=cost_func, output_func=identity)    
        linear_regression.reset_weights()
        scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
        scores = linear_regression.fit(X_train, target, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)
        pred = linear_regression.predict(X_test)
        plt.scatter(X_test[:,0], pred, c=activation_func_dict[key]['color'], marker=activation_func_dict[key]['marker'], label=key, zorder=10)
        mse = mean_squared_error(true_test, pred)
        r2 = r2_score(true_test, pred)
        print(f"\nActivation function: {key}\n\tMSE = \t{mse:.4f}\tR^2 = \t{r2:.4f}")

    # Plot
    plt.plot(X[:,0], y, 'ro', label='Data')
    plt.plot(X[:,0], y_true, 'b-', label='True')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

def compare_model_and_sklearn(X_train, X_test, target_train, true_test, layers, lmbd, eta, batches, n_epochs):
    batch_size = int(X_train.shape[0] // batches)

    # Out model
    our_model = FFNN(layers, hidden_func=sigmoid, cost_func=cost_func, output_func=identity)
    our_model.reset_weights()
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
    scores = our_model.fit(X_train, target_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)

    # SciKit Learn model
    clf = MLPRegressor(hidden_layer_sizes=layers[1:-1], activation='logistic', solver='adam', 
                       alpha=lmbd, batch_size=batch_size, learning_rate_init=10*eta, max_iter=n_epochs, 
                       shuffle=False, tol=0.001, verbose=False,
                       beta_1=0.9, beta_2=0.999, epsilon=10e-8)
    clf.fit(X_train, target_train.ravel())

    # Predict
    pred = our_model.predict(X_test)
    pred_sklearn = clf.predict(X_test)

    # Score
    score = mean_squared_error(true_test, pred)
    score_sklearn = mean_squared_error(true_test, pred_sklearn)
    r2_our = r2_score(true_test, pred)
    r2_sklearn = r2_score(true_test, pred_sklearn)

    print(f"\nOur model score: \tMSE\t{score:.4f}\t R^2\t{r2_our:.4f}")
    print(f"SciKit Learn score: \tMSE\t{score_sklearn:.4f}\t R^2\t{r2_sklearn:.4f}")

    # Plot
    sns.set()
    plt.plot(X[:,0], y, 'ro', label='Data')
    plt.plot(X[:,0], y_true, 'b-', label='True')
    plt.scatter(X_test[:,0], pred, c='g', marker='v', label='Our model', zorder=10)
    plt.scatter(X_test[:,0], pred_sklearn, c='k', marker='^', label='SciKit Learn model', zorder=10)
    plt.legend()
    # plt.title(r'$f(x) = \frac{1}{5}x^4 - \frac{4}{5}x^3 - \frac{1}{4}x^2$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

def compare_weight_inits(X_train, X_test, target_train, true_test, eta, lmbd, batches, n_epochs, hidden_activation=sigmoid, plot_title = None):
    # Compare weight initializations using normal distribution and Xavier Glorot initialization
    
    # Normal distribution
    ordinary_weight_model = FFNN(layers, hidden_func=hidden_activation, cost_func=cost_func, output_func=identity)
    ordinary_weight_model.reset_weights()
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
    scores = ordinary_weight_model.fit(X_train, target_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)
    pred = ordinary_weight_model.predict(X_test)

    # Xavier Glorot initialization
    XG_weight_model = FFNN(layers, hidden_func=hidden_activation, cost_func=cost_func, output_func=identity, weight_scheme='xavier')
    XG_weight_model.reset_weights()
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
    scores = XG_weight_model.fit(X_train, target_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)
    pred_XG = XG_weight_model.predict(X_test)

    # He initialization
    He_weight_model = FFNN(layers, hidden_func=hidden_activation, cost_func=cost_func, output_func=identity, weight_scheme='he')
    He_weight_model.reset_weights()
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
    scores = He_weight_model.fit(X_train, target_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)
    pred_He = He_weight_model.predict(X_test)

    # Scores
    score = mean_squared_error(true_test, pred)
    r2 = r2_score(true_test, pred)
    score_XG = mean_squared_error(true_test, pred_XG)
    r2_XG = r2_score(true_test, pred_XG)
    score_he = mean_squared_error(true_test, pred_He)
    r2_he = r2_score(true_test, pred_He)

    print(f"\nNormal weight initialization score: \tMSE\t{score:.4f}\t R^2\t{r2:.4f}")
    print(f"Xavier Glorot initialization score: \tMSE\t{score_XG:.4f}\t R^2\t{r2_XG:.4f}")
    print(f"He initialization score: \t\tMSE\t{score_he:.4f}\t R^2\t{r2_he:.4f}")

    # Plot
    sns.set()
    plt.plot(X[:,0], y, 'ro', label='Data')
    plt.plot(X[:,0], y_true, 'b-', label='True')
    plt.scatter(X_test[:,0], pred, c='g', marker='v', label='Normal weight initialization', zorder=10)
    plt.scatter(X_test[:,0], pred_XG, c='k', marker='^', label='Xavier Glorot initialization', zorder=10)
    plt.scatter(X_test[:,0], pred_He, c='orange', marker='*', label='He initialization', zorder=10)
    plt.legend()
    plt.title(f"{plot_title}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()
    pass

def epoch_minibatch_gridsearch(X_train, X_test, t_train, layers, eta, lam):
    #### Grid search for epochs and batches ####
    n = 10
    epoch_space = np.linspace(100, 1000, n)
    batch_space = np.linspace(5, 50, n)
    train_mse = np.zeros((n, n))
    test_mse = np.zeros((n, n))
    train_r2 = np.zeros((n, n))
    test_r2 = np.zeros((n, n))
    linear_regression = FFNN(layers, hidden_func=sigmoid, cost_func=cost_func, output_func=identity)
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)

    # This is quite slow, btw
    for i, epoch in enumerate(epoch_space):
        for j, batch in enumerate(batch_space):
            linear_regression.reset_weights()
            score = linear_regression.fit(X_train, t_train, scheduler, epochs = int(epoch), batches=int(batch), lam=lam)
            pred_train = linear_regression.predict(X_train)
            train_mse[i, j] = mean_squared_error(true_train, pred_train)
            train_r2[i, j] = r2_score(true_train, pred_train)
            pred_test = linear_regression.predict(X_test)
            test_mse[i, j] = mean_squared_error(true_test, pred_test)
            test_r2[i, j] = r2_score(true_test, pred_test)

    train_mse = pd.DataFrame(train_mse, index = epoch_space, columns = batch_space)
    test_mse = pd.DataFrame(test_mse, index = epoch_space, columns = batch_space)
    train_r2 = pd.DataFrame(train_r2, index = epoch_space, columns = batch_space)
    test_r2 = pd.DataFrame(test_r2, index = epoch_space, columns = batch_space)

    fig, ax = plt.subplots(figsize = (8, 8))
    sns.heatmap(train_mse, annot = True, ax = ax, cmap = "magma" , fmt = ".3f")
    ax.set_title("Training MSE")
    ax.set_ylabel("Epochs")
    ax.set_xlabel("Batches")
    plt.show()

    fig, ax = plt.subplots(figsize = (8, 8))
    sns.heatmap(test_mse, annot = True, ax = ax, cmap = "magma" , fmt = ".3f" )
    ax.set_title("Test MSE")
    ax.set_ylabel("Epochs")
    ax.set_xlabel("Batches")
    plt.show()

    fig, ax = plt.subplots(figsize = (8, 8))
    sns.heatmap(train_r2, annot = True, ax = ax, cmap = "magma", fmt = ".3f")
    ax.set_title("Training R2")
    ax.set_ylabel("Epochs")
    ax.set_xlabel("Batches")
    plt.show()

    fig, ax = plt.subplots(figsize = (8, 8))
    sns.heatmap(test_r2, annot = True, ax = ax, cmap = "magma", fmt = ".3f")
    ax.set_title("Test R2")
    ax.set_ylabel("Epochs")
    ax.set_xlabel("Batches")
    plt.show()

    print(np.min(train_mse))
    print(np.min(test_mse))
    print(np.max(train_r2))
    print(np.max(test_r2))

    print("r squared", train_r2)

# Set variables and construct data
rng_seed = 2023
n = 100
x = np.linspace(-3, 3, n)
noise = np.random.normal(0, 1.0, n)
y_true = 0.2*x**4 - 0.8*x**3 - 0.25*x**2
y = y_true + noise
X = np.column_stack((x.reshape(-1, 1), y.reshape(-1, 1)))

# Split data (true_ are used for plotting and metrics)
X_train, X_test, target_train, target_test, true_train, true_test = train_test_split(X, y.reshape(-1, 1), y_true.reshape(-1, 1), test_size=0.2, random_state=rng_seed)

# Set variables for the FFNN model
input_nodes = X_train.shape[1]
hidden_nodes_1 = 10
hidden_nodes_2 = 5
output_nodes = target_train.shape[1]
layers = (input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes)
n_epochs = 400
batches = 25
cost_func = CostOLS
activation_func = sigmoid
eta = 0.001
lmbd = 0.001
eta_vals = np.logspace(-5, -2, 4)
lmbd_vals = np.logspace(-5, -1, 5)

#### Uncomment the function you want to run ####
# create_eta_lambda_heatmap(X_train, X_test, t_train, eta_vals, lmbd_vals, n_epochs, batches) # Create heatmap of MSE for different learning rates and lambdas
# epoch_minibatch_gridsearch(X_train, X_test, t_train, layers, eta, lmbd) # Grid search for epochs and batches
output_func_compare(X_train, X_test, target_train, eta=eta, lmbd=lmbd, n_epochs=n_epochs, batches=batches) # Compare output functions
# plot_activation_func_comparison(X_train, X_test, target_train, n_epochs, batches, eta=eta, lmbd=lmbd)# Train with different activation functions
# compare_model_and_sklearn(X_train, X_test, target_train, true_test, layers, lmbd=lmbd, eta=eta, batches=batches, n_epochs=n_epochs) # Compare our model with SciKit Learn

# hidden_func_dict = {'Sigmoid': sigmoid, 'ReLU': RELU, 'Leaky ReLU': LRELU, 'Hyperbolic tangent': tanh}
# for key in hidden_func_dict.keys():
#     compare_weight_inits(X_train, X_test, target_train, true_test, eta=eta, lmbd=lmbd, batches=batches, n_epochs=n_epochs, hidden_activation=hidden_func_dict[key], plot_title=key) # Compare weight initializations