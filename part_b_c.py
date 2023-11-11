import numpy as np
import matplotlib.pyplot as plt
from FFNN import *
from activation_functions import *
from cost_functions import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import seaborn as sns

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
    mse_sigmoid = mean_squared_error(pred_sigmoid, true_test)
    mse_identity = mean_squared_error(pred_identity, true_test)
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

def create_eta_lambda_heatmap(X_train, t_train, eta_vals, lmbd_vals, n_epochs, batches):
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
        mse = mean_squared_error(pred, true_test)
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
                       alpha=lmbd, batch_size=batch_size, learning_rate_init=eta, max_iter=400, 
                       shuffle=False, tol=0.001, verbose=False,
                       beta_1=0.9, beta_2=0.999, epsilon=10e-8)
    clf.fit(X_train, target_train.ravel())

    # Predict
    pred = our_model.predict(X_test)
    pred_sklearn = clf.predict(X_test)

    # Score
    score = mean_squared_error(pred, true_test)
    score_sklearn = mean_squared_error(pred_sklearn, true_test)
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

def compare_weight_inits(X_train, X_test, target_train, true_test, eta, lmbd, batches, n_epochs):
    # Compare weight initializations using normal distribution and Xavier Glorot initialization
    
    # Normal distribution
    ordinary_weight_model = FFNN(layers, hidden_func=sigmoid, cost_func=cost_func, output_func=identity)
    ordinary_weight_model.reset_weights()
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
    scores = ordinary_weight_model.fit(X_train, target_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)
    pred = ordinary_weight_model.predict(X_test)

    # Xavier Glorot initialization
    XG_weight_model = FFNN(layers, hidden_func=sigmoid, cost_func=cost_func, output_func=identity, use_Xavier_Glorot_weights=True)
    XG_weight_model.reset_weights()
    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
    scores = XG_weight_model.fit(X_train, target_train, scheduler, epochs=n_epochs, batches=batches, lam=lmbd)
    pred_XG = XG_weight_model.predict(X_test)

    # Scores
    score = mean_squared_error(pred, true_test)
    score_XG = mean_squared_error(pred_XG, true_test)
    r2 = r2_score(true_test, pred)
    r2_XG = r2_score(true_test, pred_XG)

    print(f"\nNormal weight initialization score: \tMSE\t{score:.4f}\t R^2\t{r2:.4f}")
    print(f"Xavier Glorot initialization score: \tMSE\t{score_XG:.4f}\t R^2\t{r2_XG:.4f}")

    # Plot
    sns.set()
    plt.plot(X[:,0], y, 'ro', label='Data')
    plt.plot(X[:,0], y_true, 'b-', label='True')
    plt.scatter(X_test[:,0], pred, c='g', marker='v', label='Normal weight initialization', zorder=10)
    plt.scatter(X_test[:,0], pred_XG, c='k', marker='^', label='Xavier Glorot initialization', zorder=10)
    plt.legend()
    # plt.title(r'$f(x) = \frac{1}{5}x^4 - \frac{4}{5}x^3 - \frac{1}{4}x^2$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()
    pass

rng_seed = 2023
n = 100
x = np.linspace(-3, 3, n)
noise = np.random.normal(0, 1.0, n)
y_true = 0.2*x**4 - 0.8*x**3 - 0.25*x**2
y = y_true + noise
X = np.column_stack((x.reshape(-1, 1), y.reshape(-1, 1)))

# Split and scale
X_train, X_test, t_train, t_test, true_train, true_test = train_test_split(X, y.reshape(-1, 1), y_true.reshape(-1, 1), test_size=0.2, random_state=rng_seed)

# Model
input_nodes = X_train.shape[1]
hidden_nodes_1 = 10
hidden_nodes_2 = 5
output_nodes = t_train.shape[1]
layers = (input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes)
n_epochs = 500
batches = 20
cost_func = CostOLS
activation_func = sigmoid
eta_vals = np.logspace(-5, -2, 4)
lmbd_vals = np.logspace(-5, -1, 5)

# create_eta_lambda_heatmap(X_train, t_train, eta_vals, lmbd_vals, n_epochs, batches) # Create heatmap of MSE for different learning rates and lambdas
output_func_compare(X_train, X_test, t_train, eta=0.01, lmbd=0.01, n_epochs=n_epochs, batches=batches) # Compare output functions
# plot_activation_func_comparison(X_train, X_test, t_train, n_epochs, batches, eta=0.01, lmbd=0.01)# Train with different activation functions
# compare_model_and_sklearn(X_train, X_test, t_train, true_test, layers, lmbd=0.01, eta=0.01, batches=batches, n_epochs=n_epochs) # Compare our model with SciKit Learn
# compare_weight_inits(X_train, X_test, t_train, true_test, eta=0.01, lmbd=0.01, batches=batches, n_epochs=n_epochs) # Compare weight initializations