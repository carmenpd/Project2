import pandas as pd
from ucimlrepo import fetch_ucirepo 
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from FFNN_classification import *
from func_autograd import *
import seaborn as sns
import matplotlib.pyplot as plt
from activation_functions import *

def logistic_regression_sgd(X, y, eta, regularization, n_epochs, size_minibach):
    # define initial beta and m
    beta = np.random.randn(X.shape[1],)
    m = int(X.shape[0]/size_minibach)
    
    # loops for using SGD
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = size_minibach * np.random.randint(m)
            xi = X[random_index: random_index+ size_minibach]
            yi = y[random_index: random_index+ size_minibach]
            # compute the gradients and then update beta
            gradients = (np.squeeze(sigmoid(xi @ beta)) - yi) @ xi + regularization * beta # sigmoid function because we are doing logistic regression
            beta -= eta*gradients
    return beta

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets
df = pd.DataFrame(np.column_stack((X,y)))
df = df.dropna()
df = np.array(df)
X = df[:, :-1]
y = df[:, -1].astype(int)  # Convert y values to integer type
y = np.where(y == 2, 0, 1) # Map 2 to 0 and 4 to 1. If this is not done, we get 5 categories (0, 1, 2, 3, 4)
#y = to_categorical_numpy(y) # Convert to one-hot encoding

X_train, X_test, t_train, t_test = train_test_split(X, y)

eta_vals = np.logspace(-5, -1, 6)
lmb_vals = np.logspace(-6, -1, 6)
train_accuracy = np.zeros((len(eta_vals), len(lmb_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmb_vals)))

for i, eta in enumerate(eta_vals):
    for j, lmb in enumerate(lmb_vals):
        betas = logistic_regression_sgd(X_train, t_train, eta, lmb, 200, 15)
        # make the prediction and convert it into 0 if the sigmoid function is lower than 0.5, and 1 otherwise
        train_pred = np.where(sigmoid(X_train @ betas) < 0.5, 0, 1)
        test_pred = np.where(sigmoid(X_test @ betas) < 0.5, 0, 1)
        train_accuracy[i,j] = metrics.accuracy_score(t_train, train_pred)
        test_accuracy[i,j] = metrics.accuracy_score(t_test, test_pred)


fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(train_accuracy, annot = True, ax = ax, cmap = "viridis")
ax.set_title("Training accuracy - Logistic Regression")
ax.set_ylabel("Eta")
ax.set_xlabel("Regularization parameter")
plt.show()

print(test_accuracy)
fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(test_accuracy, annot = True, ax = ax, cmap = "viridis")
ax.set_title("Test accuracy - Logistic Regression")
ax.set_ylabel("Eta")
ax.set_xlabel("Regularization parameter")
plt.show()

print("Maximum train accuracy:", np.max(train_accuracy))
print("Maximum text accuracy:", np.max(test_accuracy))


# Now, do the same but wuth scikit-learn's logistic regression functionality

train_accuracy_sl = np.zeros((len(eta_vals), len(lmb_vals)))
test_accuracy_sl = np.zeros((len(eta_vals), len(lmb_vals)))

for i, eta in enumerate(eta_vals):
    for j, lmb in enumerate(lmb_vals):
        betas = logistic_regression_sgd(X_train, t_train, eta, lmb, 200, 15)
        # make the prediction and convert it into 0 if the sigmoid function is lower than 0.5, and 1 otherwise
        train_pred = np.where(sigmoid(X_train @ betas) < 0.5, 0, 1)
        test_pred = np.where(sigmoid(X_test @ betas) < 0.5, 0, 1)
        train_accuracy_sl[i,j] = metrics.accuracy_score(t_train, train_pred)
        test_accuracy_sl[i,j] = metrics.accuracy_score(t_test, test_pred)

print("Maximum train accuracy scikit-learn:", np.max(train_accuracy_sl))
print("Maximum text accuracy scikit-learn:", np.max(test_accuracy_sl))
