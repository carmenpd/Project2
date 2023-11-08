import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from FFNN_classification import *
from func_autograd import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


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
y = to_categorical_numpy(y) # Convert to one-hot encoding


X_train, X_test, t_train, t_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

input_nodes = X_train.shape[1]
hidden_nodes_1 = input_nodes//2
hidden_nodes_2 = input_nodes//3
output_nodes = t_train.shape[1] # corresponds to the number of categories, i.e. 2
hidden_activation_func = sigmoid
output_activation_func = sigmoid
cost_function = CostLogReg


# choices for hidden activation: sigmoid
# choices for output activation: sigmoid (for binary output), softmax
# choices for cost function: CostCrossEntropy (almost zero is th optimal result) or Accuracy (between zero and 1, 1 is the optimal result)
classification = FFNN_classification((input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes), 
                                     hidden_func=hidden_activation_func, output_func=output_activation_func, 
                                     cost_func=cost_function, seed=2023)

eta_vals = np.logspace(-5, 0, 6)
lmbd_vals = np.logspace(-5, 0, 6)
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
test_max_acc = 0

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals): 
        classification.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

        scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
        scores = classification.fit(X_train, t_train, scheduler, epochs = 200, batches=10, lam=lmbd)

        pred_train = classification.predict(X_train)
        train_accuracy[i, j] = metrics.accuracy_score(t_train, pred_train)
        train_r2[i, j] = metrics.r2_score(t_train, pred_train)
        
        pred_test = classification.predict(X_test)
        test_accuracy[i, j] = metrics.accuracy_score(t_test, pred_test)
        test_r2[i, j] = metrics.r2_score(t_test, pred_test)

        # find the best prediction and best eta and lambda
        if test_accuracy[i,j] > test_max_acc:
            test_max_acc = test_accuracy[i,j]
            best_test_pred = pred_test
            best_eta = eta
            best_lambda = lmbd

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(train_accuracy, annot = True, ax = ax, cmap = "magma")
ax.set_title("Training accuracy score - Neural Network")
ax.set_ylabel("$\log_{10}\eta$")
ax.set_yticklabels(np.log10(eta_vals))
ax.set_xlabel("$\log_{10}\lambda$")
ax.set_xticklabels(np.log10(lmbd_vals))
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(test_accuracy, annot = True, ax = ax, cmap = "magma")
ax.set_title("Test accuracy score - Neural Network")
ax.set_ylabel("$\log_{10}\eta$")
ax.set_yticklabels(np.log10(eta_vals))
ax.set_xlabel("$\log_{10}\lambda$")
ax.set_xticklabels(np.log10(lmbd_vals))
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(train_r2, annot = True, ax = ax, cmap = "magma")
ax.set_title("Training R2 - Neural Network")
ax.set_ylabel("$\log_{10}\eta$")
ax.set_yticklabels(np.log10(eta_vals))
ax.set_xlabel("$\log_{10}\lambda$")
ax.set_xticklabels(np.log10(lmbd_vals))
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(test_r2, annot = True, ax = ax, cmap = "magma")
ax.set_title("Test R2 - Neural Network")
ax.set_ylabel("$\log_{10}\eta$")
ax.set_yticklabels(np.log10(eta_vals))
ax.set_xlabel("$\log_{10}\lambda$")
ax.set_xticklabels(np.log10(lmbd_vals))
plt.show()


print("Maximum training accuracy:", np.max(train_accuracy))
print("Maximum test accuracy:", np.max(test_accuracy))

# plot the confusion matrix 
ConfusionMatrixDisplay.from_predictions(t_test.argmax(axis=1), best_test_pred.argmax(axis=1), normalize = "true")
plt.title("Confusion matrix - Neural Network")
plt.show()