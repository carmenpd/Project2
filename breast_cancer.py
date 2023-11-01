import pandas as pd
from ucimlrepo import fetch_ucirepo 
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from FFNN_classification import *
from func_autograd import *
import seaborn as sns
import matplotlib.pyplot as plt
  
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
  
# metadata 
#print(breast_cancer_wisconsin_original.metadata) 
  
# variable information 
#print(breast_cancer_wisconsin_original.variables)

#print(breast_cancer_wisconsin_original)


X_train, X_test, t_train, t_test = train_test_split(X, y)

input_nodes = X_train.shape[1]
hidden_nodes_1 = input_nodes//2
hidden_nodes_2 = input_nodes//3
output_nodes = t_train.shape[1] # corresponds to the number of categories, i.e. 2
print(input_nodes)
print(output_nodes)

# choices for hidden activation: sigmoid
# choices for output activation: sigmoid (for binary output), softmax
# choices for cost function: CostCrossEntropy (almost zero is th optimal result) or Accuracy (between zero and 1, 1 is the optimal result)
classification = FFNN_classification((input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes), output_func=sigmoid, cost_func=CostCrossEntropy, seed=2023)

eta_vals = np.logspace(-5, 0, 6)
lmbd_vals = np.logspace(-5, 0, 6)
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals): 
        classification.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

        scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
        scores = classification.fit(X_train, t_train, scheduler, epochs = 100, batches=2, lam=lmbd)

        pred_train = classification.predict(X_train)
        train_accuracy[i, j] = metrics.accuracy_score(t_train, pred_train)
        
        pred_test = classification.predict(X_test)
        test_accuracy[i, j] = metrics.accuracy_score(t_test, pred_test)

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(train_accuracy, annot = True, ax = ax, cmap = "viridis")
ax.set_title("Training accuracy/cross entropy score")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(test_accuracy, annot = True, ax = ax, cmap = "viridis")
ax.set_title("Test accuracy/cross entropy score")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

"""
fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(train_r2, annot = True, ax = ax, cmap = "viridis")
ax.set_title("Training R2")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(test_r2, annot = True, ax = ax, cmap = "viridis")
ax.set_title("Test R2")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
"""

print(np.min(train_accuracy))
print(np.min(test_accuracy))
#print(np.max(train_r2))
#print(np.max(test_r2))