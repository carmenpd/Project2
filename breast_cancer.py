import pandas as pd
from ucimlrepo import fetch_ucirepo 
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from FFNN import *
from func_autograd import *
import seaborn as sns
import matplotlib.pyplot as plt
  
# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets 
  
# metadata 
#print(breast_cancer_wisconsin_original.metadata) 
  
# variable information 
#print(breast_cancer_wisconsin_original.variables)

#print(breast_cancer_wisconsin_original)


X_train, X_test, t_train, t_test = train_test_split(X, y)

input_nodes = X_train.shape[1]
hidden_nodes_1 = input_nodes//2
hidden_nodes_2 = input_nodes//3
output_nodes = t_train.shape[1]
print(input_nodes)
print(output_nodes)

classification = FFNN((input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes), output_func=softmax, cost_func=Accuracy, seed=2023)

eta_vals = np.logspace(-5, 0, 6)
lmbd_vals = np.logspace(-5, 0, 6)
train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals): 
        classification.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

        scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
        scores = classification.fit(X_train, t_train, scheduler, epochs = 100, batches=2, lam=lmbd)

        pred_train = classification.predict(X_train)
        train_mse[i, j] = Accuracy(pred_train, t_train)
        #train_r2[i, j] = rsquare(pred_train, t_train)
        
        pred_test = classification.predict(X_test)
        test_mse[i, j] = Accuracy(pred_test, t_test)
        #test_r2[i, j] = rsquare(pred_test, t_test)


fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(train_mse, annot = True, ax = ax, cmap = "viridis")
ax.set_title("Training MSE")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(test_mse, annot = True, ax = ax, cmap = "viridis")
ax.set_title("Test MSE")
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

print(np.min(train_mse))
print(np.min(test_mse))
#print(np.max(train_r2))
#print(np.max(test_r2))