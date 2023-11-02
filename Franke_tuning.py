import autograd.numpy as np
from sklearn.model_selection import train_test_split
from FFNN import *
from func_autograd import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Define a function for Mean Square Error
def MSE(y_data, y_model):
	n = np.size(y_model)
	y_data = y_data.reshape(-1,1)
	y_model = y_model.reshape(-1,1)
	return np.sum((y_data - y_model)**2)/n

# Define a function for R2
def rsquare(y, ypredict):
    ypredict = ypredict.reshape(-1,1)
    y = y.reshape(-1,1)
    return 1-(np.sum((y-ypredict)**2)/(np.sum((y-np.mean(y))**2) + 1e-10))

#Generate data
N = 150
x = np.random.rand(N)
y = np.random.rand(N)

print("x " , x)
print("y " , y)
x, y  = np.meshgrid(x,y)

target = FrankeFunction(x ,  y).ravel()
target = target.reshape(-1,1)
print("target " , target.shape)
# Create design matrix
X = np.column_stack((x.ravel(), y.ravel()))


X_train, X_test, t_train, t_test = train_test_split(X, target)

print("X_train shape " , X_train.shape)
print("t_train shape " , t_train.shape)

input_nodes = X_train.shape[1]
hidden_nodes_1 = 10
hidden_nodes_2 = 5
output_nodes = t_train.shape[1]
print("Input nodes" , input_nodes)
print("output nodes " , output_nodes)

linear_regression = FFNN((input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes) , hidden_func= RELU , output_func= identity, cost_func=CostOLS, seed=2023)

eta_vals = np.logspace(-5, -1, 5)
lmbd_vals = np.logspace(-5, -1, 5)

# train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
# test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
# train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
# test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))

# for i, eta in enumerate(eta_vals):
#     for j, lmbd in enumerate(lmbd_vals): 
#         linear_regression.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

#         scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
#         scores = linear_regression.fit(X_train, t_train, scheduler, epochs = 200, batches=10, lam=lmbd)

#         pred_train = linear_regression.predict(X_train)
#         train_mse[i, j] = MSE(pred_train, t_train)
#         train_r2[i, j] = rsquare(t_train, pred_train)
        
#         pred_test = linear_regression.predict(X_test)
#         test_mse[i, j] = MSE(pred_test, t_test)
#         test_r2[i, j] = rsquare(t_test , pred_test)

n = 10
epoch_space = np.linspace(100, 1000, n)
batch_space = np.linspace(1, 10, n)
lam = 0.0001        # Presumably best values from eta-lambda grid search
eta = 0.001
train_mse = np.zeros((n, n))
test_mse = np.zeros((n, n))
train_r2 = np.zeros((n, n))
test_r2 = np.zeros((n, n))

# This is quite slow, btw
for i, epoch in enumerate(epoch_space):
    for j, batch in enumerate(batch_space):
        linear_regression.reset_weights()
        scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
        score = linear_regression.fit(X_train, t_train, scheduler, epochs = int(epoch), batches=int(batch), lam=lam)
        pred_train = linear_regression.predict(X_train)
        train_mse[i, j] = MSE(pred_train, t_train)
        train_r2[i, j] = rsquare(t_train, pred_train)
        pred_test = linear_regression.predict(X_test)
        test_mse[i, j] = MSE(pred_test, t_test)
        test_r2[i, j] = rsquare(t_test, pred_test)

train_mse = pd.DataFrame(train_mse, index = epoch_space, columns = batch_space)
test_mse = pd.DataFrame(test_mse, index = epoch_space, columns = batch_space)
train_r2 = pd.DataFrame(train_r2, index = epoch_space, columns = batch_space)
test_r2 = pd.DataFrame(test_r2, index = epoch_space, columns = batch_space)

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(train_mse, annot = True, ax = ax, cmap = "magma" , fmt = ".4f" , norm=LogNorm())
ax.set_title("Training MSE")
# ax.set_ylabel("$\eta$")
# ax.set_xlabel("$\lambda$")
ax.set_ylabel("Epochs")
ax.set_xlabel("Batches")
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(test_mse, annot = True, ax = ax, cmap = "magma" , fmt = ".4f" , norm=LogNorm() )
ax.set_title("Test MSE")
# ax.set_ylabel("$\eta$")
# ax.set_xlabel("$\lambda$")
ax.set_ylabel("Epochs")
ax.set_xlabel("Batches")
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(train_r2, annot = True, ax = ax, cmap = "magma", fmt = ".4f" , norm=LogNorm())
ax.set_title("Training R2")
# ax.set_ylabel("$\eta$")
# ax.set_xlabel("$\lambda$")
ax.set_ylabel("Epochs")
ax.set_xlabel("Batches")
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(test_r2, annot = True, ax = ax, cmap = "magma", fmt = ".4f" , norm=LogNorm())
ax.set_title("Test R2")
# ax.set_ylabel("$\eta$")
# ax.set_xlabel("$\lambda$")
ax.set_ylabel("Epochs")
ax.set_xlabel("Batches")
plt.show()

# print(np.min(train_mse))
# print(np.min(test_mse))
# print(np.max(train_r2))
# print(np.max(test_r2))

# print("r squared", train_r2)