import autograd.numpy as np
from sklearn.model_selection import train_test_split
from FFNN import *
from func_autograd import *
import seaborn as sns
import matplotlib.pyplot as plt

def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y**k)

    return X

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
    return 1-(np.sum((y-ypredict)**2)/np.sum((y-np.mean(y))**2))

step = 0.005
x = np.arange(0, 1, step)
y = np.arange(0, 1, step)
# what about random input like in project 1? I did a few runs and couldnt tell whats better
#x = np.random.rand(200)
#y = np.random.rand(200)
x, y = np.meshgrid(x, y)
target = FrankeFunction(x, y)
target = target.reshape(-1, 1)

poly_degree = 7
X = create_X(x, y, poly_degree)

X_train, X_test, t_train, t_test = train_test_split(X, target)

input_nodes = X_train.shape[1]
output_nodes = t_train.shape[1]

linear_regression = FFNN((input_nodes, output_nodes), output_func=identity, cost_func=CostOLS, seed=2023)

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals): 
        linear_regression.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

        scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
        scores = linear_regression.fit(X_train, t_train, scheduler, epochs = 100, batches=2, lam=lmbd)

        pred_train = linear_regression.predict(X_train)
        train_mse[i, j] = MSE(pred_train, t_train)
        train_r2[i, j] = rsquare(pred_train, t_train)
        
        pred_test = linear_regression.predict(X_test)
        test_mse[i, j] = MSE(pred_test, t_test)
        test_r2[i, j] = rsquare(pred_test, t_test)


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

print(np.min(train_mse))
print(np.min(test_mse))
print(np.max(train_r2))
print(np.max(test_r2))