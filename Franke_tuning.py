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

#### Grid search for eta and lambda ####
eta_vals = np.logspace(-5, -1, 5)
lmbd_vals = np.logspace(-5, -1, 5)
print("eta_vals " , eta_vals)

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

#### Grid search for epochs and batches ####
# n = 10
# epoch_space = np.linspace(100, 1000, n)
# batch_space = np.linspace(1, 10, n)
# lam = 0.0001        # Presumably best values from eta-lambda grid search
# eta = 0.001
# train_mse = np.zeros((n, n))
# test_mse = np.zeros((n, n))
# train_r2 = np.zeros((n, n))
# test_r2 = np.zeros((n, n))

# # This is quite slow, btw
# for i, epoch in enumerate(epoch_space):
#     for j, batch in enumerate(batch_space):
#         linear_regression.reset_weights()
#         scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
#         score = linear_regression.fit(X_train, t_train, scheduler, epochs = int(epoch), batches=int(batch), lam=lam)
#         pred_train = linear_regression.predict(X_train)
#         train_mse[i, j] = MSE(pred_train, t_train)
#         train_r2[i, j] = rsquare(t_train, pred_train)
#         pred_test = linear_regression.predict(X_test)
#         test_mse[i, j] = MSE(pred_test, t_test)
#         test_r2[i, j] = rsquare(t_test, pred_test)

# train_mse = pd.DataFrame(train_mse, index = epoch_space, columns = batch_space)
# test_mse = pd.DataFrame(test_mse, index = epoch_space, columns = batch_space)
# train_r2 = pd.DataFrame(train_r2, index = epoch_space, columns = batch_space)
# test_r2 = pd.DataFrame(test_r2, index = epoch_space, columns = batch_space)

# fig, ax = plt.subplots(figsize = (8, 8))
# sns.heatmap(train_mse, annot = True, ax = ax, cmap = "magma" , fmt = ".4f" , norm=LogNorm())
# ax.set_title("Training MSE")
# # ax.set_ylabel("$\eta$")
# # ax.set_xlabel("$\lambda$")
# ax.set_ylabel("Epochs")
# ax.set_xlabel("Batches")
# plt.show()

# fig, ax = plt.subplots(figsize = (8, 8))
# sns.heatmap(test_mse, annot = True, ax = ax, cmap = "magma" , fmt = ".4f" , norm=LogNorm() )
# ax.set_title("Test MSE")
# # ax.set_ylabel("$\eta$")
# # ax.set_xlabel("$\lambda$")
# ax.set_ylabel("Epochs")
# ax.set_xlabel("Batches")
# plt.show()

# fig, ax = plt.subplots(figsize = (8, 8))
# sns.heatmap(train_r2, annot = True, ax = ax, cmap = "magma", fmt = ".4f" , norm=LogNorm())
# ax.set_title("Training R2")
# # ax.set_ylabel("$\eta$")
# # ax.set_xlabel("$\lambda$")
# ax.set_ylabel("Epochs")
# ax.set_xlabel("Batches")
# plt.show()

# fig, ax = plt.subplots(figsize = (8, 8))
# sns.heatmap(test_r2, annot = True, ax = ax, cmap = "magma", fmt = ".4f" , norm=LogNorm())
# ax.set_title("Test R2")
# # ax.set_ylabel("$\eta$")
# # ax.set_xlabel("$\lambda$")
# ax.set_ylabel("Epochs")
# ax.set_xlabel("Batches")
# plt.show()

# print(np.min(train_mse))
# print(np.min(test_mse))
# print(np.max(train_r2))
# print(np.max(test_r2))

# print("r squared", train_r2)

#### Optimizer comparison ####
# eta = 1e-1
moment = 0.3
n_epochs = 500
batches = 50
eta_vals = np.logspace(-6, -1, 6)
lmbd_vals = np.logspace(-5, 0, 6)

fig, ax = plt.subplots(figsize = (9, 9), nrows=3, ncols=2)

for eta, sub in zip(eta_vals, range(len(ax.flatten()))):
    score_dict = {
        'momentum': {
            'scheduler': Momentum(eta=eta, momentum=moment),
            'color': 'blue', 
            'train': {'mse': [], 'r2': []}, 
            'test': {'mse': [], 'r2': []}
        }, 
        'adagrad': {
            'scheduler': Adagrad(eta=eta),
            'color': 'black',
            'train': {'mse': [], 'r2': []}, 
            'test': {'mse': [], 'r2': []}
        }, 
        'adagrad_momentum': {
            'scheduler': AdagradMomentum(eta=eta, momentum=moment),
            'color': 'green',
            'train': {'mse': [], 'r2': []}, 
            'test': {'mse': [], 'r2': []}
        }, 
        'rmsprop': {
            'scheduler': RMS_prop(eta=eta, rho=0.9),
            'color': 'red',
            'train': {'mse': [], 'r2': []}, 
            'test': {'mse': [], 'r2': []}
        }, 
        'adam': {
            'scheduler': Adam(eta=eta, rho=0.9, rho2=0.999),
            'color': 'purple',
            'train': {'mse': [], 'r2': []}, 
            'test': {'mse': [], 'r2': []}
        }
    }
    for key in score_dict.keys():
        for i, lmd in enumerate(lmbd_vals):
            try:
                linear_regression.reset_weights()
                score_dict[key]['scheduler'].reset()
                score = linear_regression.fit(X_train, t_train, score_dict[key]['scheduler'], epochs = n_epochs, batches=10, lam=lmd)
                pred_train = linear_regression.predict(X_train)
                score_dict[key]['train']['mse'].append(MSE(pred_train, t_train))
                score_dict[key]['train']['r2'].append(rsquare(t_train, pred_train))
                pred_test = linear_regression.predict(X_test)
                score_dict[key]['test']['mse'].append(MSE(pred_test, t_test))
                score_dict[key]['test']['r2'].append(rsquare(t_test, pred_test))
            except:
                score_dict[key]['train']['mse'].append(np.nan)
                score_dict[key]['train']['r2'].append(np.nan)
                score_dict[key]['test']['mse'].append(np.nan)
                score_dict[key]['test']['r2'].append(np.nan)
        match sub:
            case 0|1:
                if sub == 0:
                    ax[0, sub].plot(lmbd_vals, score_dict[key]['train']['mse'], label = key, color = score_dict[key]['color'])
                else:
                    ax[0, sub].plot(lmbd_vals, score_dict[key]['train']['mse'], color = score_dict[key]['color'])
                # ax[0, sub].plot(lmbd_vals, score_dict[key]['test']['mse'], color = score_dict[key]['color'], linestyle = "--")
                ax[0, sub].set_xscale("log")
                ax[0, sub].set_title(f"$\eta$ = {eta:10.0e}", fontsize = 14)
                ax[0, sub].set_xlabel("$\lambda$", fontsize = 14)
                ax[0, sub].set_ylabel("MSE", fontsize = 14)
                ax[0, sub].tick_params(axis='both', which='major', length=5)
                ax[0, sub].tick_params(axis='both', which='minor', length=3)
            case 2|3:
                ax[1, sub - 2].plot(lmbd_vals, score_dict[key]['train']['mse'], color = score_dict[key]['color'])
                # ax[1, sub - 2].plot(lmbd_vals, score_dict[key]['test']['mse'], color = score_dict[key]['color'], linestyle = "--")
                ax[1, sub - 2].set_xscale("log")
                ax[1, sub - 2].set_title(f"$\eta$ = {eta:10.0e}", fontsize = 14)
                ax[1, sub - 2].set_xlabel("$\lambda$", fontsize = 14)
                ax[1, sub - 2].set_ylabel("MSE", fontsize = 14)
                ax[1, sub - 2].tick_params(axis='both', which='major', length=5)
                ax[1, sub - 2].tick_params(axis='both', which='minor', length=3)
            case 4|5:
                ax[2, sub - 4].plot(lmbd_vals, score_dict[key]['train']['mse'], color = score_dict[key]['color'])
                # ax[2, sub - 4].plot(lmbd_vals, score_dict[key]['test']['mse'], color = score_dict[key]['color'], linestyle = "--")
                ax[2, sub - 4].set_xscale("log")
                ax[2, sub - 4].set_title(f"$\eta$ = {eta:10.0e}", fontsize = 14)
                ax[2, sub - 4].set_xlabel("$\lambda$", fontsize = 14)
                ax[2, sub - 4].set_ylabel("MSE", fontsize = 14)
                ax[2, sub - 4].tick_params(axis='both', which='major', length=5)
                ax[2, sub - 4].tick_params(axis='both', which='minor', length=3)
        # ax.plot(lmbd_vals, score_dict[key]['test']['mse'], label = key + " test", color = score_dict[key]['color'], linestyle = "--")

# ax.legend(loc='best', fontsize = 10)
labels = ['Momentum', 'Adagrad', 'Momentum Adagrad', 'RMSprop', 'Adam']
fig.subplots_adjust(wspace=0.4, hspace=0.35)
plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=5, fontsize = 10, bbox_transform=fig.transFigure)
plt.show()
