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
#breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
breast_cancer_wisconsin_original = pd.read_csv("wdbc.data", delimiter=',')  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.drop(breast_cancer_wisconsin_original.columns[1], axis = 1)
y = breast_cancer_wisconsin_original.iloc[:,1]
df = pd.DataFrame(np.column_stack((X,y)))
df = df.dropna()
df = np.array(df)
#X = df[:, :-1]
#y = df[:, -1].astype(int)  # Convert y values to integer type
y = np.where(y == "B", 0, 1) # Map 2 to 0 and 4 to 1. If this is not done, we get 5 categories (0, 1, 2, 3, 4)
y = to_categorical_numpy(y) # Convert to one-hot encoding


X_train, X_test, t_train, t_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

input_nodes = X_train.shape[1]
hidden_nodes_1 = input_nodes//2
hidden_nodes_2 = input_nodes//3
output_nodes = t_train.shape[1] # corresponds to the number of categories, i.e. 2
hidden_activation_func = sigmoid
output_activation_func = sigmoid
cost_function = CostLogReg

eta_vals = np.logspace(-5, 0, 6)
lmbd_vals = np.logspace(-5, 0, 6)

# TRY DIFFERENT ACTIVATION FUNCTIONS

fig, ax = plt.subplots(figsize = (9, 9), nrows=3, ncols=2)

for eta, sub in zip(eta_vals, range(len(ax.flatten()))):
    score_dict = {
         'sigmoid': {
             'scheduler': Adam(eta=eta, rho=0.9, rho2=0.999),
             'activationfunction': sigmoid,
             'color': 'blue', 
             'train': {'accuracy score': []}, 
             'test': {'accuracy score': []}
         }, 
        'softmax': {
            'scheduler': Adam(eta=eta, rho=0.9, rho2=0.999),
            'activationfunction': softmax,
            'color': 'red',
            'train': {'accuracy score': []}, 
            'test': {'accuracy score': []}
        }
    }
    for key in score_dict.keys():
        for i, lmd in enumerate(lmbd_vals):
            try:
                classif = FFNN_classification((input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes) , hidden_func= score_dict[key]['activationfunction'] , output_func= identity, cost_func=CostCrossEntropy, seed=2023)
                classif.reset_weights()
                score = classif.fit(X_train, t_train, scheduler=score_dict[key]['scheduler'], epochs = 200, batches=10, lam=lmd)
                pred_train = classif.predict(X_train)
                score_dict[key]['train']['accuracy score'].append(metrics.accuracy_score(t_train, pred_train))
                pred_test = classif.predict(X_test)
                score_dict[key]['test']['accuracy score'].append(metrics.accuracy_score(t_test, pred_test))
            except:
                score_dict[key]['train']['accuracy score'].append(np.nan)
                score_dict[key]['test']['accuracy score'].append(np.nan)
        match sub:
            case 0|1:
                if sub == 0:
                    ax[0, sub].plot(lmbd_vals, score_dict[key]['train']['accuracy score'], label = key, color = score_dict[key]['color'])
                else:
                    ax[0, sub].plot(lmbd_vals, score_dict[key]['train']['accuracy score'], color = score_dict[key]['color'])
                # ax[0, sub].plot(lmbd_vals, score_dict[key]['test']['mse'], color = score_dict[key]['color'], linestyle = "--")
                ax[0, sub].set_xscale("log")
                ax[0, sub].set_title(f"$\eta$ = {eta:10.0e}", fontsize = 14)
                ax[0, sub].set_xlabel("$\lambda$", fontsize = 14)
                ax[0, sub].set_xlim([lmbd_vals[0], lmbd_vals[-1]])
                ax[0, sub].set_ylabel("Accuracy score", fontsize = 14)
                ax[0, sub].set_ylim([0, 1])
                ax[0, sub].tick_params(axis='both', which='major', length=5)
                ax[0, sub].tick_params(axis='both', which='minor', length=3)
            case 2|3:
                ax[1, sub - 2].plot(lmbd_vals, score_dict[key]['train']['accuracy score'], color = score_dict[key]['color'])
                # ax[1, sub - 2].plot(lmbd_vals, score_dict[key]['test']['mse'], color = score_dict[key]['color'], linestyle = "--")
                ax[1, sub - 2].set_xscale("log")
                ax[1, sub - 2].set_title(f"$\eta$ = {eta:10.0e}", fontsize = 14)
                ax[1, sub - 2].set_xlabel("$\lambda$", fontsize = 14)
                ax[1, sub - 2].set_xlim([lmbd_vals[0], lmbd_vals[-1]])
                ax[1, sub - 2].set_ylabel("Accuracy score", fontsize = 14)
                ax[1, sub - 2].set_ylim([0, 1])
                ax[1, sub - 2].tick_params(axis='both', which='major', length=5)
                ax[1, sub - 2].tick_params(axis='both', which='minor', length=3)
            case 4|5:
                ax[2, sub - 4].plot(lmbd_vals, score_dict[key]['train']['accuracy score'], color = score_dict[key]['color'])
                # ax[2, sub - 4].plot(lmbd_vals, score_dict[key]['test']['mse'], color = score_dict[key]['color'], linestyle = "--")
                ax[2, sub - 4].set_xscale("log")
                ax[2, sub - 4].set_title(f"$\eta$ = {eta:10.0e}", fontsize = 14)
                ax[2, sub - 4].set_xlabel("$\lambda$", fontsize = 14)
                ax[2, sub - 4].set_xlim([lmbd_vals[0], lmbd_vals[-1]])
                ax[2, sub - 4].set_ylabel("Accuracy score", fontsize = 14)
                ax[2, sub - 4].set_ylim([0, 1])
                ax[2, sub - 4].tick_params(axis='both', which='major', length=5)
                ax[2, sub - 4].tick_params(axis='both', which='minor', length=3)
        # ax.plot(lmbd_vals, score_dict[key]['test']['mse'], label = key + " test", color = score_dict[key]['color'], linestyle = "--")

# with open('mydictionary.csv' , 'w') as f:
#     w = csv.DictWriter(f, score_dict.keys())
#     w.writeheader()
#     w.writerow(score_dict)

# ax.legend(loc='best', fontsize = 10)
labels = ['sigmoid', 'softmax']
fig.subplots_adjust(wspace=0.4, hspace=0.5)
plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=5, fontsize = 10, bbox_transform=fig.transFigure)
plt.show()


# EXAMPLE WITH A SPECIFIC ACTIVATION FUNCTION

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
test_max_acc = 0

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals): 
        classification.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

        scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
        scores = classification.fit(X_train, t_train, scheduler, epochs = 200, batches=10, lam=lmbd)

        pred_train = classification.predict(X_train)
        train_accuracy[i, j] = metrics.accuracy_score(t_train, pred_train)
        
        pred_test = classification.predict(X_test)
        test_accuracy[i, j] = metrics.accuracy_score(t_test, pred_test)

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

print("Maximum training accuracy:", np.max(train_accuracy))
print("Maximum test accuracy:", np.max(test_accuracy))

# plot the confusion matrix 
ConfusionMatrixDisplay.from_predictions(t_test.argmax(axis=1), best_test_pred.argmax(axis=1), normalize = "true")
plt.title("Confusion matrix - Neural Network")
plt.show()