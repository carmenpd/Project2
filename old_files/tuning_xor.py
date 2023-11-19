from FFNN import *
import seaborn as sns
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)

# The XOR gate
yXOR = np.array([[0], [1] ,[1], [0]])

input_nodes = X.shape[1]
output_nodes = yXOR.shape[1]

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

logistic_regression = FFNN((input_nodes, 2, output_nodes), output_func=sigmoid, cost_func=CostLogReg, seed=2023)

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals): 
        logistic_regression.reset_weights() # reset weights such that previous runs or reruns don't affect the weights
        scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)
        scores = logistic_regression.fit(X, yXOR, scheduler, epochs=1000, lam=lmbd)

        prediction = logistic_regression.predict(X)

        train_accuracy[i, j] = logistic_regression._accuracy(prediction, yXOR)

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(train_accuracy, annot = True, ax = ax, cmap = "viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()