import numpy as np
import autograd.numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error, r2_score
from autograd import grad

class GradientDescend:
    def __init__(self, optimizer="gd", learning_rate=0.01, max_epochs=1000, batch_size=5,
                 learning_rate_decay=0.9, patience=20, delta_momentum=0.3, lmb=0.001,
                 tol=1e-4, change=0.0 ,delta=  1e-8, rho =0.99, beta1 = 0.9 , beta2 = 0.999 , momentum=True,
                 learning_rate_decay_flag=False, Ridge=False , method = None):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.thetas = 0.0
        self.first_moment = 0.0
        self.second_moment = 0.0
        self.iter = 0.0
        self.rho = rho
        self.beta1 = beta1
        self.beta2 = beta2
        self.delta = delta
        self.method = method
        self.batch_size = int(batch_size)
        self.learning_rate_decay = learning_rate_decay
        self.patience = patience
        self.delta_momentum = delta_momentum if momentum else 0
        self.lmb = lmb if Ridge else 0
        self.tol = tol
        self.gradient_squared = 0.0     # Accumulative squared gradient

        self.change = change
        self.learning_rate_decay_flag = learning_rate_decay_flag

    def compute_hessian_eig_max(self, X):
        H = (2.0 / len(X)) * np.matmul(X.T, X)
        if self.lmb:
            H += 2 * self.lmb * np.eye(X.shape[1])
        return 1.0 / np.max(np.linalg.eigvals(H))

    def compute_gradient(self, X, y, thetas):
        residuals = np.dot(X, thetas) - y
        gradient = 2.0 * np.dot(X.T, residuals) / len(X) + 2 * self.lmb * thetas
        return gradient

    def cost_function(self, X, y, beta):
        residuals = np.dot(X, beta) - y
        cost = np.sum(residuals ** 2) / len(X) + self.lmb * (np.sqrt(np.sum(beta ** 2)))**2  # The sqrt is the squared L2 norm (autograd doesn't support np.linalg.norm arguments like ord=2)
        return cost

    def gradient_descent_step(self, X, y, thetas):
        gradient = self.compute_gradient(X, y, thetas)
        change = self.learning_rate * gradient + self.delta_momentum * self.change
        thetas -= change
        self.change = change
        return thetas

    def adagrad_step(self, X, y, thetas):
        gradient = grad(self.cost_function, 2)(X, y, thetas)
        self.gradient_squared = gradient ** 2
        adjusted_grad = gradient / (np.sqrt(self.gradient_squared) + self.delta)
        thetas -= self.learning_rate * adjusted_grad
        return thetas

    def rmsprop_step(self, X, y, thetas):
        gradient = grad(self.cost_function, 2)(X, y, thetas)
        self.gradient_squared = self.rho * self.gradient_squared + (1 - self.rho) * gradient ** 2
        adjusted_grad = gradient / (np.sqrt(self.gradient_squared) + self.delta)
        thetas -= self.learning_rate * adjusted_grad
        return thetas

    def adam_step(self, X, y, thetas):
        gradients = grad(self.cost_function, 2)(X, y, thetas)
        self.first_moment = self.beta1*self.first_moment + (1-self.beta1)*gradients
        self.second_moment = self.beta2*self.second_moment+(1-self.beta2)*gradients**2
        first_term = self.first_moment/(1.0-self.beta1**self.iter)
        second_term = self.second_moment/(1.0-self.beta2**self.iter)
        # Scaling with rho the new and the previous results
        update = self.learning_rate*first_term/(np.sqrt(second_term)+self.delta)
        thetas -= update
        return thetas

    def _gradient_descent(self, X_train, y_train, X_val, y_val):
        thetas = np.random.randn(X_train.shape[1],1)
        patience_counter = 0
        best_val_loss = float('inf')
        best_thetas = np.copy(thetas)


        for epoch in range(self.max_epochs):
            thetas = self.gradient_descent_step(X_train, y_train, thetas)

            if self.learning_rate_decay_flag:
                val_loss = mean_squared_error(y_val, np.dot(X_val, thetas))
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    best_thetas = thetas
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > self.patience:
                        self.learning_rate *= self.learning_rate_decay
                        patience_counter = 0

        return best_thetas

    def _stochastic_gradient_descent(self, X_train, y_train, X_val, y_val):
        num_samples = len(X_train)
        thetas = np.random.randn(X_train.shape[1],1)
        patience_counter = 0
        best_val_loss = float('inf')
        best_thetas = np.copy(thetas)


        if self.method == 'gd':
            update_fn = self.gradient_descent_step
        elif self.method == 'adam':
            self.m = np.zeros((X_train.shape[1], 1))
            self.v = np.zeros((X_train.shape[1], 1))
            self.t = 0
            update_fn = self.adam_step
        elif self.method == 'rmsprop':
            update_fn = self.rmsprop_step
        elif self.method == 'adagrad':
            update_fn = self.adagrad_step
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

        for epoch in range(self.max_epochs):
            indices = np.random.permutation(num_samples)
            self.iter += 1      # Before loop, else division by zero in adam_step
            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
                thetas = update_fn(batch_X, batch_y, thetas)

            if self.learning_rate_decay_flag:
                val_loss = mean_squared_error(y_val, np.dot(X_val, thetas))
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    best_thetas = thetas
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > self.patience and self.learning_rate_decay_flag:
                        self.learning_rate *= self.learning_rate_decay

        return best_thetas
    
    def fit(self, X_train, y_train, X_val, y_val):
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        if not self.learning_rate_decay_flag:
            self.learning_rate = self.compute_hessian_eig_max(X_train)

        if self.optimizer == "gd":
            self.thetas = self._gradient_descent(X_train, y_train, X_val, y_val)
            return self.thetas
        elif self.optimizer == "sgd":
            self.thetas = self._stochastic_gradient_descent(X_train, y_train, X_val, y_val)
            return  self.thetas
        else:
            raise ValueError("Unsupported optimizer. Use 'gd' or 'sgd'.")
        
    def predict(self, X):
        return np.dot(X, self.thetas).reshape(-1, 1)


def generate_data(noise=True, step_size=0.05 , FrankesFunction=True):
    # Arrange x and y
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)
    # Create meshgrid of x and y
    X, Y = np.meshgrid(x, y)
    
    if FrankesFunction:
        # Calculate the values for Franke function
        z = FrankeFunction(X, Y, noise=noise).flatten()
    else:
        z = TestFunction(X, Y, noise=noise).flatten()

    # Flatten x and y for plotting
    x = X.flatten()
    y = Y.flatten()
    
    return x, y, z

def TestFunction(x, y, noise=False):
    if noise: 
        random_noise = np.random.normal(0, 0.1 , x.shape)
    else: 
        random_noise = 0

    return  x**2 + y**2 + 2*x*y + random_noise

def FrankeFunction(x, y, noise=False):
    if noise: 
        random_noise = np.random.normal(0, 0.1 , x.shape)
    else: 
        random_noise = 0
    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + random_noise

def create_X(x, y, n, intercept=True):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)

    if intercept:
        X = np.ones((N,l))

        for i in range(1,n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)
    else:
        X = np.ones((N,l-1))

        for i in range(1,n+1):
            q = int((i)*(i+1)/2) - 1
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)
    return X