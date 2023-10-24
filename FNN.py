from func_autograd  import *
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, layers ,X_train , epochs = 50 , batch_size = 5 , eta = 0.01 , lmbd = 0.01 , Ridge = False)  :
        """Initialize the neural network.
        
        Args:
            layers (list): A list where the first item is the number of input neurons,
                           the last item is the number of output neurons, and the 
                           intermediate items are the number of neurons in each hidden layer.
        """
        self.layers = layers
        self.weights = []
        self.biases = []   
        self.n_inputs, self.n_features = X_train.shape
        self.n_hidden_layers = len(layers) - 2
        self.n_layers = len(layers)
        self.weights = []
        self.biases = []

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd if Ridge else 0
        

        # Initialize weights and biases
        for i in range(self.n_layers - 1):
            # Weights
            W = np.random.randn(layers[i], layers[i+1])
            self.weights.append(W)
            
            # Biases
            b = np.zeros(layers[i+1]) + 0.01
            self.biases.append(b)
        

    def __repr__(self): 
        return "Neural Network with architecture: " + " -> ".join(str(layer) for layer in self.layers)


        # one-hot in numpy
    def to_categorical_numpy(self, integer_vector):
        n_inputs = len(integer_vector)
        n_categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros((n_inputs, n_categories))
        onehot_vector[range(n_inputs), integer_vector] = 1
        
        return onehot_vector
    
   

    def feed_forward_train(self, X):
        a = X
        activations = [a]  # to store the activations for all layers

        # Loop through each layer except the last one (output layer)
        for i in range(self.n_hidden_layers):
            z = np.matmul(a, self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            activations.append(a)

        # Handle the output layer separately
        z_o = np.matmul(a, self.weights[-1]) + self.biases[-1]
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        activations.append(probabilities)
        return activations

    def backpropagation(self, X, Y):
        activations = self.feed_forward_train(X)
        
        # List to store errors for all layers
        errors = [None] * len(self.layers) 
        
        # error in the output layer
        errors[-1] = activations[-1] - Y
        
        # loop backward through layers, starting from the last hidden layer
        # note that the derivative of the sigmoid func is derivative = sigmoid*(1 - sigmoid)
        for i in range(len(self.layers)-2, 0, -1):
            errors[i] = np.matmul(errors[i+1], self.weights[i].T) * activations[i] * (1 - activations[i])
        
        # List to store gradients for weights and biases
        weights_gradient = []
        biases_gradient = []
        
        # Compute gradients for each layer
        for i in range(len(self.layers)-2, -1, -1):
            if i == 0:  # Input layer
                weights_gradient.insert(0, np.matmul(X.T, errors[i+1]))
            else:
                weights_gradient.insert(0, np.matmul(activations[i].T, errors[i+1]))
                
            biases_gradient.insert(0, np.sum(errors[i+1], axis=0))
        
        return weights_gradient, biases_gradient
        
    def predict(self, X):
        probabilities = self.feed_forward_train(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_train(X)
        return probabilities
        
    def accuracy_on_training_data(self, X_train, Y_train):
        print("Old accuracy on training data: " + str(accuracy_score(self.predict(X_train), Y_train)))

    def adjust_weights_biases( self , X ,Y , updated_weights , updated_biases):
        
        weights_gradient, biases_gradient = self.backpropagation(X, Y)
        # Used list comprehension because its not possible to add a scalar (lambda * updated_weight[-1]) to a numpy array (weights_gradient)
        # doesnt updated weights have to have (hidden neurons+output neuron) many elements?
        # I think its weights_gradient[i] and not i, isnt it?
        weights_gradient = [weights_gradient[i] + self.lmbd * updated_weights[-1] for i in weights_gradient]  # L2 regularization
        biases_gradient = [biases_gradient[i] + self.lmbd * updated_biases[-1] for i in biases_gradient]     # L2 regularization
        
        # Simple gradient descent update rule
        for w, w_grad in zip(self.weights, weights_gradient):
            w_updated = w - self.eta * w_grad
            updated_weights = np.append(updated_weights, w_updated)
        
        for b, b_grad in zip(self.biases, biases_gradient):
            b_updated = b - self.eta * b_grad
            updated_biases = np.append(updated_biases, b_updated)
        
        return updated_weights, updated_biases