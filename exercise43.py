from FNN import *

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# The XOR gate
yXOR = np.array([ 0, 1 ,1, 0])
# The OR gate
yOR = np.array([ 0, 1 ,1, 1])
# The AND gate
yAND = np.array([ 0, 0 ,0, 1])
layers = [2, 2, 1]

model = NeuralNetwork(layers, X)

N = 100
updated_weights = [np.array([0 , 0 , 0 , 0 , 0])]
updated_biases = [np.array([0 , 0 , 0 , 0 , 0])]


for i in range(N):
    # renamed to weights_update and bias_update to avoid overwriting each iteration
    weights_update, bias_update = model.adjust_weights_biases(X, yXOR, updated_weights[-1], updated_biases[-1])
    updated_weights.append(np.array(weights_update))
    updated_biases.append(np.array(bias_update))

print("Predictions: " + str(model.predict(X)))
