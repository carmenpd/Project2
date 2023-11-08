import pandas as pd
from ucimlrepo import fetch_ucirepo 
from FFNN import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# one-hot in numpy
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

df = pd.DataFrame(np.column_stack((X, y)))
df = df.dropna()

# Convert the dataframe back to numpy array
df = np.array(df)

# Split the array into features and targets
X = df[:, :-1]
y = df[:, -1].astype(int)  # Convert y values to integer type
y = np.where(y == 2, 0, 1) # Map 2 to 0 and 4 to 1

  
# metadata 
print(breast_cancer_wisconsin_original.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_original.variables)

print(breast_cancer_wisconsin_original)


print("This is X " , X)
print("This is y " , y)
y = to_categorical_numpy(y)

print("This is y_onehot " , y)
print("This is y shape " , y.shape)

#Split dataset
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)

#Scaling dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Setting architecture on the FFNN
input_neurons = X_train.shape[1]
hidden_neurons1 = 5
hidden_neurons2 = 3
output_neurons = y_train.shape[1]
layers = (input_neurons , hidden_neurons1 , hidden_neurons2 ,  output_neurons)

#Deciding the activation function and the cost function
model = FFNN(layers , output_func=sigmoid , cost_func=CostLogReg , seed=2023  )
model.reset_weights()

#Setting the optimizer
scheduler = Adam(eta=0.5 , rho=0.9 , rho2=0.999 )

#Training the model
scores = model.fit(X_train , y_train , scheduler , batches= 1, epochs=100 , lam = 0.00001 , X_val= X_test ,t_val= y_test)

#Testing the model
result = model.predict(X_test)
accuracy = model._accuracy(y_test , result)

print("This is result " , result)
print("This is the accuracy of the training " , accuracy)


# For the entire dataset (including both features and target)
element = df[9, :]

# For only the features
element_features = X[9, :]

# For only the target
element_target = y[9]

print("First element (entire dataset):", element)
print("First element (only features):", element_features)
print("First element (only target):", element_target)

result = model.predict(element_features)
print("This is the result " , result)

