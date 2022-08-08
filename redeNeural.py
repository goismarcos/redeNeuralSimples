import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def MSE(Y_target, Y_pred):
    return np.mean((Y_target - Y_pred) ** 2)

base = pd.read_csv('base_dados.csv')

factures = base.iloc[:,0:4].values

target = np.array([base.iloc[:,4].values])

target = target.reshape(-1, 1)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(factures,
                                                                  target,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
N = 0.001

EPOCHS = 200

cost = np.array([])

n_neurons_input_layer = 4
n_neurons_hidden_layer_1 = 8
n_neurons_hidden_layer_2 = 4
n_neurons_hidden_layer_3 = 2
n_neurons_output_layer = 1

w_hidden_layer_1 = np.random.rand(n_neurons_input_layer, n_neurons_hidden_layer_1)
w_hidden_layer_2 = np.random.rand(n_neurons_hidden_layer_1, n_neurons_hidden_layer_2)
w_hidden_layer_3 = np.random.rand(n_neurons_hidden_layer_2, n_neurons_hidden_layer_3)
w_output_layer = np.random.rand(n_neurons_hidden_layer_3, n_neurons_output_layer)

b_hidden_layer_1 = np.zeros(n_neurons_hidden_layer_1)
b_hidden_layer_2 = np.zeros(n_neurons_hidden_layer_2)
b_hidden_layer_3 = np.zeros(n_neurons_hidden_layer_3)
b_output_layer = np.zeros(n_neurons_output_layer)

for epoch in range(EPOCHS):
    
    activation_hidden_layer_1 = sigmoid(np.dot(X_treinamento, w_hidden_layer_1) + b_hidden_layer_1)
    activation_hidden_layer_2 = sigmoid(np.dot(activation_hidden_layer_1, w_hidden_layer_2) + b_hidden_layer_2)
    activation_hidden_layer_3 = sigmoid(np.dot(activation_hidden_layer_2, w_hidden_layer_3) + b_hidden_layer_3)
    activation_output_layer = sigmoid(np.dot(activation_hidden_layer_3, w_output_layer) + b_output_layer)
    
    cost = np.append(cost, MSE(y_treinamento, activation_output_layer))
    
    delta_output_layer = (y_treinamento - activation_output_layer) * sigmoid_derivative(activation_output_layer)
    delta_hidden_layer_3 = np.dot(delta_output_layer, w_output_layer.T) * sigmoid_derivative(activation_hidden_layer_3)
    delta_hidden_layer_2 = np.dot(delta_hidden_layer_3, w_hidden_layer_3.T) * sigmoid_derivative(activation_hidden_layer_2)
    delta_hidden_layer_1 = np.dot(delta_hidden_layer_2, w_hidden_layer_2.T) * sigmoid_derivative(activation_hidden_layer_1)
    
    w_output_layer += N * np.dot(activation_hidden_layer_3.T, delta_output_layer)
    w_hidden_layer_3 += N * np.dot(activation_hidden_layer_2.T, delta_hidden_layer_3)
    w_hidden_layer_2 += N * np.dot(activation_hidden_layer_1.T, delta_hidden_layer_2)
    w_hidden_layer_1 += N * np.dot(X_treinamento.T, delta_hidden_layer_1)
    
plt.plot(cost)
plt.title('Função de custo da rede')
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.show()



    