#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


dataset= pd.read_csv(r"C:\Users\user\Downloads\moonDataset.csv")
X = dataset.iloc[:, :-1].values  # inputs: x1, x2, x3
y = dataset.iloc[:, -1].values  # labels: 0 or 1

# Initial weights and biases
W1 = np.array([0.2, -0.3, 0.4])  # Weights for hidden layer neuron 1
W2 = np.array([0.1, -0.5, 0.2])  # Weights for hidden layer neuron 2
W3 = np.array([-0.3, -0.2])  # Weights for output layer
b1 = -0.4  # Bias for hidden layer neuron 1
b2 = 0.2   # Bias for hidden layer neuron 2
b3 = 0.1   # Bias for output layer

# Activation function: Sigmoid
def sigmoid(x)
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Loss function: Squared error
def loss_function(target, output):
    return (target - output) ** 2

# Derivative of loss function
def loss_derivative(target, output):
    return -2 * (target - output)

# Learning rate for SGD
learning_rate = 0.1

# Number of epochs
epochs = 20
لا
# To store loss amount after each epoch
loss_history = []

# Training loop
for epoch in range(epochs):
    epoch_loss = 0
    
    # Loop through each example in the dataset
    for i in range(len(X)):
        # Forward pass
        x1, x2, x3 = X[i]
        target = y[i]
        
        # Hidden layer neuron 1
        z1 = W1[0] * x1 + W1[1] * x2 + W1[2] * x3 + b1
        h1 = sigmoid(z1)
        
        # Hidden layer neuron 2
        z2 = W2[0] * x1 + W2[1] * x2 + W2[2] * x3 + b2
        h2 = sigmoid(z2)
        
        # Output layer
        z3 = W3[0] * h1 + W3[1] * h2 + b3
        output = sigmoid(z3)
        
        # Calculate loss for this example
        loss = loss_function(target, output)
        epoch_loss += loss
        
        # Backward pass
        # Compute gradients for output layer
        d_loss_output = loss_derivative(target, output)
        d_output_z3 = sigmoid_derivative(z3)
        d_loss_z3 = d_loss_output * d_output_z3
        
        # Gradients for weights and biases in output layer
        grad_W3_0 = d_loss_z3 * h1
        grad_W3_1 = d_loss_z3 * h2
        grad_b3 = d_loss_z3
        
        # Compute gradients for hidden layer
        d_loss_h1 = d_loss_z3 * W3[0]
        d_loss_h2 = d_loss_z3 * W3[1]
        
        # Gradients for hidden layer neuron 1
        d_h1_z1 = sigmoid_derivative(z1)
        grad_W1_0 = d_loss_h1 * d_h1_z1 * x1
        grad_W1_1 = d_loss_h1 * d_h1_z1 * x2
        grad_W1_2 = d_loss_h1 * d_h1_z1 * x3
        grad_b1 = d_loss_h1 * d_h1_z1
        
        # Gradients for hidden layer neuron 2
        d_h2_z2 = sigmoid_derivative(z2)
        grad_W2_0 = d_loss_h2 * d_h2_z2 * x1
        grad_W2_1 = d_loss_h2 * d_h2_z2 * x2
        grad_W2_2 = d_loss_h2 * d_h2_z2 * x3
        grad_b2 = d_loss_h2 * d_h2_z2
        
        # Update weights and biases using SGD
        W3[0] -= learning_rate * grad_W3_0
        W3[1] -= learning_rate * grad_W3_1
        b3 -= learning_rate * grad_b3
        
        W1[0] -= learning_rate * grad_W1_0
        W1[1] -= learning_rate * grad_W1_1
        W1[2] -= learning_rate * grad_W1_2
        b1 -= learning_rate * grad_b1
        
        W2[0] -= learning_rate * grad_W2_0
        W2[1] -= learning_rate * grad_W2_1
        W2[2] -= learning_rate * grad_W2_2
        b2 -= learning_rate * grad_b2
        
    # Store epoch loss
    loss_history.append(epoch_loss / len(X))
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(X)}")

# Plot loss over epochs
plt.plot(range(1, epochs + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.show()


# In[ ]:




