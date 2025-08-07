import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predit_prob(X, m, b):
    z = m * X + b
    return sigmoid(z)

def loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def update(X, y_true, y_pred, m, b, learning_rate):
    dm = np.mean(X * (y_pred - y_true))  # Correct gradient
    db = np.mean(y_pred - y_true)

    m = m - learning_rate * dm          # Subtract to minimize loss
    b = b - learning_rate * db

    return m, b

X = np.array([1, 2, 3, 4, 5])
y_true = np.array([0, 0, 0, 1, 1])

m = 0.0
b = 0.0
epochs = 1000
learning_rate = 0.1

loss_history = []

for e in range(epochs):
    y_pred = predit_prob(X, m, b)
    curr_loss = loss(y_true, y_pred)
    loss_history.append(curr_loss)

    if e % 100 == 0:
        print(f"Epoch {e}: Loss = {curr_loss:.4f}, m = {m:.4f}, b = {b:.4f}")
    m, b = update(X, y_true, y_pred, m, b, learning_rate)

print(f"\nFinal loss: {loss_history[-1]:.4f}")
print(f"Trained weights: m = {m:.4f}, b = {b:.4f}")

plt.plot(loss_history)
plt.title("Logistic Regression Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
