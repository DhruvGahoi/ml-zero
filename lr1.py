import numpy as np
import matplotlib . pyplot as plt

# Here we are making a basic Linear Regression Model

m = 0.5
b = 20

X = np.array([2, 3, 5, 6, 8])
y_true = np.array([65, 70, 75, 85, 90])

def predict(X, m, b):
    # y = mx + b equation
    return m * X + b

y_pred = predict(X, m, b)
print(y_pred)

def loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

init_loss = loss(y_true, y_pred)
print(f"Initial Loss : {init_loss}")

learning_rate = 0.01
def update(X, y_true, y_pred, m, b, learning_rate):
    dm = -2 * np.mean(X * (y_true - y_pred))
    db = -2 * np.mean(y_true - y_pred)

    new_m = m - learning_rate * dm
    new_b = b - learning_rate * db
    return new_m, new_b

# Reinitialized variables 
m = 0.0
b = 0.0
epochs = 2000
learning_rate = 0.01

loss_history = []
for i in range(epochs):
    y_pred = predict(X, m, b)
    curr_loss = loss(y_true, y_pred)
    loss_history.append(curr_loss)

    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {curr_loss:.2f}, m = {m:.2f}, b = {b:.2f}")

    m, b = update(X, y_true, y_pred, m, b, learning_rate)

print("\n --- Training Complete ---")
print(f"Final Loss: {loss_history[-1]:.2f}")
print(f"Optimal parameters: m = {m:.2f}, b = {b:.2f}")


plt.plot(loss_history)
plt.title("Loss Curve during Training")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()
