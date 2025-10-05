import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Definition of Some Variables
hx = 0
teta0 = 1
teta1 = 4
new_teta0 = 0
new_teta1 = 0
x = []
y = []
x_norm = 0
y_norm = 0
m = 1000
alpha = 1e-2
epsilon = 1e-6
max_iter = 100000
count = 0

# Calculate h(x(i))
def h(x_norm, i):
return teta0 + teta1 * x_norm[i]

# Generate Data for Traning Module
# x = np.random.uniform(1,1000,m)
# y = np.random.uniform(1,1000,m)
x = np.random.normal(loc=500, scale=150, size=m)
y = 2 * x + 50 + np.random.normal(0, 200, m)

# Normalized Data
x_norm = (x - np.mean(x)) / np.std(x)
y_norm = (y - np.mean(y)) / np.std(y)

while True:
dt0 = 0
dt1 = 0
J = 0
# Calculate MSE and its Gradient
for i in range(m):
hx = h(x_norm, i)
J += (hx - y_norm[i]) ** 2
dt0 += (hx - y_norm[i])
dt1 += (hx - y_norm[i]) * x_norm[i]

J /= 2 * m
dt0 /= m
dt1 /= m

# Calculate New Tetas
new_teta0 = teta0 - alpha * dt0
new_teta1 = teta1 - alpha * dt1

# Convergence Condition
if math.sqrt((new_teta0 - teta0) ** 2 + (new_teta1 - teta1) ** 2) < epsilon and math.sqrt((dt0) ** 2 + (dt1) ** 2) < epsilon:
break

# Update Tetas Values
teta0 = new_teta0
teta1 = new_teta1

# Break Down the Program if its Repeat Goes to Infinite
count += 1
if count > max_iter:
break

# Show Teta 0 and Teta 1 Value
print(f"The value of Teta 0 is: {teta0}")
print(f"The value of Teta 1 is: {teta1}")

# Generate two Seperated Plot in one Figure
fig, ax = plt.subplots(1,2, figsize=(12,5))

#Plot Normalized Data and Normalized Regression Line

# Generate X Axsis and Calculate the Y for each specific X
x_line = np.linspace(min(x_norm), max(x_norm), 100)
y_line = teta0 + teta1 * x_line

# Plot Data, Title and Label for each Axsis
ax[0].scatter(x_norm, y_norm, alpha=0.5)
ax[0].plot(x_line, y_line, color='red', linewidth=2, label="Prediction Line")
ax[0].set_xlabel("X (Normalized)")
ax[0].set_ylabel("Y (Normalized)")
ax[0].set_title("Normalized Data with Regression Line")
ax[0].legend()

# Plot Original Data and Original Regression Line

# Convert Tetas Values from Normalized to Original
teta1_original = teta1 * (np.std(y) / np.std(x))
teta0_original = teta0 * np.std(y) + np.mean(y) - teta1_original * np.mean(x)

# Generate X Axsis and Calculate the Y for each specific X
x_line = np.linspace(min(x), max(x), 100)
y_line = teta0_original + teta1_original * x_line

# Plot Data, Title and Label for each Axsis
ax[1].scatter(x, y, alpha=0.5)
ax[1].plot(x_line, y_line, color='red', linewidth=2, label="Regression Line")
ax[1].set_xlabel("X (Original)")
ax[1].set_ylabel("Y (Original)")
ax[1].set_title("Original Data with Regression Line")
ax[1].legend()

# Config Layout
plt.tight_layout()
plt.show()