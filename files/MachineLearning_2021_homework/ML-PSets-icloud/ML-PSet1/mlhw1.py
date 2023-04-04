import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt("data1.txt", dtype=float)

l = len(data1)
x_0 = np.zeros(l)
x_1 = np.zeros((l, 2))
x_2 = np.zeros((l, 3))
for i in range(0, l):
    x_0[i] = data1[i, 0]
for i in range(0, l):
    x_1[i, 0] = 1
    x_1[i, 1] = x_0[i]
for i in range(0, l):
    x_2[i, 0] = 1
    x_2[i, 1] = x_0[i]
    x_2[i, 2] = round(x_0[i] * x_0[i], 8)
y = [0]*l
for i in range(0, l):
    y[i] = data1[i, 1]
w_1 = np.linalg.inv(x_1.T @ x_1) @ x_1.T @ y
w_2 = np.linalg.inv(x_2.T @ x_2) @ x_2.T @ y
y_1 = x_1 @ w_1
y_2 = x_2 @ w_2

er_1 = er_2 = 0
for i in range(0, l):
    er_1 = er_1 + (y[i] - y_1[i]) * (y[i] - y_1[i])
    er_2 = er_2 + (y[i] - y_2[i]) * (y[i] - y_2[i])
er_1 = round(er_1 / l, 8)
er_2 = round(er_2 / l, 8)

print("Fitting Coefficient of order 1 \n", w_1)
print("Fitting Coefficient of order 2 \n", w_2)
print("Empirical error of order 1 \n", er_1)
print("Empirical error of order 2 \n", er_2)

plt.figure()
plt.xlim(-1, 1)
plt.ylim(0, 1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_0, y, color="green")
xx = np.linspace(-1, 1, 50)
yy_1 = w_1[0] + w_1[1] * xx
plt.plot(xx, yy_1, color="red")
yy_2 = w_2[0] + w_2[1] * xx + w_2[2] * xx * xx
plt.plot(xx, yy_2, color="blue")
plt.show()
