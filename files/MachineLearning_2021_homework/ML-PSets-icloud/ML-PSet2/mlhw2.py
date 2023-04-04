import numpy as np
import matplotlib.pyplot as plt


L = 100
N = 25
R = 120
S = 40


tup_y = tuple()
tup_xty = tuple()
x = np.zeros((N, 24))
xtx = np.zeros((24, 24))
temp1 = np.zeros(24)

for i in range(0, L):

    data = np.loadtxt('/Users/mac/PycharmProjects/ML-PSets/ML-PSet2/data/data_%d' % (i+1), dtype=float)

    if i == 0:

        for j in range(0, 24):

            x[:, j] = np.exp(-np.power((data[:, 0] - 0.2 * (j - 12.5)), 2))

        xtx = x.T @ x

    y = np.zeros((N, 1))
    y[:, 0] = data[:, 1]
    xty = x.T @ y

    tup_y = tup_y + (y,)
    tup_xty = tup_xty + (xty,)

    if i < 25:

        plt.figure(figsize=(8, 8), dpi=200)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-2, 2)
        plt.xlabel('x')
        plt.ylabel('y')
        xx = np.linspace(-1.5, 1.5, R)

        plt.scatter(data[:, [0]], data[:, [1]], color="yellow")

        clr = ["red", "purple", "blue", "green"]
        t_ind = 0

        for t in [-10, -5, -1, 1]:

            w = np.linalg.inv(xtx + np.identity(24) * np.float_power(10, t)) @ xty

            yy = np.linspace(0, 0, R)

            for m in range(0, R):

                for j in range(0, 24):

                    temp1[j] = np.exp(-np.power((xx[m]-0.2*(j-12.5)), 2))

                yy[m] = temp1 @ w

            plt.plot(xx, yy, color=clr[t_ind])
            t_ind = t_ind + 1

        plt.savefig('/Users/mac/PycharmProjects/ML-PSets/ML-PSet2/output1/2.%d.png' % (i+1))
        plt.clf()


tup_bias2 = tuple()
tup_var = tuple()
tup_sum = tuple()
h = np.zeros(N)

for k in range(0, N):

    for i in range(0, L):

        h[k] = h[k] + tup_y[i][k]

    h[k] = h[k] / L

tt = np.linspace(-5, 1, S)

for s in range(0, S):

    tup_w = tuple()
    y_bar = np.zeros(N)
    y_l = np.zeros((L, N))

    xtx_tt = np.linalg.inv(xtx + np.identity(N - 1) * np.float_power(10, tt[s]))

    for i in range(0, L):

        w = xtx_tt @ tup_xty[i]

        tup_w = tup_w + (w,)

    for k in range(0, N):

        for i in range(0, L):

            y_l[i, k] = x[k, :] @ tup_w[i]

            y_bar[k] = y_bar[k] + y_l[i, k]

        y_bar[k] = y_bar[k] / L

    bias2 = 0

    for k in range(0, N):

        bias2 = bias2 + np.power(y_bar[k] - h[k], 2)

    bias2 = bias2 / N

    tup_bias2 = tup_bias2 + (bias2,)

    var = 0
    v = np.zeros(N)
    temp2 = 0

    for k in range(0, N):

        for i in range(0, L):

            temp2 = temp2 + np.power(y_l[i, k] - y_bar[k], 2)

        temp2 = temp2 / L

        var = var + temp2

    var = var / N

    tup_var = tup_var + (var,)
    tup_sum = tup_sum + (bias2+var,)

plt.figure(figsize=(8, 8), dpi=200)
plt.xlim(-5.5, 1.5)
plt.ylim(0, 0.2)
plt.xlabel('lg \lambda')
plt.ylabel('Error')
plt.plot(tt, tup_bias2, color="red", label="Bias^2")
plt.plot(tt, tup_var, color="blue", label="Variance")
plt.plot(tt, tup_sum, color="purple", label="Sum")
plt.legend()
plt.savefig('/Users/mac/PycharmProjects/ML-PSets/ML-PSet2/output1/3.png')
plt.clf()