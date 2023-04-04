import numpy.linalg
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

datadir = "/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/EX7.mat"
data = spio.loadmat(datadir)
# print(type(data))

dtz = data['X']/255
# print(dtz.shape)
dth = np.squeeze(data['y']/255)
# print(dth.shape)


"""

zznorm = np.linalg.norm(dtz @ dtz.T, ord=2, axis=None)
lbound = zznorm*2/n
print(lbound)

"""


n = 784
d = 60000

M = 100
N = 2000
c = 1.2
w = np.linspace(0.002, 0.2, M)


"""

dtu = np.zeros ((M,d))
fu = np.zeros(M)
gu = np.zeros(M)
c0unt = np.zeros(M)

lbound = 5830


for i1 in range(0, M):

    u = np.zeros(d)
    v = np.zeros(d)
    L = 100

    for i2 in range(0, N):
        hzuterm = dth - dtz @ u
        u1coef = dtz.T @ hzuterm * 2/n
        u0val = pow(np.linalg.norm(hzuterm), 2) / n
        for ind in range(1, 50):
            test = 0
            v = u + u1coef/L
            for i3 in range(0, d):
                if v[i3] < -w[i1]/L:
                    v[i3] = v[i3] + w[i1]/L
                elif v[i3] > w[i1]/L:
                    v[i3] = v[i3] - w[i1]/L
                else:
                    v[i3] = 0
            if L >= lbound:
                break
            hzvterm = dth - dtz @ v
            fvreal = pow(np.linalg.norm(hzvterm), 2) / n
            fvapprx = u0val - np.transpose(v-u) @ u1coef + L/2 * pow(np.linalg.norm(v-u), 2)
            if fvreal < fvapprx:
                break
            else:
                L = c * L

        u = v

    for i3 in range(0, d):
        dtu[i1,i3] = u[i3]
        if u[i3] == 0:
            c0unt[i1] = c0unt[i1] + 1

    hzuterm = dth - dtz @ u
    fu[i1] = pow(np.linalg.norm(hzuterm), 2)/n
    gu[i1] = np.linalg.norm(u, ord=1)*w[i1]

    print(round(w[i1], 4), round(fu[i1], 4), round(gu[i1], 4))
    print(60000-c0unt[i1])

np.save('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/dtu.npy', dtu)
np.save('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/count.npy', 60000-c0unt)
np.save('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/fu.npy', fu)
np.save('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/gu.npy', gu)

"""


c0unt = np.round(60000 - np.load('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/count.npy'))
fu = np.round(np.load('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/fu.npy'), 4)
gu = np.round(np.load('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/gu.npy'), 4)


plt.figure(figsize=(8, 8), dpi=400)
plt.xlim(0, 0.2)
plt.ylim(0, 0.1)
plt.xlabel('lambda')
plt.ylabel('value')
plt.plot(w, fu, color="red", label="f(u_opt)")
plt.plot(w, gu, color="blue", label="g(u_opt)")
plt.plot(w, fu+gu, color="purple", label="sum")
plt.legend()
plt.savefig('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/1.png')
plt.clf()

plt.figure(figsize=(8, 8), dpi=400)
plt.xlim(0.08, 0.2)
plt.ylim(0, 80)
plt.xlabel('lambda')
plt.ylabel('sparsity')
plt.plot(w, 60000-c0unt, color="green", label="non-zero dims")
plt.savefig('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/2.png')
plt.clf()

plt.figure(figsize=(8, 8), dpi=400)
plt.xlim(-3, -0.5)
plt.ylim(0, 10000)
plt.xlabel('lg lambda')
plt.ylabel('sparsity')
plt.scatter(np.log10(w), 60000-c0unt, c="green", alpha=0.6, label="non-zero dims")
plt.savefig('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/3.png')
plt.clf()


"""

dtl = np.squeeze(data['label'])
dtu = np.round(np.load('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/dtu.npy'), 4)

dtl1 = np.zeros((d, 10))
for i3 in range(0, d):
    num = dtl[i3]
    dtl1[i3, num] = 1

prdy1 = np.round(dtu @ dtl1, 3)
np.save('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/prdy.npy', prdy1)

"""

prdy1 = np.round(np.load('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/prdy.npy'), 4)

check = M
for i1 in range(0,M):
    if (prdy1[M-i1-1,] != np.zeros(10)).any():
        break
    else:
        check = check-1

prdy = np.argmax(prdy1,axis=1)

plt.figure(figsize=(6, 6), dpi=400)
plt.xlim(0, 0.2)
plt.ylim(-1, 10)
plt.xlabel('lambda')
plt.ylabel('prediction')
plt.scatter(w[0:check], prdy[0:check], c="cyan", alpha=0.3)
plt.savefig('/Users/mac/Documents/ML-PSets-icloud/ML-PSet4/4.png')
plt.clf()
