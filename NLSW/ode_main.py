import numpy as np
import matplotlib.pyplot as plt
import ode_solver

lbd = 1
alp = 2
omg = 0.1
u0 = 2
T = 4

lve = np.array([2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
ve = 1. / lve
nve = lve.size

lvt = 10*np.array([10**4, 2**1, 2**2, 2**3, 2**4,
                   2**5, 2**6, 2**7, 2**8, 2**9])
vt = 1. / lvt
nvt = lvt.size

error_y = np.zeros((nve, nvt - 1))
ratio = np.zeros((nve, nvt - 2))

for ie in range(nve):
    eps = ve[ie]
    v0 = 1j * (-lbd**2 * u0 - np.abs(u0)**2 * u0) + eps**alp * omg
    for it in range(nvt):
        t = vt[it]
        (x, u, p) = ode_solver.solver(lbd, T, eps, t, u0, v0)
        '''
        print(x[-1])
        plt.plot(x, p, label="eps=" + str(eps))
        if(it == 1):
        plt.plot(x, p, label="tau=" + str(t))
        plt.legend()
        '''
        if it == 0:
            st = u
        else:
            error_y[ie, it - 1] = np.abs(st[-1] - u[-1])
        if it >= 2:
            ratio[ie, it - 2] = np.log(error_y[ie, it - 2] /
                                       error_y[ie, it - 1]) / np.log(vt[it - 1] / vt[it])


# plt.show()
print('ratio')
print(ratio)
print('error')
print(error_y)
