import numpy as np
import matplotlib.pyplot as plt
import ode_solver

lbd = 1
alp = 0
omg = 1j* 2
u0 = 0.5
T = 1

lve = np.array([2**2, 2**3, 2**4,2**5, 2**12])
ve = 1. / lve
nve = lve.size

lvt = 10*np.array([10**4])
vt = 1. / lvt
nvt = lvt.size

#error_y = np.zeros((nve, nvt - 1))
#ratio = np.zeros((nve, nvt - 2))

for ie in range(nve):
    eps = ve[ie]
    v0 = 1j * (-lbd**2 * u0 - np.abs(u0)**2 * u0) + eps**alp * omg
    for it in range(nvt):
        t = vt[it]
        (x, u, p) = ode_solver.solver(lbd, T, eps, t, u0, v0)
        print(x[-1])
        if (ie != 4):
            plt.plot(x, p, label="$\epsilon$=$1/2^" + str(ie+2)+"$")
        else:
            plt.plot(x, p, label="$\epsilon$=0")
        '''
        if(it == 1):
            plt.plot(x, p, label="tau=" + str(t))
        '''
        plt.legend()
        '''
        if it == 0:
            st = u
        else:
            error_y[ie, it - 1] = np.abs(st[-1] - u[-1])
        if it >= 2:
            ratio[ie, it - 2] = np.log(error_y[ie, it - 2] /
                                       error_y[ie, it - 1]) / np.log(vt[it - 1] / vt[it])
        '''

plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()
'''
np.set_printoptions(precision=3)
print('ratio')
print(ratio)
print('error')
print(error_y)
'''
