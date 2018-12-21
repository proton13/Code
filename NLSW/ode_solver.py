import numpy as np
import math


def solver(lbd, T, eps, t, u0, v0):

    iA = np.array([[1, 0], [0, 1], [0, 0]])
    iB = np.array([[-1, 0], [0, -1]])
    n = int(round(T / t)) + 1
    x = np.zeros(n)
    u = np.zeros(n, dtype=complex)
    v = np.zeros(n, dtype=complex)
    p = np.zeros(n)

    bp = (1 + np.sqrt(1 + 4 * eps**2 * lbd**2)) / (2 * eps**2)
    bm = (-2 * lbd**2) / (1 + np.sqrt(1 + 4 * eps**2 * lbd**2))
    b = bp - bm

    I = np.zeros((5, 5), dtype=complex)
    for Ip in range(5):
        for Im in range(5):
            tb = bp * (Ip - 2) + bm * (Im - 2)
            if tb != 0:
                I[Ip, Im] = 2 * np.exp(1j * tb * t / 2) * \
                    np.sin(tb * t / 2) / tb
            else:
                I[Ip, Im] = t

    u[0] = u0
    v[0] = v0
    p[0] = u[0].real
    print("eps=",eps,"tau=",t)
    for k in range(n):
        x[k] = k * t
    for k in range(1, n):
        gmm = -(bm * u[k - 1] + 1j * v[k - 1]) / b
        nu = (bp * u[k - 1] + 1j * v[k - 1]) / b
        d = u[k - 1] * u[k - 1].conjugate() * u[k - 1] / (eps**2 * b)
        A = np.array([gmm + d / bp, nu - d / bm, -
                      d / bp + d / bm], dtype=complex)
        B = np.array([[np.exp(1j * bp * t), 1j * bp * np.exp(1j * bp * t)],
                      [-np.exp(1j * bm * t), -1j * bm * np.exp(1j * bm * t)]])
        C = np.zeros((5, 5, 2), dtype=complex)

        for ai1 in range(3):
            for ai2 in range(3):
                for ai3 in range(3):
                    for bi in range(2):
                        ci = iA[ai1, 0] - iA[ai2, 0] + \
                            iA[ai3, 0] + iB[bi, 0] + 2
                        cj = iA[ai1, 1] - iA[ai2, 1] + \
                            iA[ai3, 1] + iB[bi, 1] + 2
                        C[ci, cj, 0] += A[ai1] * A[ai2].conjugate() * \
                            A[ai3] * B[bi, 0]
                        C[ci, cj, 1] += A[ai1] * A[ai2].conjugate() * \
                            A[ai3] * B[bi, 1]

        [u1i, u2i] = np.tensordot(C, I, ((0, 1), (0, 1)))

        u[k] = gmm * np.exp(1j * bp * t) + nu * \
            np.exp(1j * bm * t) + u1i * 1j / (eps**2 * b)
        v[k] = gmm * np.exp(1j * bp * t) * 1j * bp + nu * \
            np.exp(1j * bm * t) * 1j * bm + u2i * 1j / (eps**2 * b)

        p[k] = u[k].real

    return (x, u, p)
