# hybrid_cy.pyx - Cython for High-Precision PHI/Kappa
# Compile: cythonize -i hybrid_cy.pyx
from libc.math cimport sin, cos, exp, log, sqrt, M_PI
from cython cimport boundscheck, wraparound
import mpmath
mpmath.mp.dps = 19
@boundscheck(False)
@wraparound(False)
cpdef double compute_phi_kappa(double[:, :] points):
    \"\"\"PHI-Scaled Kappa Computation (19 decimals precision for curvature calculation).\"\"\"
    cdef int n = points.shape[0]
    cdef double[:] l = points[:, 0]
    cdef double[:] h = points[:, 1]
    cdef double[:] dl = memoryview(np.diff(l))
    cdef double[:] dh = memoryview(np.diff(h))
    cdef double[:] d2l = memoryview(np.diff(dl))
    cdef double[:] d2h = memoryview(np.diff(dh))
    cdef double[:] kappa = np.zeros(n-2)
    cdef double phi = float(mpmath.phi)
    cdef int i
    for i in range(n-2):
        kappa[i] = abs(dl[i] * d2h[i] - dh[i] * d2l[i]) / (dl[i]**2 + dh[i]**2)**1.5 * phi
    return np.mean(kappa)
