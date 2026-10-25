"""Reproducible first-pass checks of the FWM manuscript.

Run from the repository root: code/.venv/bin/python code/audit_initial.py
Some checks deliberately reproduce errors in the original manuscript; they
assert the counterexample, rather than treating an incorrect identity as true.
"""

import sympy as s
from mpmath import mp

from weierstrass_modified import Weierstrass


u = s.symbols('u1:5')
v = s.symbols('v1:5')
I = [u[j] * v[j] for j in range(4)]
U, V = s.prod(u), s.prod(v)
a = s.symbols('a1:5')
A = s.Matrix(4, 4, lambda j, k: s.Symbol(f'a{min(j,k)+1}{max(j,k)+1}'))
H = U + V - sum(a[j] * I[j] for j in range(4)) - sum(
    A[j, k] * I[j] * I[k] / 2 for j in range(4) for k in range(4)
)


def bracket(f, g):
    return s.expand(sum(s.diff(f, u[j]) * s.diff(g, v[j])
                        - s.diff(f, v[j]) * s.diff(g, u[j]) for j in range(4)))


C = [I[j] - I[3] for j in range(3)]
assert all(bracket(p, H) == V - U for p in I)
assert all(bracket(c, H) == 0 for c in C)
assert all(bracket(c, d) == 0 for c in C for d in C)
assert s.expand(u[0] * s.diff(H, u[0]) - v[0] * s.diff(H, v[0])) == U - V
assert s.Matrix(C).jacobian(u + v).subs(dict(zip(u + v, range(1, 9)))).rank() == 3
print('PASS: modal-product evolution, involution, and generic independence argument')

# General-N local normalisation in the new manuscript subsection. Verify its
# derivative independently on the conserved power line, including weighted modes.
rho, aa, bb, gam, beta = s.symbols('rho aa bb gam beta', nonzero=True)
hh = 1 / (aa + bb*rho)
paired = hh*(gam-rho)/(aa+bb*gam)
assert s.factor(s.diff(paired, rho)+hh**2) == 0
weighted = hh*(gam-beta*rho)/(aa+bb*gam/beta)
assert s.factor(s.diff(weighted, rho)+beta*hh**2) == 0
mode_count = s.Symbol('N', integer=True, positive=True)
assert s.solve(2-mode_count/2, mode_count) == [4]
print('PASS: local-normalisation cancellation and interaction-degree-four condition')

x = s.Symbol('x')
g1, g2, g3 = s.symbols('g1 g2 g3')
b0, b1, b2 = s.symbols('b0 b1 b2')
q = lambda t: b0 + b1 * t + b2 * t**2
L3 = (g1 - x) * (g2 - x) / ((g1 - g3) * (g2 - g3))
L2 = (g1 - x) * (g3 - x) / ((g1 - g2) * (g3 - g2))
L1 = (g2 - x) * (g3 - x) / ((g1 - g2) * (g1 - g3))
printed = L3 * q(g1) + L2 * q(g2) + L1 * q(g3)
corrected = L3 * q(g3) + L2 * q(g2) + L1 * q(g1)
assert s.factor(corrected - q(x)) == 0
sample = {g1: 1, g2: 2, g3: -3, x: 0, b0: 0, b1: 1, b2: 0}
assert printed.subs(sample) == s.Rational(-28, 5)
assert q(x).subs(sample) == 0
print('HISTORICAL TYPO: original interpolation gives -28/5 rather than 0; corrected formula verified')

mp.dps = 60
w = Weierstrass()
om = [mp.mpf(1), mp.mpc('.2', '1.1')]
z = mp.mpf('.17')
for m, n in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (-2, 1)]:
    O = m * om[1] + n * om[0]
    eta = m * w.wzeta(om[1], om) + n * w.wzeta(om[0], om)
    rhs = (-1)**(m*n+m+n) * w.wsigma(z, om) * mp.exp(2 * (z + O) * eta)
    assert abs(w.wsigma(z + 2 * O, om) / rhs - 1) < mp.mpf('1e-50')
print('PASS: quasiperiod law with linear combinations of half-period eta values')
legendre = w.wzeta(om[0], om)*om[1] - w.wzeta(om[1], om)*om[0]
assert abs(legendre - mp.j*mp.pi/2) < mp.mpf('1e-50')
print('PASS: positively oriented Legendre convention')

# Independent numerical check of the displayed 4x4 determinant identity.
nu = [mp.mpf('.23'), mp.mpf('.37'), mp.mpf('.51')]
sigma = lambda t: w.wsigma(t, om)
wp = lambda t: w.wp(t, om)
wpp = lambda t: w.wpprime(t, om)
# The -g2/2 constant in wp'' cancels from the determinant's last column.
M = mp.matrix([[1, wp(z), wpp(z), 6*wp(z)**2]] +
              [[1, wp(t), -wpp(t), 6*wp(t)**2] for t in nu])
const = 12 * mp.fprod(sigma(nu[j] - nu[k]) for j, k in [(0,1),(0,2),(1,2)]) / mp.fprod(sigma(t)**4 for t in nu)
lhs = const * mp.fprod(sigma(z+t) for t in nu) * sigma(z-sum(nu)) / sigma(z)**4
assert abs(lhs / mp.det(M) - 1) < mp.mpf('1e-50')
# Check cancellation for arbitrary lattice representatives, including even shifts.
eta1, eta3 = (w.wzeta(t, om) for t in om)
for p_lattice, q_lattice in [(0,0), (1,0), (0,-1), (2,2), (-2,1)]:
    nu4 = 2*p_lattice*om[0] + 2*q_lattice*om[1] - sum(nu)
    R_modes = -2*(p_lattice*eta1 + q_lattice*eta3)
    absorbed = sigma(z+nu4)/sigma(nu4)*mp.exp(z*R_modes)
    expected = -sigma(z-sum(nu))/sigma(sum(nu))
    assert abs(absorbed/expected-1) < mp.mpf('1e-50')
print('PASS: exponential absorption for zero, odd and even lattice shifts')
# Nonzero half-period specialisations used outside the determinant proof.
for m, n in [(1,0), (0,1), (1,1), (1,3)]:
    a_half = m*om[1] + n*om[0]
    specialized = (-1)**(m*n+m+n)*sigma(z)*mp.exp(2*(z+a_half)*w.wzeta(a_half,om))
    assert abs(sigma(z+2*a_half)/specialized-1) < mp.mpf('1e-50')
print('PASS: finite-zeta specialisations at nonzero half-periods')

print('PASS: displayed 4x4 FS identity on a nonsingular test lattice (60-digit arithmetic)')

# Extra reciprocal exponentials make the un-gauged mode Wronskian vary.
c = mp.mpf('.7')
alpha = mp.mpf('.31')
p = lambda t: sigma(t+alpha) / (sigma(t)*sigma(alpha)) * mp.exp(-w.wzeta(alpha,om)*t)
r = lambda t: sigma(t-alpha) / (sigma(t)*sigma(alpha)) * mp.exp(w.wzeta(alpha,om)*t)
f = lambda t: mp.exp(c*t)*p(t)
g = lambda t: mp.exp(-c*t)*r(t)
wr = lambda f, g, t: mp.diff(f,t)*g(t)-f(t)*mp.diff(g,t)
assert abs(wr(p,r,z) - wr(p,r,z+mp.mpf('.08'))) < mp.mpf('1e-50')
assert abs(wr(f,g,z) - wr(f,g,z+mp.mpf('.08'))) > mp.mpf('.01')
for y in [p, r]:
    assert abs(mp.diff(y,z,2) / y(z) - 2*wp(z)-wp(alpha)) < mp.mpf('1e-50')
print('PASS: constant Wronskian and Lame equation after reciprocal exponential gauge')
print('CONFIRMED: Wronskian generally varies before that gauge')
