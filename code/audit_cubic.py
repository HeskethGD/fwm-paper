"""Generic four-mode cubic tests with nonzero Lambda_1, for both b2 signs.
Run from the repository root: code/.venv/bin/python code/audit_cubic.py
Checks the corrected paper formulas and the actual notebook evaluator.
"""
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from mpmath import mp
from scipy.integrate import solve_ivp
from weierstrass_modified import Weierstrass

mp.dps=60
we=Weierstrass()

def lattice_invariants(a,b):
    q=mp.exp(mp.pi*mp.j*b/a)
    t2,t3=mp.jtheta(2,0,q),mp.jtheta(3,0,q)
    scale=mp.pi/(2*a)
    return (mp.mpf(4)/3*scale**4*(t2**8-(t2*t3)**4+t3**8),
            mp.mpf(8)/27*scale**6*(t2**12-mp.mpf(3)/2*(t2**8*t3**4+t2**4*t3**8)+t3**12))

nb=json.loads((Path(__file__).parent/'Numeric plots for four-wave mixing.ipynb').read_text())
namespace={'we':we,'mpexp':mp.exp}
exec(''.join(nb['cells'][20]['source']),namespace)
notebook_evaluator=namespace['kron_weierstrass_sigma_cubic']
for sign in (1,-1):
    A=[[mp.mpf(t) for t in row] for row in [
        ['2','.2','-.1','.3'],['.2','-.5','.4','-.2'],
        ['-.1','.4','1','.1'],['.3','-.2','.1','.1']]]
    A[3][3]+=4*(sign-1)
    a=[mp.mpc('.2','.1'),mp.mpf('-.3'),mp.mpc('.1','-.2'),mp.mpf('.4')]
    u0=[mp.mpc('.7','.1'),mp.mpc('.9','-.2'),mp.mpc('1.1','.15'),mp.mpc('.8','.3')]
    v0=[mp.mpc('.6','-.12'),mp.mpc('1.2','.1'),mp.mpc('.75','-.2'),mp.mpc('.95','.08')]
    I0=[u0[j]*v0[j] for j in range(4)]
    rho0=-sum(I0)/4; gamma=[ij+rho0 for ij in I0]
    H=mp.fprod(u0)+mp.fprod(v0)-sum(a[j]*I0[j] for j in range(4))-sum(A[j][k]*I0[j]*I0[k]/2 for j in range(4) for k in range(4))
    b0=H+sum(a[j]*gamma[j] for j in range(4))+sum(A[j][k]*gamma[j]*gamma[k]/2 for j in range(4) for k in range(4))
    b1=-sum(a)-sum(A[j][k]*gamma[k] for j in range(4) for k in range(4))
    b2=sum(sum(row) for row in A)/2
    assert abs(b2-2*sign)<mp.mpf('1e-55')
    es={k:sum(mp.fprod(gamma[j] for j in js) for js in combinations(range(4),k)) for k in (2,3,4)}
    d0=b0*b0-4*es[4];d1=2*b0*b1+4*es[3]
    d2=2*b0*b2+b1*b1-4*es[2];d3=2*b1*b2
    g2=d2*d2/12-d1*d3/4
    g3=-d0*d3*d3/16+d1*d2*d3/48-d2**3/216
    om=list(we.omega_from_g(g2,g3)[:2])
    if mp.im(om[1]/om[0])<0:om[1]=-om[1]
    om=list(mp.findroot(lambda x,y:tuple(c-d for c,d in zip(lattice_invariants(x,y),(g2,g3))),tuple(om),tol=mp.mpf('1e-55')))
    y0=we.invwp(d3*rho0/4+d2/12,om,d3*(mp.fprod(u0)-mp.fprod(v0))/4)
    q=lambda t:b0+b1*t+b2*t*t
    aj=[we.invwp(d3*gj/4+d2/12,om,d3*q(gj)/4) for gj in gamma]
    total=sum(sum(row) for row in A)
    L1=[sum(row)-total/4 for row in A]
    L0=[-a[j]-gamma[j]*total/4-sum(A[j][k]*gamma[k] for k in range(4))
        +sum(gamma[k]*sum(A[k])/2 for k in range(4))+sum(a)/2 for j in range(4)]
    assert all(abs(l)>mp.mpf('.1') for l in L1)
    sigma=lambda z:we.wsigma(z,om)
    def bare(t,j,conj=False):
        y=t+y0
        E=t*(L0[j]-d2*L1[j]/(3*d3)-we.wzeta(aj[j],om))-4*L1[j]*we.wzeta(y,om)/d3
        return 2/mp.sqrt(d3)*sigma(y+(-aj[j] if conj else aj[j]))/(sigma(aj[j])*sigma(y))*mp.exp(-E if conj else E)
    alpha=[u0[j]/bare(0,j) for j in range(4)]
    def solution(t):return [alpha[j]*bare(t,j) for j in range(4)]+[bare(t,j,True)/alpha[j] for j in range(4)]
    def rhs(t,vals):
        u,v=vals[:4],vals[4:]
        diag=[a[j]+sum(A[j][k]*u[k]*v[k] for k in range(4)) for j in range(4)]
        return [-diag[j]*u[j]+mp.fprod(v[k] for k in range(4) if k!=j) for j in range(4)]+[diag[j]*v[j]-mp.fprod(u[k] for k in range(4) if k!=j) for j in range(4)]
    assert max(abs(x-y) for x,y in zip(solution(0),u0+v0))<mp.mpf('1e-50')
    residual=mp.mpf(0)
    for t in [mp.mpf(0),mp.mpf('.04'),mp.mpc('.08','.02')]:
        expected=rhs(t,solution(t))
        residual=max(residual,max(abs(mp.diff(lambda z:solution(z)[j],t)-expected[j]) for j in range(8)))
    assert residual<mp.mpf('1e-48'),residual
    grid=np.linspace(0,.2,41)
    numerical=solve_ivp(lambda t,y:np.array([complex(x) for x in rhs(t,y)]),(0,.2),np.array([complex(x) for x in u0+v0]),method='DOP853',t_eval=grid,rtol=1e-12,atol=1e-14)
    analytic=np.array([[complex(x) for x in solution(t)] for t in grid]).T
    error=float(np.max(np.abs(analytic-numerical.y)))
    assert numerical.success and error<1e-10,error
    # Check the actual notebook callable; its explicit complex casts limit precision.
    z1=om[0];z0=z1-y0
    functions=[]
    for conj in (False,True):
        for j in range(4):functions.append(notebook_evaluator(z0,z1,aj[j]-y0,d2,d3,L0[j],L1[j],0,mp.log(alpha[j]),om,conj=conj))
    nb_error=max(abs(f(mp.mpf('.04'))-solution(mp.mpf('.04'))[j]) for j,f in enumerate(functions))
    assert nb_error<mp.mpf('1e-10'),nb_error
    # The historical formulas must fail for this test, rather than pass vacuously.
    wrong_linear=max(abs(2*d2*l/(3*d3)) for l in L1)
    wrong_paper=max(abs(2*l*(d2-6*we.wp(y0,om))/(3*d3)) for l in L1)
    assert wrong_linear>mp.mpf('.01') and wrong_paper>mp.mpf('.01')
    print(f'PASS b2={2*sign}: Lambda1={list(map(float,L1))}; equation residual={mp.nstr(residual,4)}; DOP853 error={error:.4g}; notebook error={mp.nstr(nb_error,4)}')
