"""Final algebraic checks for the FWM manuscript.

Run: code/.venv/bin/python code/audit_final.py
Uses exact symbolic arithmetic; the numerical notebook checks are documented
separately in audits/final-review-audit.md.
"""
from itertools import combinations

import sympy as s

u=s.symbols('u1:5'); v=s.symbols('v1:5')
I=[u[j]*v[j] for j in range(4)]
U,V=s.prod(u),s.prod(v)
rho,lam,gam,D=s.symbols('rho lam gam D', nonzero=True)
b0,b1,b2=s.symbols('b0 b1 b2')
q=lambda t:b0+b1*t+b2*t*t

# Gauge: arbitrary symmetric phase-modulation matrix, on the conserved power line.
g=s.symbols('g1:4'); gs=(*g,-sum(g))
a=s.symbols('a1:5')
A=s.Matrix(4,4,lambda j,k:s.Symbol(f'a{min(j,k)}{max(j,k)}'))
powers=[gj-rho for gj in gs]
B1=-sum(a)-sum(A[j,k]*gs[k] for j in range(4) for k in range(4))
B2=sum(A)/2
phip=[]
for j in range(4):
    phip.append(a[j]-sum(a)/4-gs[j]*sum(A)/4
                +sum((A[j,k]-sum(A[l,k] for l in range(4))/4)*powers[k] for k in range(4)))
    actual=-a[j]-sum(A[j,k]*powers[k] for k in range(4))+phip[j]
    assert s.expand(actual-(B1-2*B2*powers[j])/4)==0
assert s.expand(sum(phip))==0
print('PASS: original-to-hat gauge, all four diagonal terms and zero total gauge')

# Root-based normalisation, tested for both coupling signs. The energy relation
# is Uhat+Vhat=q(rho), and rho'=Uhat-Vhat.
Uh,Vh=s.symbols('Uh Vh')
h=lam*D/(lam-rho)
c2=4*lam*(gam-lam)
J=h*(gam-rho)/c2
th=q(lam)/(2*(gam-lam))+b1/4+b2*gam/2+b2*lam/2-D/(8*q(lam))
assert s.factor(4*lam*J-lam*D/(gam-lam)-h)==0
assert s.factor(s.diff(J,rho)-D/(4*(lam-rho)**2))==0
for sign in (1,-1):
    chi=8*q(lam)/D
    Cprod=8*lam**2*q(lam)/sign
    barU,barV=h*h*Uh/Cprod,h*h*Vh/Cprod
    logu=(b1-2*b2*(gam-rho))/4+Vh/(gam-rho)+th+(Uh-Vh)/(2*(lam-rho))
    logv=-(b1-2*b2*(gam-rho))/4-Uh/(gam-rho)-th+(Uh-Vh)/(2*(lam-rho))
    expected_u=-1/chi+chi*J/4-sign*chi*barV/(4*J)
    expected_v=1/chi-chi*J/4+sign*chi*barU/(4*J)
    assert s.factor((logu-expected_u).subs(Uh,q(rho)-Vh))==0
    assert s.factor((logv-expected_v).subs(Vh,q(rho)-Uh))==0
    assert s.factor(s.diff(J,rho)*(Uh-Vh)-sign*chi*(barU-barV)/4)==0
# Sum theta vanishes using F'(lam)=2*q*q'+q^2*sum(1/(gamma-lam)).
R=s.symbols('R')
Droot=2*q(lam)*s.diff(q(lam),lam)+q(lam)**2*R
assert s.factor((q(lam)*R/2+b1+2*b2*lam-D/(2*q(lam))).subs(D,Droot))==0
print('PASS: full hat-to-bar equations, power sign, inverse h, and theta sum')

chi=s.symbols('chi',nonzero=True)
Ht=sum(t*t/8-t for t in I)-(U+V)/4
fu=[s.diff(Ht,vj) for vj in v];fv=[-s.diff(Ht,uj) for uj in u]
for j in range(4):
    assert s.expand(fu[j]+(1-I[j]/4)*u[j]+s.prod(v[k] for k in range(4) if k!=j)/4)==0
    assert s.expand(fu[j]*v[j]+u[j]*fv[j]-(U-V)/4)==0
for sign,root in [(1,s.S.One),(-1,(1+s.I)/s.sqrt(2))]:
    us=sign*root/chi;vs=sign/root/chi
    sub=dict(zip(u+v,tuple(us*t for t in u)+tuple(vs*t for t in v)))
    Hb=chi*sum(t*t for t in I)/8-sum(I)/chi-sign*chi*(U+V)/4
    assert s.simplify(s.expand(Hb.subs(sub,simultaneous=True)*chi**3-Ht))==0
    for j in range(4):
        assert s.simplify(s.expand(chi/us*s.diff(Hb,v[j]).subs(sub,simultaneous=True)-fu[j]))==0
        assert s.simplify(s.expand(-chi/vs*s.diff(Hb,u[j]).subs(sub,simultaneous=True)-fv[j]))==0
print('PASS: bar-to-tilde scaling, Hamiltonian scaling, all equations and power derivatives')

# Constant Wronskians and generated potentials, derived directly from the
# autonomous normal form before identifying the Lame equation.
S=sum(I)
gamma=[ij-S/4 for ij in I]
def along_normal_form(expr):
    return s.expand(sum(s.diff(expr,u[j])*fu[j]+s.diff(expr,v[j])*fv[j] for j in range(4)))
for j in range(4):
    cj=1+gamma[j]/2
    W=fu[j]*v[j]-u[j]*fv[j]-2*cj*I[j]
    conserved=Ht-sum(g*g for g in gamma)/8-gamma[j]**2/2-4*gamma[j]
    assert s.expand(W-conserved)==0
    assert along_normal_form(W)==0
    potential=4-Ht/4+sum(g*g for g in gamma)/16+gamma[j]-S/2
    assert s.expand(along_normal_form(fu[j])-2*cj*fu[j]+cj**2*u[j]-potential*u[j])==0
    assert s.expand(along_normal_form(fv[j])+2*cj*fv[j]+cj**2*v[j]-potential*v[j])==0
print('PASS: constant gauged Wronskians and generated potentials directly from normal-form equations')

# FS final interpolation step, including determinant factor and derivative sheet.
w,wp=s.symbols('w wp'); wj=s.symbols('w1:4'); yj=s.symbols('y1:4')
L=sum(yj[j]*s.prod((w-wj[k])/(wj[j]-wj[k]) for k in range(3) if k!=j) for j in range(3))
Delta=(wj[0]-wj[1])*(wj[0]-wj[2])*(wj[1]-wj[2])
M=s.Matrix([[1,w,wp,6*w*w]]+[[1,wj[j],-yj[j],6*wj[j]**2] for j in range(3)])
assert s.factor(M.det()+6*Delta*(wp+L))==0
K,aa=s.symbols('K aa',nonzero=True)
r=lam+K/(w-aa); divisor=-K/(w-aa)**2
assert s.factor(s.diff(r,w)*wp/divisor-wp)==0
assert s.degree(s.cancel(q(r)/divisor),w)==2
print('PASS: exact determinant/interpolant factor and quartic-to-cubic normalising divisor')

# Tilde spectral curve and both prescribed initial-data points: arbitrary invariants.
E,e2,e3,e4,p=s.symbols('E e2 e3 e4 p')
# Mean power r; gamma sum is zero, so sum I^2=-2e2+4r^2.
r=s.symbols('r')
P=r**4+e2*r*r+e3*r+e4  # product(gamma_j+r)
# product u + product v from H=E.
Q=2*r*r-e2-16*r-4*E
F=s.expand((Q*Q-4*P)/16)  # mean'= (U-V)/4, mean'=I_j'
aa=s.Rational(4,3)-E/12-e2/24
Fwp=s.Poly(s.expand(F.subs(r,aa-p)),p)
assert Fwp.coeff_monomial(p**3)==4
assert Fwp.coeff_monomial(p**2)==0
G2=-Fwp.coeff_monomial(p);G3=-Fwp.coeff_monomial(1)
expected2=E**2/12+E*(e2/12+s.Rational(16,3))+e2**2/48+2*e2/3-e3/4+s.Rational(64,3)
expected3=E**3/216+E**2*(e2/144-s.Rational(5,9))+E*(e2**2/288-2*e2/9-e3/48-s.Rational(64,9))+e2**3/1728-5*e2**2/144-e2*e3/96-8*e2/9+e3/3+e4/4-s.Rational(512,27)
assert s.expand(G2-expected2)==0
assert s.expand(G3-expected3)==0
print('PASS: tilde g2 and g3 from scalar reduction')

# Independent normal-form mode reconstruction: 60-digit evaluation of the printed
# sigma formula and its derivatives, including complex initial data.
from mpmath import mp
from weierstrass_modified import Weierstrass
mp.dps=60
we=Weierstrass()
worst=mp.mpf(0)
for shift in [mp.mpf(0),mp.mpc('.1','.07'),mp.mpc('-.08','.12')]:
    u0=[mp.mpc('.7','.1')+shift,mp.mpc('.9','-.2'),mp.mpc('1.1','.15'),mp.mpc('.8','.3')]
    v0=[mp.mpc('.6','-.12'),mp.mpc('1.2','.1')-shift,mp.mpc('.75','-.2'),mp.mpc('.95','.08')]
    Ip=[u0[j]*v0[j] for j in range(4)]; mean=sum(Ip)/4
    gp=[ij-mean for ij in Ip]
    ep={k:sum(mp.fprod(gp[j] for j in js) for js in combinations(range(4),k)) for k in [2,3,4]}
    En=sum(ij*ij/8-ij for ij in Ip)-(mp.fprod(u0)+mp.fprod(v0))/4
    sub={E:En,e2:ep[2],e3:ep[3],e4:ep[4]}
    gn=[mp.mpc(str(s.re(expr.subs(sub)).evalf(65)),str(s.im(expr.subs(sub)).evalf(65))) for expr in [G2,G3]]
    om=list(we.omega_from_g(*gn)[:2])
    if mp.im(om[1]/om[0])<0:om[1]=-om[1]
    # The historical half-period initializer uses machine-precision fractional
    # exponents. Refine its output here, without changing the helper.
    def exact_invariants(a,b):
        g2,g3=we.g_from_omega(a,b)
        return g2*(mp.mpf(4)/3)/mp.mpf(4/3),g3*(mp.mpf(8)/27)/mp.mpf(8/27)
    om=list(mp.findroot(lambda a,b: tuple(c-d for c,d in zip(exact_invariants(a,b),gn)),tuple(om),tol=mp.mpf("1e-55")))
    offset=mp.mpf(4)/3-En/12-ep[2]/24
    x0=we.invwp(offset-mean,om,(mp.fprod(u0)-mp.fprod(v0))/4)
    aj=[we.invwp(offset+gj,om,En+ep[2]/4-gj*gj/2-4*gj) for gj in gp]
    sig=lambda z:we.wsigma(z,om)
    Phi=lambda x,a:sig(x+a)/(sig(x)*sig(a))*mp.exp(-x*we.wzeta(a,om))
    coeff=[u0[j]/Phi(-x0,aj[j]) for j in range(4)]
    def sol(t):
        x=t-x0
        us=[coeff[j]*mp.exp((1+gp[j]/2)*t)*Phi(x,aj[j]) for j in range(4)]
        vs=[-mp.exp(-(1+gp[j]/2)*t)*Phi(x,-aj[j])/coeff[j] for j in range(4)]
        return us+vs
    assert max(abs(sol(0)[j]-(u0+v0)[j]) for j in range(8))<mp.mpf('1e-45')
    for t in [mp.mpf('0'),mp.mpf('.04'),mp.mpc('.08','.02')]:
        vals=sol(t);us,vs=vals[:4],vals[4:]
        expected=[-(1-us[j]*vs[j]/4)*us[j]-mp.fprod(vs[k] for k in range(4) if k!=j)/4 for j in range(4)]
        expected += [(1-us[j]*vs[j]/4)*vs[j]+mp.fprod(us[k] for k in range(4) if k!=j)/4 for j in range(4)]
        for j in range(8):
            err=abs(mp.diff(lambda z:sol(z)[j],t)-expected[j])
            worst=max(worst,err)
    assert worst<mp.mpf('1e-45'),mp.nstr(worst)
print('PASS: all eight normal-form sigma solutions and initial data, three complex cases; max residual',mp.nstr(worst,5))
