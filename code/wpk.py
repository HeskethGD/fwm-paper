from sympy import *
pw = Function('pw')
pwp = Function('pwp')
zw = Function('zw')
sigma = Function('sigma')
(z, g2, g3) = symbols('z, g2, g3')

def wpk(k: int) -> Equality:
    """
    Symbolic equation for kth order derivative of Weierstrass P function in terms of Weierstrass P and Weierstrass P Prime.
    Even orders: result is an O(k/2 + 1) polynomial in the Weierstrass P function.
    Odd orders: result is an O((k-1)/2) polynomial in the Weierstrass P function, multiplied by Weierstrass P Prime.
    
    Params
    - pw (Function): Weierstrass P function.
    - pwp (Function): First order derivative of Weierstrass P function (Weierstrass P Prime).
    - z (Symbol): Argument of function.
    - g2 (Symbol): Elliptic invariant.
    - g3 (Symbol): Elliptic invariant.
    
    Args
    - k (int): Order of derivative (k > 0).
    
    Returns
    - wpk_equation (Equality): Sympy equation for kth order derivative of Weierstrass P function.
    """
    
    if k < 1:
        print('k must be greater than 0')
        raise
        
    wp_1 = Eq(diff(pw(z,g2,g3),z), pwp(z,g2,g3))
    if k == 1:
        return wp_1
   
    wp_2 = Eq(diff(pw(z, g2, g3), (z, 2)), -g2/2 + 6*pw(z, g2, g3)**2)
    if k == 2: 
        return wp_2
    
    # Recursively calculate the higher order derivative equations and express them in pw and pwp
    wp_1_sqrd = Eq((diff(pw(z,g2,g3),z))**2, 4*pw(z,g2,g3)**3 - g2 * pw(z,g2,g3) - g3)
    wps = [wp_1_sqrd, wp_2]
    for j in range(2, k):
        wps.append(Eq(wps[j-1].lhs.diff(z), wps[j-1].rhs.diff(z).subs([eq.args for eq in wps])).expand())
    
    wpk_equation = Eq(wps[-1].lhs, wps[-1].rhs.subs(*wp_1.args).factor())
    
    return wpk_equation

def wzk(k: int) -> Equality:
    
    """
    Builds the symbolic equation for the nth_order derivative of the Weierstrass zeta function.
    Odd orders: result is an O((k-1)/2 + 1) polynomial in the Weierstrass P function.
    Even orders: result is an O(k/2 - 1) polynomial in the Weierstrass P function, multiplied by P Prime.
    
    Params
    - pw (Function): Weierstrass P function.
    - pwp (Function): First order derivative of Weierstrass P function (Weierstrass P Prime).
    - zw (Function): Weierstrass zeta function.
    - z (Symbol): Argument of function.
    - g2 (Symbol): Elliptic invariant.
    - g3 (Symbol): Elliptic invariant.
    
    Args:
    - k (int): Order of derivative (k > 0).
    
    Returns:
    - wzk_equation (Equality): Sympy equation for kth order derivative of Weierstrass zeta function.
    
    Params: g2, g3, Weierstrass elliptic invariants
    """
    if k < 1:
        print("k must not be less than 1")
        raise
        
    wz_k = Eq(diff(zw(z, g2, g3), (z, k)), - diff(pw(z, g2, g3), (z, k - 1)))
    if k == 1:
        return wz_k
    wzk_equation = wz_k.subs(*wpk(k - 1).args)
        
    return wzk_equation

def wsk_wz_uneval(_k: int) -> Equality:
    if _k < 1:
        print("k must not be less than 1")
        raise
    return Eq(diff(sigma(z, g2, g3), (z, _k)), diff(sigma(z, g2, g3)*zw(z, g2, g3), (z, _k - 1)).doit())

def wsk(k: int) -> Equality:
    
    """
    Builds the symbolic equation for the kth derivative of the Weierstrass sigma function.
    The result is a polynomial in the Weierstrass P, P Prime, zeta, and sigma functions.
    
    Params
    - pw (Function): Weierstrass P function.
    - pwp (Function): First order derivative of Weierstrass P function (Weierstrass P Prime).
    - zw (Function): Weierstrass zeta function.
    - sigma (Function): Weierstrass sigma function.
    - z (Symbol): Argument of function.
    - g2 (Symbol): Elliptic invariant.
    - g3 (Symbol): Elliptic invariant.
    
    Args:
    - k (int): Order of derivative (k > 0).
    
    Returns:
    - wsk_equation (Equality): Sympy equation for kth order derivative of Weierstrass sigma function.
    
    Params: g2, g3, Weierstrass elliptic invariants
    """
    
    if k < 1:
        print("k must not be less than 1")
        raise
    
    if k == 1:
        return wsk_wz_uneval(k)
        
    # Calculate all orders of diff sigma in terms of diff zw
    all_orders_dsigma_dzw = [wsk_wz_uneval(_n) for _n in range(1, k + 1)]
    
    # Recursively substitute lower orders
    wsk_equation = all_orders_dsigma_dzw[-1]
    for kth_dsigma_dzw in all_orders_dsigma_dzw[0:-1][::-1]:
        wsk_equation = Eq(wsk_equation.lhs, wsk_equation.rhs.subs(*kth_dsigma_dzw.args))
        
    # Substitute derivatives of Weierstrass zeta in terms of polynomials in Weierstrass P and P Prime
    all_orders_dzw_dpw_args = [wzk(_n).args for _n in range(1, k)][::-1]
    wsk_equation = wsk_equation.subs(all_orders_dzw_dpw_args).expand()
        
    return wsk_equation


def run_tests():
    """
    Checks that the first few orders return the expected equations
    """

    defined_wps = [
        Eq(diff(pw(z,g2,g3),z), pwp(z,g2,g3)),
        Eq(diff(pw(z, g2, g3), (z, 2)), -g2/2 + 6*pw(z, g2, g3)**2),
        Eq(diff(pw(z, g2, g3), (z, 3)), 12*pw(z, g2, g3)*pwp(z, g2, g3)),
        Eq(diff(pw(z, g2, g3), (z, 4)), -18*g2*pw(z, g2, g3) - 12*g3 + 120*pw(z, g2, g3)**3),
        Eq(diff(pw(z, g2, g3), (z, 5)), -18*g2*pwp(z, g2, g3) + 360*pw(z, g2, g3)**2*pwp(z, g2, g3)),
        Eq(diff(pw(z, g2, g3), (z, 6)), 9*g2**2 - 1008*g2*pw(z, g2, g3)**2 - 720*g3*pw(z, g2, g3) + 5040*pw(z, g2, g3)**4)
    ]
    defined_wss = [
        Eq(diff(sigma(z, g2, g3), z), sigma(z, g2, g3)*zw(z, g2, g3)),
        Eq(diff(sigma(z, g2, g3), (z, 2)), -pw(z, g2, g3)*sigma(z, g2, g3) + sigma(z, g2, g3)*zw(z, g2, g3)**2),
        Eq(diff(sigma(z, g2, g3), (z, 3)), 
           -3*pw(z, g2, g3)*sigma(z, g2, g3)*zw(z, g2, g3) - 
           pwp(z, g2, g3)*sigma(z, g2, g3) + sigma(z, g2, g3)*zw(z, g2, g3)**3)
    ]
    
    for k in range(len(defined_wps)):
        print(defined_wps[k] == wpk(k + 1).expand())
        
    for k in range(len(defined_wss)):
        print(defined_wss[k] == wsk(k + 1).expand())