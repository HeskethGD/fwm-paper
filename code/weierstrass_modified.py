from mpmath import jtheta, pi, exp, sqrt, agm, qfrom, mpc, elliprf, im, zeta, polyroots, gamma, isinf


class Weierstrass:
    
    # This is a slightly modified version of the package developed by user stla on github
    # https://github.com/stla/pyweierstrass/blob/main/pyweierstrass/weierstrass.py
    
    """
    Useful resource: Apostol T.M. Modular functions and Dirichlet series in number theory,
    Section 2.8 Application to the Inversion of Eisenstein Series
    http://www.paris8.free.fr/Apostol%20T.M.%20Modular%20functions%20and%20Dirichlet%20series%20in%20number%20theory%20(Springer,1990)(600dpi)(T)(216s)_MT_.pdf
    """
    
    def __init__(self):
        pass  
    
    def kleinj(self, g2, g3):
        """
        Klein-j invariant function formed from elliptic invariants g2 and g3 (the one with the 1728 factor).

        Parameters
        ----------
        g2 : complex
            A complex number.
        g3 : complex
            A complex number.

        Returns
        -------
        complex
            The value of the klein-j invariant.
       
        https://en.wikipedia.org/wiki/J-invariant
        """
        j = 1728 / (1 - 27*g3**2/g2**3)
        return j

    def inverse_kleinj(self, _j):
        """
        Inverse of the klein-j invariant function (the one with the 1728 factor).

        Parameters
        ----------
        z : complex
            A complex number.

        Returns
        -------
        complex
            The value of the inverse of the klein-j invariant function at `z`.
            
        Note: See Inverse Function Method 1 here:
        https://en.wikipedia.org/wiki/J-invariant

        """
        isinf_condition = False
        try:
            # Note there is s type definition variation on j so the below does not always work. 
            # Revisit this later but for now we wrap in try/except
            isinf_condition = (isinf(_j.real) or isinf(_j.imag))
        except Exception:
            pass
            
        if isinf_condition:
            x = 0
        else:
            sqrt_arg = 3*(1728*_j**2 - _j**3)
            try:
                sqrt_arg = sqrt_arg.evalf()
            except Exception:
                pass
            t = ( -_j**3 + 2304*_j**2 - 884736*_j + 12288*sqrt(sqrt_arg) )**(1/3)
            x = 1/768*t + (1 - _j/768) - ( 1536*_j - _j**2 ) / (768*t)
            try:
                x = x.evalf()
            except Exception:
                pass
        lbd = (1 + sqrt(1 - 4*x)) / 2 # there are two possible choices here +/- sqrt
        tau = 1j * agm(1, sqrt(1-lbd)) / agm(1, sqrt(lbd))
        return tau

    def eisenstein_E4_E6(self, tau, E6_with_sqrt=False):
        """
        Eisenstein E-series' of weight 4 and 6.

        Parameters
        ----------
        tau : complex
            A complex number.

        Returns
        -------
        complex
            The value of the Eisenstein E4 and E6 series' at `tau`.
            
        Note: See Identities Involving Eisenstein Series here: 
        https://en.wikipedia.org/wiki/Eisenstein_series
        
        Note: See Eq1.7 here for a different way to express E6 and G6 in terms of theta functions
        that avoids the sqrt on the previous wikipedia link and appears to prevent some numerical errors in tests
        https://arxiv.org/pdf/1806.06725.pdf

        """
        q = qfrom(tau = tau)
        j2 = jtheta(2, 0, q)
        j3 = jtheta(3, 0, q)
        j4 = jtheta(4, 0, q)
        E4 = ( j2**8 + j3**8 + j4**8 ) / 2
        if E6_with_sqrt:
            E6 = 1/2 * sqrt(( j2**8 + j3**8 + j4**8 )**3 / 2 - 27*(j2*j3*j4)**8)
        else:
            E6 = (-3*j2**8 * ( j3**4 + j4**4 ) + ( j3**12 + j4**12 )) / 2  
        
        return E4, E6

    def eisenstein_G4_G6(self, tau):
        """
        Eisenstein G-series' of weight 4 an 6.

        Parameters
        ----------
        tau : complex
            A complex number.

        Returns
        -------
        complex
            The value of the Eisenstein G4 and G6 series at `tau`.
            
        Note: See Fourier Series section here to relate G to E: 
        https://en.wikipedia.org/wiki/Eisenstein_series
        2*zeta(4) = pi**4 / 45

        """
        E4, E6 = self.eisenstein_E4_E6(tau)
        G4 = 2 * zeta(4) * E4
        G6 = 2 * zeta(6) * E6
        return G4, G6 
    
    def tau_from_g(self, g2, g3):
        """
        Tau (half period ratio) from elliptic invariants

        Parameters
        ----------
        g2 : complex
            A complex number.
        g3 : complex
            A complex number.

        Returns
        -------
        tau : complex
            A complex number.
        """
        j = self.kleinj(g2, g3)
        try:
            j = j.evalf()
        except Exception:
            pass
        try:
            # Note there is s type definition variation on j so the below does not always work. 
            # Revisit this later but for now we wrap in try/except
            if (isinf(j.real) or isinf(j.imag)):
                return -1j*pi/2/sqrt(3), mpc("inf", "inf")
        except Exception:
            pass
        tau = self.inverse_kleinj(j)
        
        return tau
    
    # Weierstrass p-function
    def _wp_from_tau(self, z, tau):
        """
        Parameters
        ----------
        z : complex
            A complex number.
        tau : complex
            The half-period ratio

        Returns
        -------
        wp : complex
            The Weierstrass P function at z with hal-period ratio tau
            
        See Eq1.11 here https://arxiv.org/pdf/1806.06725.pdf
        """
        q = qfrom(tau = tau)
        j1z = jtheta(1, pi*z, q)
        j2 = jtheta(2, 0, q)
        j3 = jtheta(3, 0, q)
        j4z = jtheta(4, pi*z, q)
        wp = (pi * j2 * j3 * j4z / j1z)**2 - pi**2 * (j2**4 + j3**4) / 3
        return wp
    
    # Weierstrass p-function
    def wp(self, z, omega):
        """
        Weierstrass p-function.

        Parameters
        ----------
        z : complex
            A complex number.
        omega : complex pair
            A pair of complex numbers, the half-periods.

        Returns
        -------
        complex 
            The Weierstrass p-function evaluated at `z`.

        """
        omega1, omega2 = omega
        if im(omega2/omega1) <= 0:
            raise Exception("The imaginary part of the periods ratio is negative.")
        return self._wp_from_tau(z/omega1/2, omega2/omega1) / omega1**2 / 4
    
    def complex_sort(self, list_of_mpcs, ascending=True):
        """
        Sorts a list or array of complex numbers in mpc format first by the real part,
        and where the real part is equal, then by the imaginary part. 
        Uses pythons default approach for sorting a list of tuples.
        """
        sorted_list_of_complex_tuples = sorted([(c.real, c.imag) for c in list_of_mpcs], reverse=not ascending)
        sorted_list_of_mpcs = [mpc(real=t[0], imag=t[1]) for t in sorted_list_of_complex_tuples]
        return sorted_list_of_mpcs
    
    def sorted_roots_from_g2_g3(self, g2, g3):
        """
        Parameters
        ----------
        g2 : complex
            A complex number.
        g3 : complex
            A complex number.

        Returns
        -------
        The roots e1, e2, e3 of the equation 4*z**3 - g2*z - g3 = 0.
        The roots are sorted descending, first by the real part,
        and then where the real part is equal, by the imaginary part.
        For real roots: e1 > e2 > e3.
        
        Note: This work in C++ appears to define roots based on some relation to theta functions
        It is not yet clear why they do that, it may be motivated by order or speed.
        https://github.com/bluescarni/w_elliptic/blob/master/src/w_elliptic.hpp
        https://dlmf.nist.gov/23.6
        Using theta functions appears to be 10x faster than solving the cubic. 
        Recommend using roots_from_omega1_omega2 where possible.
            
        """
        e1, e2, e3 = self.complex_sort(polyroots([4, 0, -g2, -g3]), ascending=False)
        return e1, e2, e3
    
    def roots_from_omega1_omega2(self, omega1, omega2, sorting=True):
        """
        Parameters
        ----------
        g2 : complex
            A complex number.
        g3 : complex
            A complex number.

        Returns
        -------
        The roots e1, e2, e3 of the equation 4*z**3 - g2*z - g3 = 0.
        The roots are sorted descending, first by the real part,
        and then where the real part is equal, by the imaginary part.
        For real roots: e1 > e2 > e3.
        
        Note: This work in C++ appears to define roots based on their relation to theta functions
        This may be motivated by order or speed.
        https://github.com/bluescarni/w_elliptic/blob/master/src/w_elliptic.hpp
        https://dlmf.nist.gov/23.6
        
        Using theta functions appears to be 10x faster than solving the cubic:
        from mpmath import timing
        g2_num = mpc(real=f'{2*num_span*(random() - 0.5)}', imag=f'{2*num_span*(random() - 0.5)}')
        g3_num = mpc(real=f'{2*num_span*(random() - 0.5)}', imag=f'{2*num_span*(random() - 0.5)}')
        omega1, omega2, omega3 = wst.omega_from_g(g2_num, g3_num, tolerance)
        omega1, omega2, omega3
        if im(omega2/omega1) <= 0:
            omega2 = -omega2
        print(timing(wst.roots_from_omega1_omega2, omega1, omega2))
        print(timing(wst.sorted_roots_from_g2_g3, g2_num, g3_num))
            
        """
        tau = omega2/omega1
        q = qfrom(tau = tau)
        j24 = jtheta(2, 0, q)**4
        j44 = jtheta(4, 0, q)**4
        c = pi**2/omega1**2/12
        e1 = c * ( j24 + 2 * j44 )
        e2 = c * ( j24 - j44 )
        e3 = - c * ( 2 * j24 + j44 )
        if sorting:
            e1, e2, e3 = self.complex_sort([e1, e2, e3], ascending=False)
        return e1, e2, e3

    def omega_from_g(self, g2, g3, tolerance=1e-10):
        """
        Half-periods from elliptical invariants.

        Parameters
        ----------
        g2 : complex
            A complex number.
        g3 : complex
            A complex number.

        Returns
        -------
        complex tuple
            The three half-periods corresponding to the elliptical invariants g2 and g3.
            Defined as the points such that: wp(omega1) = e1, wp(omega2) = e2, wp(omega3) = e3.
            Where wp is the Weierstrass P function and e1, e2, e3 are the roots of
            4*z**3 - g2*z - g3*z = 0,
            sorted descending first by the real part, and where the real part is equal, 
            then by the imaginary part. For real roots: e1 > e2 > e3.
            Note that only two half-periods are linearly independent.
            
        Note: Series relationship for half periods: 
        https://functions.wolfram.com/EllipticFunctions/WeierstrassHalfPeriods/02/
        - The series for g2 alone only defines the periods up to a quartic root of unity which can cause errors 
          if the correct root is not chosen.
              g2 = 60/2**4*G4/omega1**4 => omega1 = (60/2**4*G4/g2)**(1/4) ... *1? *-1? *1i? *-1i?
        - The series for g3 can be used to decide whether to multiply the periods by the imaginary unit or not, 
          then the periods are defined up to a sign, i.e the ratio g2/g3 is quadratic in the periods. 
          As the half periods are only defined up to a sign because of periodicity anyway this suffices.
              g2(omega1,omega2) = 60/2**4*G4(omega1,omega2), g3(omega1,omega2) = 140/2**6*G6(omega1,omega2)
              G4(omega1,omega2) = 1/omega1**4 * G4(1,omega2/omega1) = 1/omega1**4 * G4
              G6(omega1,omega2) = 1/omega1**6 * G6(1,omega2/omega1) = 1/omega1**6 * G6
              g2/g3 = 60/140 * 2**2 * omega1**2 * G4/G6
              omega1 = +/- sqrt(g2/g3 * G6/G4 * 7/12)
        - 

        """
        if g2 == 0:
            omegaA = g3 ** (-1/6) * gamma(1/3) ** 3 / 4 / pi
            tau = mpc(0.5, sqrt(3)/2)
        elif g3 == 0:
            tau = self.tau_from_g(g2, g3)
            G4, G6 = self.eisenstein_G4_G6(tau) # G4(1,tau), G6(1, tau)
            omegaA = 1j * (15 / 4 / g2 * G4) ** (1/4)
        else:
            tau = self.tau_from_g(g2, g3)
            G4, G6 = self.eisenstein_G4_G6(tau) # G4(1,tau), G6(1, tau)
            _sqrt_arg = g2/g3 * G6/G4 * 7/12
            try:
                _sqrt_arg = _sqrt_arg.evalf()
            except Exception:
                pass
            omegaA = sqrt(_sqrt_arg)
            
        omegaB = tau * omegaA
        omegaC = omegaA + omegaB
        
        # Assign omega1, omega2, omega3 labels to match e1, e2, e3 using mean absolute error comparrison
        omegas = [omegaA, omegaB, omegaC]
        wps = [self.wp(omegaN, (omegaA, omegaB)) for omegaN in omegas]
        index_combos = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
        e1, e2, e3 = self.roots_from_omega1_omega2(omegaA, omegaB)
        maes = [(abs(e1 - wps[ic[0]]) + abs(e2 - wps[ic[1]]) + abs(e3 - wps[ic[2]])) / 3 for ic in index_combos]
        mae = min(maes)
        min_index= maes.index(mae)
        if mae > tolerance:
            raise Exception(
                f"""
                Mean absolute error {mae} of roots minus Weierstrass P at 
                half periods is greater than tolerance {tolerance}.
                e1={e1}, e2={e2}, e3={e3}, wp(omega1)={wps[index_combos[min_index][0]]}, 
                wp(omega2)={wps[index_combos[min_index][1]]},
                wp(omega3)={wps[index_combos[min_index][2]]}, tau={tau}
                """
            )
        omega1, omega2, omega3 = [omegas[k] for k in index_combos[min_index]]
        
        # Define omega such that im(tau) has the right sign
        if im(omega2/omega1) <= 0:
            omega2 = -omega2
        
        return omega1, omega2, omega3

    def g_from_omega(self, omega1, omega2):
        """
        Elliptical invariants from half periods.

        Parameters
        ----------
        omega1 : complex
            A complex number.
        omega2 : complex
            A complex number.

        Returns
        -------
        complex pair
            The elliptical invariants `g2` and `g3`.

        """
        tau = omega2/omega1
        q = qfrom(tau = tau)
        j2 = jtheta(2, 0, q)
        j3 = jtheta(3, 0, q)
        g2 = 4/3 * (pi/2/omega1)**4 * (j2**8 - (j2*j3)**4 + j3**8)
        g3 = (8/27 * (pi/2/omega1)**6 * 
              (j2**12 - (3/2 * j2**8 * j3**4 + 3/2 * j2**4 * j3**8) + j3**12))
        return g2, g3

    def _wpprime_from_omega1_and_tau(self, z, omega1, tau):
        """
        https://functions.wolfram.com/EllipticFunctions/WeierstrassPPrime/27/02/03/
        """
        q = qfrom(tau = tau)
        z1 = pi * z / (2 * omega1)
        j10p = jtheta(1, 0, q, 1)
        j20 = jtheta(2, 0, q)
        j30 = jtheta(3, 0, q)
        j40 = jtheta(4, 0, q)
        k0 = j10p ** 3 / (j20 * j30 * j40)
        j1z1 = jtheta(1, z1, q)
        j2z1 = jtheta(2, z1, q)
        j3z1 = jtheta(3, z1, q)
        j4z1 = jtheta(4, z1, q)
        kz = j2z1 * j3z1 * j4z1 / j1z1 ** 3
        wp_prime = - pi ** 3 /( 4 * omega1 ** 3) * k0 * kz 
        return wp_prime

    # Weierstrass p-function prime
    def wpprime(self, z, omega):
        """
        Derivative of Weierstrass p-function.

        Parameters
        ----------
        z : complex
            A complex number.
        omega : complex pair
            A pair of complex numbers, the half-periods.

        Returns
        -------
        complex 
            The derivative of the Weierstrass p-function evaluated at `z`.

        """
        omega1, omega2 = omega
        if im(omega2/omega1) <= 0:
            raise Exception("The imaginary part of the periods ratio is negative.")
        return self._wpprime_from_omega1_and_tau(z, omega1, omega2 / omega1)

    # Weierstrass sigma function
    def wsigma(self, z, omega):
        """
        Weierstrass sigma function.

        Parameters
        ----------
        z : complex
            A complex number.
        omega : complex pair
            A pair of complex numbers, the half-periods.

        Returns
        -------
        complex 
            The Weierstrass sigma function evaluated at `z`.
            
        Note: The formula relating sigma to theta functions can be found here:
        https://mathworld.wolfram.com/WeierstrassSigmaFunction.html
        https://functions.wolfram.com/EllipticFunctions/WeierstrassSigma/27/01/02/0002/

        """
        omega1, omega2 = omega
        z1 = pi * z / (2 * omega1)
        tau = omega2 / omega1
        if im(tau) <= 0:
            raise Exception("The imaginary part of the periods ratio is negative.")
        q = qfrom(tau = tau)
        j10p = jtheta(1, 0, q, 1)
        j10ppp = jtheta(1, 0, q, 3)
        j1z1 = jtheta(1, z1, q)
        w_sigma = 2 * omega1 / (pi * j10p) * exp( - z1 ** 2 * j10ppp / (6 * j10p) ) * j1z1
        # eta1 = - pi ** 2 * j10ppp / ( 12 * omega1 * j10p)
        # w_sigma = 2 * omega1 / (pi * j10p) * exp( eta1 * z ** 2 / (2 * omega1)) * j1z1
        return w_sigma
    
    def _mpc_to_float(self, mpc_val):
        return float(mpc_val.real) + float(mpc_val.imag)*1j
    
    def _sympy_add_to_float(self, z):
        try:
            z_real, z_imag = z.as_real_imag()
            return float(z_real) + float(z_imag)*1j
        except Exception:
            return z
    

    # Weierstrass zeta function
    def wzeta(self, z, omega):
        """
        Weierstrass zeta function.

        Parameters
        ----------
        z : complex
            A complex number.
        omega : complex pair
            A pair of complex numbers, the half-periods.

        Returns
        -------
        complex 
            The Weierstrass zeta function evaluated at `z`.

        """
        omega1, omega2 = omega
        if im(omega2/omega1) <= 0:
            raise Exception("The imaginary part of the periods ratio is negative.")
        w1 = - omega1 / pi
        tau = omega2 / omega1
        q = qfrom(tau = tau)
        p = 1 / 2 / w1
        eta1 = p / 6 / w1 * jtheta(1, 0, q, 3) / jtheta(1, 0, q, 1)
        return -eta1 * z + p * jtheta(1, p*z, q, 1) / jtheta(1, p*z, q)

    # Inverse Weierstrass p-function
    def invwp(self, w, omega, w_prime=None):
        """
        Inverse Weierstrass p-function.

        Parameters
        ----------
        w : complex
            A complex number.
        omega : complex pair
            A pair of complex numbers, the half-periods.

        Returns
        -------
        z : complex 
            The value z of the inverse Weierstrass p-function evaluated at w.
            If the value of the derivative Weierstrass p prime evaluated at z is specified
            an attempt to pick the right sign for z is made. 
            
        Note: See Eq 1.31 here https://arxiv.org/pdf/1806.06725.pdf

        """
        omega1, omega2 = omega
        if im(omega2/omega1) <= 0:
            raise Exception("The imaginary part of the periods ratio is negative.")
#         e1 = self.wp(omega1, omega)
#         e2 = self.wp(omega2, omega)
#         e3 = self.wp(-omega1-omega2, omega)
        e1, e2, e3 = self.roots_from_omega1_omega2(omega1, omega2, sorting=False)
        z = elliprf(w-e1, w-e2, w-e3)
        if w_prime:
            if abs(self.wpprime(-z, omega) - w_prime) < abs(self.wpprime(z, omega) - w_prime):
                return -z
        return z
    
    # Weierstrass sigma function
    def wsigma_from_g2_g3(self, z, g2, g3):
        """
        Weierstrass sigma function.

        Parameters
        ----------
        z : complex
            A complex number.
        g2 : complex
            A complex number.
        g3 : complex
            A complex number.

        Returns
        -------
        complex 
            The Weierstrass sigma function evaluated at `z`.
        """
        
        z = self._sympy_add_to_float(z)
        g2 = self._sympy_add_to_float(g2)
        g3 = self._sympy_add_to_float(g3)
        omega = self.omega_from_g(g2, g3)
        return self.wsigma(z, (omega[0], omega[1]))
    
    # Weierstrass p-function
    def wp_from_g2_g3(self, z, g2, g3):
        """
        Weierstrass p-function.

        Parameters
        ----------
        z : complex
            A complex number.
        g2 : complex
            A complex number.
        g3 : complex
            A complex number.

        Returns
        -------
        complex 
            The Weierstrass p-function evaluated at `z`.

        """
        z = self._sympy_add_to_float(z)
        g2 = self._sympy_add_to_float(g2)
        g3 = self._sympy_add_to_float(g3)
        omega = self.omega_from_g(g2, g3)
        return self.wp(z, (omega[0], omega[1]))
    
    # Weierstrass p-function prime
    def wpprime_from_g2_g3(self, z, g2, g3):
        """
        Derivative of Weierstrass p-function.

        Parameters
        ----------
        z : complex
            A complex number.
        g2 : complex
            A complex number.
        g3 : complex
            A complex number.

        Returns
        -------
        complex 
            The derivative of the Weierstrass p-function evaluated at `z`.

        """
        z = self._sympy_add_to_float(z)
        g2 = self._sympy_add_to_float(g2)
        g3 = self._sympy_add_to_float(g3)
        omega = self.omega_from_g(g2, g3)
        return self.wpprime(z, (omega[0], omega[1]))
        
    # Weierstrass zeta function
    def wzeta_from_g2_g3(self, z, g2, g3):
        """
        Weierstrass zeta function.

        Parameters
        ----------
        z : complex
            A complex number.
        g2 : complex
            A complex number.
        g3 : complex
            A complex number.

        Returns
        -------
        complex 
            The Weierstrass zeta function evaluated at `z`.

        """
        z = self._sympy_add_to_float(z)
        g2 = self._sympy_add_to_float(g2)
        g3 = self._sympy_add_to_float(g3)
        omega = self.omega_from_g(g2, g3)
        return self.wzeta(z, (omega[0], omega[1]))