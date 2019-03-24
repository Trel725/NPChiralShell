from scipy.special import spherical_jn, spherical_yn
import numpy as np
from scipy.constants import mu_0, epsilon_0
np.seterr(all='raise')


def calculate_opt_prop(core, shell, lmbd, chiral_coef, rad_core, rad_shell, surrounding=1):
    """calculates optical properties of nanoparticle, coated with
    optically active shell, according to C.F. Bohren "Scattering of electromagnetic waves by an optically active spherical shell." 
    The Journal of Chemical Physics 62.4 (1975): 1566-1571.
    
    input: 
    core - complex relative permittivity of NP core
    shell - complex relative permittivity of NP shell
    lmbd - wavelength, in m
    chiral coefficient - coefficient of chirality of NP, alpha + beta from eq. 3
    rad_core - NP core radius, in m
    rad_shell - NP shell radius, in m (outer radius)
    surrounding - complex relative permittivity of surrounding, vacuum by default (1)
    
    returns:
    list: ext_l, ext_r, scat_left, scat_right, phi, theta
    ext - extinction cross-section
    scat - scattering cross-section
    phi - optical rotation, refer to eq. 50 for details
    theta - circular dichroism

    usage example:
    % lambdas = np.linspace(200, 1000, num=100)
    % core_eps = 5 * np.ones(100) + 1j * np.ones(100)
    % shell_eps = 5 * np.ones(100) + 0j
    % chiral_coef = 1e-8
    % rad_core = 10e-9
    % rad_shell = 15e-9
    % prop = calculate_opt_prop(core_eps, shell_eps, lambdas, chiral_coef, rad_core, rad_shell)
    % ext_l, ext_r, scat_left, scat_right, phi, theta = prop
    """

    # physical constants ans related values
    free_omega = 2 * np.pi * 299792458 / lmbd  # free space angular velocity
    # 299792458 - speed of light
    core_epsilon = np.complex_(core * epsilon_0)  # np core absolute permittivity
    core_mu = np.complex_((1) * mu_0)  # suppose unity relative permettivity
    shell_epsilon = np.complex_(shell * epsilon_0)  # same for shell
    shell_mu = np.complex_((1) * mu_0)

    # physical properties of system, including chirality
    # eq. 3, gamma+beta in equation = chiral coef
    k_l = np.sqrt(shell_epsilon * shell_mu) * free_omega * \
        (1 + chiral_coef * free_omega * np.sqrt(shell_epsilon * shell_mu))
    k_r = free_omega * np.sqrt(shell_epsilon * shell_mu) * \
        (1 - chiral_coef * free_omega * np.sqrt(shell_epsilon * shell_mu))

    # wavenumbers, i.e. free space omega * sqrt(epsilon*mu)
    k_i = np.complex_(free_omega * np.sqrt(core_epsilon * core_mu))  # np core
    k_hat = np.complex_(free_omega * np.sqrt(surrounding * epsilon_0 * mu_0))  # np surrounding, vacuum
    a_hat = np.complex_(rad_core)
    b_hat = np.complex_(rad_shell)

    # auxilary constants, eq. 31
    k_ii = np.complex_((k_l + k_r)) / 2
    nu = np.complex_(k_hat) * b_hat
    alpha = np.complex_(k_hat) * a_hat
    N_l = np.complex_(k_l) / k_hat
    N_r = np.complex_(k_r) / k_hat
    N_ii = np.complex_(k_ii) / k_hat
    N_i = np.complex_(k_i) / k_hat

    # zillion of auxilary functions...

    def z_1(kr, order):
        res = spherical_jn(order, kr)
        return res

    def z_2(kr, order):
        res = spherical_yn(order, kr)
        return res

    def z_3(kr, order):
        return spherical_jn(order, kr) + (1j * spherical_yn(order, kr))

    def eta_1(rho, order):  # rho^-1 * d[rho*z]/d[rho]
        res = spherical_jn(order, rho, derivative=True) + spherical_jn(order, rho) / rho
        return res

    def eta_2(rho, order):  # rho^-1 * d[rho*z]/d[rho]
        res = spherical_yn(order, rho, derivative=True) + spherical_yn(order, rho) / rho
        return res

    def eta_3(rho, order):  # rho^-1 * d[rho*z]/d[rho]
        res = eta_1(rho, order) + (1j * eta_2(rho, order))
        return res

    def F_n(N, order):
        res = N_ii * z_2(N * alpha, order) * eta_1(N_i * alpha, order) \
            - (N_i * eta_2(N * alpha, order) * z_1(N_i * alpha, order))
        return res

    def G_n(N, order):
        res = N_i * z_2(N * alpha, order) * eta_1(N_i * alpha, order) \
            - (N_ii * eta_2(N * alpha, order) * z_1(N_i * alpha, order))
        return res

    def H_n(N, order):
        res = N_ii * z_1(N * alpha, order) * eta_1(N_i * alpha, order) \
            - (N_i * eta_1(N * alpha, order) * z_1(N_i * alpha, order))
        return res

    def K_n(N, order):
        res = N_i * z_1(N * alpha, order) * eta_1(N_i * alpha, order) \
            - (N_ii * eta_1(N * alpha, order) * z_1(N_i * alpha, order))
        return res

    def D_n(order):
        res = F_n(N_l, order) * G_n(N_r, order) + G_n(N_l, order) * F_n(N_r, order)
        return res

    def D_1n(order):
        res = -np.reciprocal(D_n(order)) * \
            (G_n(N_r, order) * H_n(N_l, order) + F_n(N_r, order) * K_n(N_l, order))
        return res

    def D_2n(order):
        res = np.reciprocal(D_n(order)) * \
            (F_n(N_r, order) * K_n(N_r, order) - G_n(N_r, order) * H_n(N_r, order))
        return res

    def D_3n(order):
        res = np.reciprocal(D_n(order)) * \
            (G_n(N_l, order) * H_n(N_l, order) - F_n(N_l, order) * K_n(N_l, order))
        return res

    def D_4n(order):
        res = -np.reciprocal(D_n(order)) * \
            (G_n(N_l, order) * H_n(N_r, order) + F_n(N_l, order) * K_n(N_r, order))
        return res

    def X_rn(sign, order):
        res = z_1(N_r * nu, order) + (D_4n(order) * z_2(N_r * nu, order)) \
            + (sign * D_2n(order) * z_2(N_l * nu, order))
        return res

    def X_ln(sign, order):
        res = z_1(N_l * nu, order) + D_1n(order) * z_2(N_l * nu, order) \
            + (sign * D_3n(order) * z_2(N_r * nu, order))
        return res

    def U_rn(sign, order):
        res = eta_1(N_r * nu, order) + D_4n(order) * eta_2(N_r * nu, order) \
            + (sign * D_2n(order) * eta_2(N_l * nu, order))
        return res

    def U_ln(sign, order):
        res = eta_1(N_l * nu, order) + D_1n(order) * eta_2(N_l * nu, order) \
            + (sign * D_3n(order) * eta_2(N_r * nu, order))
        return res

    def A_rn(order):
        res = X_rn(-1, order) * eta_1(nu, order) - N_ii * U_rn(-1, order) * z_1(nu, order)
        return res

    def A_ln(order):
        res = X_ln(+1, order) * eta_1(nu, order) - N_ii * U_ln(+1, order) * z_1(nu, order)
        return res

    def W_ln(order):
        res = X_ln(-1, order) * N_ii * eta_3(nu, order) - U_ln(-1, order) * z_3(nu, order)
        return res

    def W_rn(order):
        res = X_rn(+1, order) * N_ii * eta_3(nu, order) - U_rn(+1, order) * z_3(nu, order)
        return res

    def V_ln(order):
        res = X_ln(+1, order) * eta_3(nu, order) - N_ii * U_ln(1, order) * z_3(nu, order)
        return res

    def V_rn(order):
        res = X_rn(-1, order) * eta_3(nu, order) - N_ii * U_rn(-1, order) * z_3(nu, order)
        return res

    def B_ln(order):
        res = X_ln(-1, order) * N_ii * eta_1(nu, order) - U_ln(-1, order) * z_1(nu, order)
        return res

    def B_rn(order):
        res = X_rn(+1, order) * N_ii * eta_1(nu, order) - U_rn(+1, order) * z_1(nu, order)
        return res

    def delta_n(order):
        res = W_ln(order) * V_rn(order) + W_rn(order) * V_ln(order)
        return res

    def dn(order):
        res = 1j * np.reciprocal(delta_n(order)) * (B_rn(order) * W_ln(order) - B_ln(order) * W_rn(order))
        return res

    def cn(order):
        res = 1j * np.reciprocal(delta_n(order)) * \
            (A_ln(order) * V_rn(order) - A_rn(order) * V_ln(order))
        return res

    def bn(order):
        res = -1 * np.reciprocal(delta_n(order)) * \
            (B_ln(order) * V_rn(order) + B_rn(order) * V_ln(order))
        return res

    def an(order):
        res = -1 * np.reciprocal(delta_n(order)) * \
            (A_rn(order) * W_ln(order) + A_ln(order) * W_rn(order))
        return res

    # end of auxilary functions

    def calculate_coefs(order):
        coefs = []
        for n in range(1, order + 1):
            coefs.append([an(n), bn(n), cn(n), dn(n)])
        return coefs

    def S_l(coefs):  # extinction coefficients, eq. 44
        summa = np.complex_(0)
        for n in range(1, len(coefs) + 1):
            a, b, c, d = coefs[n - 1]
            summa += (2 * n + 1) * (-a - b - (1j * c) + (1j * d))
        return summa

    def S_r(coefs):  # extinction coefficients, eq. 44
        summa = np.complex_(0)
        for n in range(1, len(coefs) + 1):
            a, b, c, d = coefs[n - 1]
            summa += (2 * n + 1) * (-a - b + (1j * c) - (1j * d))
        return summa

    def extinction(Sl, Sr):  # extinction cross_section for both polarizations
        ext_r = 2 * np.pi * np.reciprocal(np.power(k_hat, 2)) * np.real(Sr)
        ext_l = 2 * np.pi * np.reciprocal(np.power(k_hat, 2)) * np.real(Sl)
        return ext_r, ext_l

    def scat_l(coefs):  # scattering cross_section for left polarized light, eq. 42
        summa = np.complex_(0)
        for n in range(1, len(coefs) + 1):
            a, b, c, d = coefs[n - 1]
            summa += (2 * n + 1) * (np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2 + np.abs(d)**2 +
                                    1j * (np.conj(a) * c - a * np.conj(c) + b * np.conj(d) - np.conj(b) * d))
        return 2 * np.pi * (1 / (k_hat**2)) * summa

    def scat_r(coefs):  # scattering cross_section for right polarized light, eq. 42
        summa = np.complex_(0)
        for n in range(1, len(coefs) + 1):
            a, b, c, d = coefs[n - 1]
            summa += (2 * n + 1) * (np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2 + np.abs(d)**2 -
                                    1j * (np.conj(a) * c - a * np.conj(c) + b * np.conj(d) - np.conj(b) * d))

        return 2 * np.pi * (1 / (k_hat**2)) * summa

    def phi_theta(Sl, Sr):  # phi - optical rotation, theta - circular dichroism, eq. 50
        N = 1e15  # number of particles per unit volume
        h = 0.01  # slab thickness, 1 cm
        phi = np.imag(Sr - Sl) * np.pi * np.reciprocal(np.power(k_hat, 2)) * N * h
        theta = np.real(Sl - Sr) * np.pi * np.reciprocal(np.power(k_hat, 2)) * N * h
        return phi, theta

    # final calculations
    surrounding_index = 1
    x = 2 * np.pi * rad_core * surrounding_index / lmbd
    num_of_iter = int(np.max(np.floor(x + 4 * (x**(1 / 3)) + 4)))
    print("Going to perform {} iterations...".format(num_of_iter))
    coefs = calculate_coefs(num_of_iter)
    s_l, s_r = S_l(coefs), S_r(coefs)
    e_l, e_r = extinction(s_l, s_r)
    scat_left, scat_right = scat_l(coefs), scat_r(coefs)
    phi, theta = phi_theta(s_l, s_r)
    return e_l, e_r, scat_left, scat_right, phi, theta
