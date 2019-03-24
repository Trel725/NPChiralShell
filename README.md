# NPChiralShell
Script for calculation optical properties of nanoparticles coated with optically active material  
Calculates optical properties of nanoparticle, coated with
optically active shell, according to *C.F. Bohren "Scattering of electromagnetic waves by an optically active spherical shell." The Journal of Chemical Physics 62.4 (1975): 1566-1571.*

## Function description
```python
calculate_opt_prop(core, shell, lmbd, chiral_coef, rad_core, rad_shell, surrounding=1)
```

**Parameters:**  

core - complex relative permittivity of NP core  
shell - complex relative permittivity of NP shell  
lmbd - wavelength, in m  
chiral coefficient - coefficient of chirality of NP, alpha + beta from eq. 3  
rad_core - NP core radius, in m  
rad_shell - NP shell radius, in m (outer radius)  
surrounding - complex relative permittivity of surrounding, vacuum by default (1)  
    
**Return**  
list: ext_l, ext_r, scat_left, scat_right, phi, theta  
  * ext - extinction cross-section  
  * scat - scattering cross-section  
  * phi - optical rotation, refer to eq. 50 for details  
  * theta - circular dichroism  

### Usage example:
``` python
lambdas = np.linspace(200, 1000, num=100) * 1e-9 # define wavelength range in meters
core_eps = 5 * np.ones(100) + 1j * np.ones(100) # constant permittivity for NP core
shell_eps = 5 * np.ones(100) + 0j  # constant permittivity for NP surrounding
chiral_coef = 1e-8 
rad_core = 10e-9  
rad_shell = 15e-9  
prop = calculate_opt_prop(core_eps, shell_eps, lambdas, chiral_coef, rad_core, rad_shell)  
ext_l, ext_r, scat_left, scat_right, phi, theta = prop  
```