# etdrk4cp 
## An Adaptive 4th Order Exponential Time Differencing Runge-Kutta Scheme on GPUs using cupy (or CPUs using numpy).

Here is a barebones adaptive solver written in python that uses the Exponential Time Differencing Runge-Kutta scheme on GPUs using cupy arrays. It is based on [[Kassam and Lloyd 05](https://doi.org/10.1137/S1064827502410633)] as 
implemented in python by [@farenga](https://github.com/farenga/ETDRK4), with a simple adaptation algorithm inspired by [[Deka and Einkemmer 22](https://doi.org/10.1016/j.camwa.2022.07.011)].

The solver consists of three files, 

- [etdrk4cp.py](etdrk4cp.py) which contains the low level solver (similar to scipy.integrate.RK45 for example), which computes the exponential Runge Kutta coefficients. 
- [gsol.py](gsol.py) which is a slightly higher general level solver, which deals with the main time integration loop, and calling the specified list of user callbacks. Note that it currently lacks all kinds of usual protections (no max_step_size, no max_iter_number) and so it may fail in a number of ways, and is provided as is.
- [h5tools.py](h5tools.py), which consists of the save_data utility function that handles writing the data into a cleanly formatted hdf5 file.

The general idea is to write the right hand side of an ODE as a stiff/linear part (L) and a non-stiff/nonlinear part (rhsNL). Since this version has no matrix inversion capability, it can only deal with diagonal L, but the method is not too hard to generalize.

## Examples

### 1D Burger's Equation

The file [burg1d_ex.py](burg1d_ex.py) is an example that solves the one dimensional Burgers equation, where the nonlinear term is computed using a pseud-spectral method, with a large scale forcing, with a padded resolution of 262144, and a viscosity 1e-4. Large viscosity is necessary because Burger's equation generates shocks, and the pseudo-spectral method is not good at dealing with those. However the large resolution together with the high value of viscosity makes the problem extremely stiff. With these parameters, it would take forever to solve this with scipy.integrate.DOP853. And if we use a fixed time step, it must be smaller than dt=1e-5-1e-4. However the adaptive time step algorithm, when initialized with a small enough time step, manages to crunch through, with a reasonable looking solution with a tolerance of tol=1e-7. Running this example (up to t=100) took less than 45 minutes on my RTX 2000 Ada Generation Laptop GPU.

Note that running this file will basically generate a hdf5 file called "outburg.h5", below we show the velocity as a function of time (at the last 20 secs of the simulation) and the wave-number spectrum averaged over this same range.

![burg1d](https://github.com/gurcani/img/blob/main/burg1d.png)

We see the shocks as sharp caroonish edges on the time evolution plot on the left, whereas the familiar k**(-2) power law for the Burgers equation can be observed on the right. One can also see a small plateau near the peak of the spectrum, which corresponds to the forcing region.

### 2D Navier-Stokes Equation

The file [ns2d_forced.py](ns2d_forced.py) is another example that solves the two dimensional Navier-Stokes equation, this time with small scale random forcing (in a shell around k=100, with dtforce_update=0.1 and an amplitude A=10.0), in order to observe inverse cascase, again using the pseudo-spectral method, with regular viscosity to remove small scale enstrophy (nu=1e-4) and hypo-viscosity of the form nuL/k**4 with nuL=10.0.

![ns2d_spec](https://github.com/gurcani/img/blob/main/ns2d_spec.png)
