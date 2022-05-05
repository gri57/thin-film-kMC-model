from sosmodel import *
from tictoc import * # import tic() and toc()

tic()

''' Create class instances (objects). '''

thinfilm = ThinFilm(30) # the number of sites (N) along an edge of the square film must be provided
gaslayer = GasLayer()
observables = Observables(thinfilm.N, 0.1, 1.0) # provide N, the coupling time and the total time

# Optimization for calc_xgrow_PDE function - helps to avoid repeating the same calculations
gaslayer.eqn_3_20_denominator = np.power(2. * gaslayer.a * np.power(thinfilm.N, 2.) * observables.coupling_time, -1.)

''' Calculate the dimensionless stream function.

 Solve the Fluid Flow conservation equation at steady state (equation 3-1 of Shabnam's PhD thesis).
 fsolve will find the correct value of the 2nd derivative of the stream function at eta = 0 boundary
 in an iterative fashion. On every iteration of fsolve, the stream function values will be updated
 within calcFluidFlowSS and stored as self.f (or gaslayer.f for external access).'''

# The last iteration of fsolve will leave gaslayer.f in the most up-to-date state.
fsolve(gaslayer.calcFluidFlowSS, 1.2) # provide the initial guess for the 2nd derivative at eta = 0

# Optimization for MassTransfMoL function - helps to avoid repeating the same calculations
gaslayer.f_2_d_eta_inv = gaslayer.f[1:-1] * 2. * gaslayer.d_eta_inv

''' Conduct the coupled KMC PDE simulation. '''

counter = 1 # index for output arrays

while observables.current_time < observables.total_time_minus_1:

	if thinfilm.dtkmc < observables.coupling_time:

		# run solid-on-solid Kinetic Monte Carlo model
		run_sos_KMC(thinfilm, gaslayer)

	else:
		
		# update precursor mole fraction on the surface of the thin film
		calc_xgrow_PDE(thinfilm, gaslayer, observables)

		# update current time
		observables.current_time += observables.coupling_time
		
		# calculate observables (roughness, growth rate, thickness...)
		observables.calculate_observables(thinfilm.surfacemat, counter)
		
		# update the index for the output arrays
		counter += 1
		
		# reset parameters
		thinfilm.dtkmc = 0. 
		thinfilm.Na = 0.
		thinfilm.Nd = 0.
		thinfilm.Nm = 0.
		
		''' @grigoriy - for some strange reason if at this point observables.current_time equals to 
		observables.total_time, Python will still think that current_time is less than total_time. 
		As a result, it was necessary to use observables.total_time_minus_1 instead of observables.total_time.
		It is possible that memory locations for current_time and total_time are compared rather than
		the values themselves. '''


observables.roughness += 1. # fulfillment of equation 3-17 (see Observables.calculate_observables method documentation for details)

toc()

produce_output(observables)

