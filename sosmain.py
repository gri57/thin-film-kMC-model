from sosmodel import *
from tictoc import * # import tic() and toc()

''' Create class instances (objects) '''

thinfilm = ThinFilm( 100 ) # the number of sites (N) along an edge of the square film must be provided
gaslayer = GasLayer()
observables = Observables( thinfilm.N, 0.1, 100.0 ) # provide N, the coupling time and the total time

''' Calculate the dimensionless stream function '''

# Solve the Fluid Flow conservation equation at steady state (equation 3-1 of Shabnam's PhD thesis).
# fsolve will find the correct value of the 2nd derivative of the stream function at eta = 0 boundary
# in an iterative fashion. On every iteration of fsolve, the stream function values will be updated
# within calcFluidFlowSS and stored as self.f (or gaslayer.f for external access).

# The last iteration of fsolve will leave gaslayer.f in the most up-to-date state.
fsolve( gaslayer.calcFluidFlowSS, 1.2 ) # provide the initial guess for the 2nd derivative at eta = 0

''' Conduct the coupled KMC PDE simulation '''

tic()

while observables.get_current_time() < (observables.total_time - observables.coupling_time):

	if thinfilm.dtkmc < observables.coupling_time:

		run_sos_KMC(thinfilm, gaslayer)

	else:
		
		calc_xgrow_PDE( thinfilm, gaslayer, observables )

		if gaslayer.xgrow < 0.0:
			raise ValueError('Precursor mole fraction on the thin film surface (xgrow) cannot be negative.')
		
		observables.update_current_time()
		observables.calculate_observables(thinfilm.surfacemat)
		
		# reset parameters
		thinfilm.dtkmc = 0. 
		thinfilm.Na = 0.
		thinfilm.Nd = 0.
		thinfilm.Nm = 0.
		
		# progress report
		print 'Current simulation time:', observables.current_time 
		''' @grigoriy - for some strange reason if at this point observables.current_time equals to 
		observables.total_time, Python will still think that current_time is less than total_time
		and will execute the while loop one more time. As a result, it was necessary to use 
		(observables.total_time - observables.coupling_time) in the while loop condition.
		'''


observables.roughness += 1. # fulfillment of equation 3-17 (see Observables.calculate_observables method documentation for details)

toc()

produce_output(observables)

