from sosmodel import *
from matplotlib import pyplot as plt
import time

def tic():
    """ Homemade version of matlab tic function
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    """
    
    #import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    """ Homemade version of matlab toc function
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    """
    
    #import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"
     

''' Create class instances (objects) '''

thinfilm = ThinFilm( 100 ) # the number of sites along an edge of the square film must be provided
gaslayer = GasLayer()

''' Calculate the dimensionless stream function '''

# Solve the Fluid Flow conservation equation at steady state (equation 3-1 of Shabnam's PhD thesis).
# fsolve will find the correct value of the 2nd derivative of the stream function at eta = 0 boundary
# in an iterative fashion. On every iteration of fsolve, the stream function values will be updated
# within calcFluidFlowSS and stored as self.f (or gaslayer.f for external access).

# The last iteration of fsolve will leave gaslayer.f in the most up-to-date state.
fsolve( gaslayer.calcFluidFlowSS, 1.2 ) # provide the initial guess for the 2nd derivative at eta = 0

''' Conduct the coupled KMC PDE simulation '''

coupling_time = 0.1
total_time = .5
current_time = 0.0

tic()

while current_time < total_time:

	if thinfilm.dtkmc < coupling_time:

		run_sos_KMC(thinfilm, gaslayer)

	else:

		calc_xgrow_PDE( thinfilm, gaslayer, coupling_time )
		current_time += coupling_time # @grigoriy - should coupling_time or thinfilm.dtkmc be added to current_time?
		thinfilm.dtkmc = 0. # reset KMC time
		thinfilm.Na = 0.
		thinfilm.Nd = 0.
		print 'Current simulation time:', current_time # progress report

toc()

