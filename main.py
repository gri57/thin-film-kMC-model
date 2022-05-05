# 2015-Nov-16
# Script for the main program that will drive multiscale simulation

from functions import *
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint

# Variable to keep track of adsorbed atoms
Na = 0.
# Variable to keep track of desorbed atoms
Nd = 0.
# Number of sites along an edge of the square simple cubic lattice
# 	N^2 is the total number of atoms
N = 10
# Array with the number of atoms at every site
# start with a perfect surface
surfacemat = np.ones( (N,N), 'int' )
surfacemat[0][3] = 2
# Array with the number of neighbours of each atom at every site 
# start with a perfect surface
neighsmat = 5.*np.ones( (N,N), 'int' )
neighsmat[0][3] = 1
# Array that tells how many atoms in total there are with 1, 2, 3, 4 and 5 neighbours
neighstally = np.zeros( 5, 'int' )
neighstally[4] = N*N # perfect surface - all atoms have 5 neighbours
neighstally[4] = N*N-1
neighstally[0] = 1
# Bulk mole fraction
X = 2e-6

# Coupling time for the PDE gas phase and KMC solid-on-solid models
dtcouple = 0.1
# initialize KMC timestep (updated by the runSolidOnSolidKMC() function)
dtkmc = 0.0
# Total simulation time
Ttot = 1.0
# Current simulation time
Tcurr = 0.0

# Find the value of the 2nd derivative of the dependent variable in the Fluid Flow conservation equation at the first boundary
correct2ndderivative = fsolve( residualFluidFlowSS, 1.2 )
# Solve the Fluid Flow conservation equation at steady state (equation 3-1 of Shabnam's PhD thesis)
# get the values of the dependent variable and its first and second derivatives 
fvalues = calcFluidFlowSS( correct2ndderivative )

# Boundary condition for the mole fraction of the precursor (x) at infinite eta ( x(eta = inf) = X )
xinit = np.zeros( max(fvalues.shape), 'float' )
xinit[-1] = X

# Run initial calculations (gas is provided to the chamber, when it makes its way down to the substrate adsorption/desorption/migration begins)
Wa, Wd, Wm, xvalues = runGasPhasePDE( N, dtcouple, Na, Nd, fvalues, xinit, neighstally )

# Carry on with the simulation until the final time is reached
while Tcurr < Ttot:

	# Check the time, integrate PDE equations if coupling time has been reached
	if dtkmc < dtcouple:
		
		### microscale model (solid-on-solid) ###
		
		# the update of dtkmc, and the number of adsorbed and desorbed atoms, is done within the function
		surfacemat, neighsmat, neighstally, dtkmc, Na, Nd = runSolidOnSolidKMC( N, surfacemat, neighsmat, neighstally, dtkmc, Wa, Wd, Wm, Na, Nd )
		
	else:
		
		### macroscale model (PDE model) ###
		
		# update the total simulation time
		Tcurr += dtkmc
		
		# reset the KMC timestep
		dtkmc *= 0.0
		
		# run the PDE model to get the new boundary condition value and update xgrow for the KMC model
		Wa, Wd, Wm, xvalues = runGasPhasePDE( N, dtcouple, Na, Nd, fvalues, xvalues, neighstally )

		### Calculate roughness, growth rate and thickness ###
		
		
print "Done!"
