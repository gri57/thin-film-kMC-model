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
N = 12
# Array with the number of atoms at every site
# start with a perfect surface
surfacemat = np.ones( (N,N), 'int' )

# Array with the number of neighbours of each atom at every site 
# start with a perfect surface
neighsmat = 5.*np.ones( (N,N), 'int' )

# Array that tells how many atoms in total there are with 1, 2, 3, 4 and 5 neighbours
neighstally = np.zeros( 5, 'int' )
neighstally[4] = N*N # perfect surface - all atoms have 5 neighbours

# add one atom to the surface so that there will be at some sites with 1 neighbour
surfacemat, neighsmat, neighstally = adsorption_event( N, 0, 3, surfacemat, neighsmat, neighstally )
surfacemat, neighsmat, neighstally = adsorption_event( N, 1, 2, surfacemat, neighsmat, neighstally )
surfacemat, neighsmat, neighstally = adsorption_event( N, 1, 4, surfacemat, neighsmat, neighstally )
surfacemat, neighsmat, neighstally = adsorption_event( N, 1, 6, surfacemat, neighsmat, neighstally )

# add two atoms so that there will be at least two atoms with 2 neighbours
surfacemat, neighsmat, neighstally = adsorption_event( N, 3, 4, surfacemat, neighsmat, neighstally )
surfacemat, neighsmat, neighstally = adsorption_event( N, 3, 5, surfacemat, neighsmat, neighstally )

# add some atoms so that there are at least two atoms with 3 neighbours
surfacemat, neighsmat, neighstally = adsorption_event( N, 7, 8, surfacemat, neighsmat, neighstally )
surfacemat, neighsmat, neighstally = adsorption_event( N, 8, 8, surfacemat, neighsmat, neighstally )
surfacemat, neighsmat, neighstally = adsorption_event( N, 7, 7, surfacemat, neighsmat, neighstally )

surfacemat, neighsmat, neighstally = adsorption_event( N, 10, 3, surfacemat, neighsmat, neighstally )
surfacemat, neighsmat, neighstally = adsorption_event( N, 10, 2, surfacemat, neighsmat, neighstally )
surfacemat, neighsmat, neighstally = adsorption_event( N, 11, 2, surfacemat, neighsmat, neighstally )

# remove an atom to have some atoms with 4 neighbours
surfacemat, neighsmat, neighstally = desorption_event( N, 10, 10, surfacemat, neighsmat, neighstally )

print surfacemat
print neighsmat
print neighstally

# Bulk mole fraction
X = 2e-6

# Coupling time for the PDE gas phase and KMC solid-on-solid models
dtcouple = 0.1
# initialize KMC timestep (updated by the runSolidOnSolidKMC() function)
dtkmc = 0.0
# Total simulation time
Ttot = 10.0
# Current simulation time
Tcurr = 0.0

# Output data
rough = np.zeros( int(Ttot/dtcouple), 'float' )
thick = np.zeros( int(Ttot/dtcouple), 'float' )
growr = 123.45*np.ones( int(Ttot/dtcouple), 'float' )

# Find the value of the 2nd derivative of the dependent variable in the Fluid Flow conservation equation at the first boundary
correct2ndderivative = fsolve( residualFluidFlowSS, 1.2 )
# Solve the Fluid Flow conservation equation at steady state (equation 3-1 of Shabnam's PhD thesis)
# get the values of the dependent variable and its first and second derivatives 
fvalues = calcFluidFlowSS( correct2ndderivative )

# Boundary condition for the mole fraction of the precursor (x) at infinite eta ( x(eta = inf) = X )
xinit = np.zeros( max(fvalues.shape), 'float' )
xinit[-1] = X

# Run initial calculations (gas is provided to the chamber, when it makes its way down to the substrate adsorption/desorption/migration begins)
Wa, Wd, Wm, xvalues = runGasPhasePDE( N, dtcouple*1e-6, Na, Nd, fvalues, xinit, neighstally )

# Index for roughness, growth rate and thickness tracking arrays
counter = 0

# Initialize the variable for storing "old" surface height information (for growth rate calculations)
surfacemat_prev = surfacemat.copy()

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
		
		# roughness
		# find out the difference between current location and row immediately below
		# current row and row immediately above yields the same result, hence multiplication by 2
		rough[counter] += 2.*np.sum(np.abs( surfacemat[1:,:] - surfacemat[0:-1,:] )) + 2.*np.sum(np.abs( surfacemat[0,:] - surfacemat[-1,:] ))
		# the difference between the current column and column immediately before is the same as between current and immediately after
		rough[counter] += 2.*np.sum(np.abs( surfacemat[:,1:] - surfacemat[:,0:-1] )) + 2.*np.sum(np.abs( surfacemat[:,-1] - surfacemat[:,0] ))
		rough[counter] /= (2.*N*N)
		rough[counter] += 1.
		
		# thickness
		thick[counter] = np.sum( surfacemat ) * np.power( N, -2. )
		
		# growth rate
		growr[counter] = np.sum( surfacemat - surfacemat_prev ) / np.power( N, 2. ) / dtcouple
		
		# store the matrix with heights at each site for the next growth rate calculation
		surfacemat_prev = surfacemat.copy()
		
		# update the index
		counter += 1
		
print "Done!"
print surfacemat
print neighsmat
print neighstally
print '\nRoughness:'
print rough
print '\nThicknesses:'
print thick
print '\nGrowth rates'
print growr
