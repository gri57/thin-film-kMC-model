# 2015-Nov-16
# Script for the main program that will drive multiscale simulation

from functions import *
import numpy as np


# Number of sites along an edge of the square simple cubic lattice
# N^2 is the total number of atoms
N = 20

surfacemat = np.ndarray( (N,N), 'int' )
neighsmat = np.ndarray( (N,N), 'int' )
neighstally = np.ndarray( 5, 'int' )


### Macroscale model (PDE model) ###

# Integrate the conservation equations

# Calculate boundary conditions


### Microscale model (solid-on-solid) ###

# Variables - N of lattice sites, KMC time, coupling time

# Calculate Wa, Wd, Wm
Wa = Pa * np.power( N, 2. )
Wd = calcWd( neighstally )
Wm = calcWm( neighstally )

# Use Wa, Wd and Wm to choose between adsorption, desorption or migration (a/d/m)
Wtotal_inv = np.power( Wa+Wd+Wm, -1. )

# Get a random number on the half-open interval [1e-53,1.0) to decide which event to perform
# 1e-53 is practically zero
zeta = np.random.uniform( low=1e-53, high=1.0 )

# In an if/elif/else structure, as soon as one of the conditions is true, everything below is skipped.

if zeta < Wa*Wtotal_inv:
	# perform adsorption
	# choose a site randomly (pick a random index for the arrays, Python indexing starts at zero)
	x = np.random.random_integers(0,N-1)
	y = np.random.random_integers(0,N-1)
	surfacemat, neighsmat, neighstally = adsorption_event( N, x, y, surfacemat, neighsmat, neighstally )
	
elif zeta < (Wa+Wd)*Wtotal_inv:
	# perform desorption
	# choose a site randomly (pick a random index for the arrays, Python indexing starts at zero)
	x = np.random.random_integers(0,N-1)
	y = np.random.random_integers(0,N-1)
	surfacemat, neighsmat, neighstally = desorption_event( N, x, y, surfacemat, neighsmat, neighstally )
	
else:
	# perform migration
	# choose one of the 5 "classes" (atoms with 1 neighbour, 2, 3, 4 or 5)
	
	# it is already known that the value of zeta is between (Wa+Wd)/(Wa+Wd+Wm) and 1
	# Wm is calculated by summing Mi*Pm(i) products for i from 1 to 5 (as per equation 3-15 of Shabnam's PhD thesis)
	# zeta can be anywhere between (Wa+Wd)/(Wa+Wd+Wm(i=5)) and (Wa+Wd+Wm(i))/(Wa+Wd+Wm(i=5)), where 1<=i<=5 ("i" is an integer)
	
	# modify zeta to save on CPU time (less calculations); the expression below is: zeta*(Wa+Wd+Wm)-Wa-Wd
	zetamod = zeta * np.power( Wtotal_inv, -1. ) - Wa - Wd
	
	# find "i" value that matches to where zeta is on this discretized region
	Wm_vari = 0.0
	neighclass = 0 
	while zetamod > Wm_vari:
		neighclass += 1
		Wm_vari += calcPm( neighclass )
		# this "while" loop will stop as soon as zetamod <= Wm_vari, leaving us with the correct "neighclass" value
	del Wm_vari # housekeeping
	
	# find out the x and y indeces of atoms that have the specified number of neighbours
	indecesofatoms = np.where( neighsmat == neighclass )
	
	# find out how many atoms have the specified number of neighbours
	numofatoms = len( indecesofatoms[0] )
	
	# choose one of those atoms at random
	chosenatom = np.random.random_integers( 0, numofatoms-1 )
	
	# get the atom's x and y coordinates
	xi = indecesofatoms[0][chosenatom]
	yi = indecesofatoms[1][chosenatom]
	
	# randomly find the location xf,yf where the atom selected above will migrate to by adding/subtracting 1 to xi/yi
	chosenneighbour = np.random.random_integers( 0, 3 )
	if chosenneighbour == 0:
		xf = xi - 1
		yf = yi
		# use periodic boundary conditions
		if xf < 0:
			xf += N
	elif chosenneighbour == 1:
		xf = xi + 1
		yf = yi
		# use periodic boundary conditions
		if xf >= N:
			xf -= N
	elif chosenneighbour == 2:
		xf = xi
		yf = yi - 1
		# use periodic boundary conditions
		if yf < 0:
			yf += N
	elif chosenneighbour == 3:
		xf = xi
		yf = yi + 1
		# use periodic boundary conditions
		if yf >= N:
			yf -= N
	
	# Perform the migration event
	surfacemat, neighsmat, neighstally = migration_event( N, xi, yi, xf, yf, surfacemat, neighsmat, neighstally )


# Increment the KMC timestep
# 1e-53 is practically zero, but prevents np.log(sigma) from giving "-inf" for an answer
sigma = np.random.uniform( low=1e-53, high=1.0 )
dtkmc += -np.log( sigma ) * Wtotal_inv

# Check the time, integrate PDE equations if coupling time has been reached
dtkmc < dtcouple

