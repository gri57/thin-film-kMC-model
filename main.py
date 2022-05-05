# 2015-Nov-16
# Script for the main program that will drive multiscale simulation

from functions import *
import numpy as np

# Number of sites in the square simple cubic lattice
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
Wd = 0.0
for i in range(5):
	Wd += float(neighstally[i])*Pd(i+1) 
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
		# this while loop will not execute as soon as zetamod <= Wm_vari, leaving us with the correct neighclass value
	del Wm_vari # housekeeping
	
	#  form an array containing indeces (row and column numbers) of atoms that have "neighclass" neighbours
	neighsmat == neighclass
	
	#  randomly pick one of those atoms
	#  randomly find the location xf,yf where the atom selected above will migrate to by adding/subtracting 1 to xi/yi
	
	surfacemat, neighsmat, neighstally = migration_event( N, xi, yi, xf, yf, surfacemat, neighsmat, neighstally )


# Increment the KMC timestep
# 1e-53 is practically zero, but prevents np.log(sigma) from giving "-inf" for an answer
sigma = np.random.uniform( low=1e-53, high=1.0 )
dtkmc += -np.log( sigma ) * Wtotal_inv

# Check the time, integrate PDE equations if coupling time has been reached
dtkmc < dtcouple

