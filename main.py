# 2015-Nov-16
# Script for the main program that will drive multiscale simulation


from functions import *
import numpy as np


# Number of sites along an edge of the square simple cubic lattice
# 	N^2 is the total number of atoms
N = 20

# Array with the number of atoms at every site
surfacemat = np.ndarray( (N,N), 'int' )

# Array with the number of neighbours of each atom at every site
neighsmat = np.ndarray( (N,N), 'int' )

# Array that tells how many atoms in total there are with 1, 2, 3, 4 and 5 neighbours
neighstally = np.ndarray( 5, 'int' )

# Coupling time for the PDE gas phase and KMC solid-on-solid models
dtcouple = 0.1

# KMC timestep
dtkmc = 0.0

# Total simulation time
Ttot = 1.0

# Current simulation time
Tcurr = 0.0


while Tcurr < Ttot:

	# Calculate Wa, Wd, Wm
	Wa = Pa * np.power( N, 2. )
	Wd = calcWd( neighstally )
	Wm = calcWm( neighstally )

	# Check the time, integrate PDE equations if coupling time has been reached
	if dtkmc < dtcouple:
		
		### Microscale model (solid-on-solid) ###
		# update of dtkmc is done within the function
		dtkmc = runSolidOnSolidKMC( N, surfacemat, neighsmat, neighstally, Wa, Wd, Wm, dtkmc )
		
	else:
		
		# update the total simulation time
		Tcurr += dtkmc
		
		# run the PDE model to get the new boundary condition value and update xgrow for the KMC model
		runGasPhasePDE()



### Macroscale model (PDE model) ###


# Integrate the gas phase conservation equations


# Calculate boundary conditions

