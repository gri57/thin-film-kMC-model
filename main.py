# 2015-Nov-16
# Script for the main program that will drive multiscale simulation


from functions import *
import numpy as np
from scipy.optimize import fsolve


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

# Calculate the values of dimensionless stream function versus eta
f_vs_eta_values = calcFluidFlowSS()

# Calculate Wa, Wd, Wm
Wa = Pa * np.power( N, 2. )
Wd = calcWd( neighstally )
Wm = calcWm( neighstally )

# Find the value of the 2nd derivative of the dependent variable in the Fluid Flow conservation equation at the first boundary
correct2ndderivative = fsolve( residualFluidFlowSS, 1.2 )

# Solve the Fluid Flow conservation equation at steady state (equation 3-1 of Shabnam's PhD thesis)
# get the values of the dependent variable and its first and second derivatives 
fvalues = calcFluidFlowSS( correct2ndderivative )

# Carry on with the simulation until the final time is reached
while Tcurr < Ttot:

	# Check the time, integrate PDE equations if coupling time has been reached
	if dtkmc < dtcouple:
		
		### microscale model (solid-on-solid) ###
		
		# calculate Wa, Wd, Wm
		Wa = Pa * np.power( N, 2. )
		Wd = calcWd( neighstally )
		Wm = calcWm( neighstally )
		
		# the update of dtkmc is done within the function
		dtkmc = runSolidOnSolidKMC( N, surfacemat, neighsmat, neighstally, Wa, Wd, Wm, dtkmc )
		
	else:
		
		# update the total simulation time
		Tcurr += dtkmc
		
		
		# run the PDE model to get the new boundary condition value and update xgrow for the KMC model
		runGasPhasePDE()



### Macroscale model (PDE model) ###


# Integrate the gas phase conservation equations


# Calculate boundary conditions

