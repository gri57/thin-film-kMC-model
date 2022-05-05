# 2015-Nov-17
# File with the functions called by main.py


#from contracts import contract
import numpy as np
from scipy.integrate import odeint


#@contract( N='int,>0', x='int,>=0,<N', y='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray,ndarray' )
def adsorption_event( N, x, y, surfacemat, neighsmat, neighstally ):
	
	""" 
	Perform an adsorption event and call the function that does a 
	neighbour update. 
	"""
	
	# Add an atom to the surface at the specified site.
	surfacemat[x,y] += 1
	
	# Update the number of neighbours at each site affected by adsorption, use periodic boundaries.
	neighsmat, neighstally = update_neighs_pbc( N, 'ads', x, y, surfacemat, neighsmat, neighstally )
	
	return surfacemat, neighsmat, neighstally


#@contract( N='int,>0', x='int,>=0,<N', y='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray,ndarray' )
def desorption_event( N, x, y, surfacemat, neighsmat, neighstally ):
	
	""" 
	Perform a desorption event and call the function that does a 
	neighbour update. 
	"""
	
	# Remove an atom from the surface at the specified site.
	surfacemat[x,y] -= 1
	
	# Update the number of neighbours at each site affected by desorption, use periodic boundaries.
	neighsmat, neighstally = update_neighs_pbc( N, 'des', x, y, surfacemat, neighsmat, neighstally )
	
	return surfacemat, neighsmat, neighstally


#@contract( N='int,>0', xi='int,>=0,<N', yi='int,>=0,<N', xf='int,>=0,<N', yf='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray,ndarray' )
def migration_event( N, xi, yi, xf, yf, surfacemat, neighsmat, neighstally ):
	
	"""
	Perform a migration event (which is a desorption event followed by 
	adsorption). The functions for desorption and adsorption each call 
	the neighbour update function (with periodic boundary conditions).
	"""
	
	# Desorption event.
	surfacemat, neighsmat, neighstally = desorption_event( N, xi, yi, surfacemat, neighsmat, neighstally )
	
	# Adsorption event.
	surfacemat, neighsmat, neighstally = adsorption_event( N, xf, yf, surfacemat, neighsmat, neighstally )
	
	return surfacemat, neighsmat, neighstally


#@contract( N='int,>0', callerid='str', x='int,>=0,<N', y='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray' )
def update_neighs_pbc( N, callerid, x, y, surfacemat, neighsmat, neighstally ):
	
	""" 
	Do a local update of neighbours for atoms affected by adsorption, 
	desorption, and migration events. Use periodic boundary conditions.
	"""
	
	# Retrieve the current height at the site.
	currh = surfacemat[x,y] 
	
	# Using the callerid flag, find out the height before the action of 
	# the function that called update_neighs_pbc.
	if callerid == 'ads':
		prevh = currh - 1
	elif callerid == 'des':
		prevh = currh + 1
	
	# Find out the coordinates of the four sites that are neighbours to
	# the site of interest (x,y). Periodic boundary conditions apply.
	neighcoords = 0*np.ndarray( (4,2), 'int' )
	neighcoords[0,...] = [x-1, y]
	neighcoords[1,...] = [x+1, y]
	neighcoords[2,...] = [x, y-1]
	neighcoords[3,...] = [x, y+1]
	
	# Implement periodic boundary conditions.
	if x-1 < 0:
		neighcoords[0,0] += N
	if x+1 >= N: 
		# this coordinate should never be greater than N, could be equal to it at most
		neighcoords[1,0] -= N
	if y-1 < 0:
		neighcoords[2,1] += N
	if y+1 >= N:
		# this coordinate should never be greater than N, could be equal to it at most
		neighcoords[3,1] -= N
	
	# Update neighbour count at (x,y) site.
	siteneighs = 1
	for i in range( 4 ):
		# 4 is the number of nearest neighbours that may be surrounding the atom of interest.
		
		if currh <= surfacemat[ neighcoords[i,0], neighcoords[i,1] ]:
			# For any site, a neighbouring site contributes to neighbour count if 
			# height of the site is <= height of the neighbouring site.
			siteneighs += 1
			
		# Update neighbour count at those sites that were affected by the adsorption/desorption event.
		if surfacemat[ neighcoords[i,0], neighcoords[i,1] ] <= currh and surfacemat[ neighcoords[i,0], neighcoords[i,1] ] > prev:
			# current site now is but was not a neighbour of the other site 
			neighsmat[ neighcoords[i,0], neighcoords[i,1] ] += 1
		elif surfacemat[ neighcoords[i,0], neighcoords[i,1] ] > currh and surfacemat[ neighcoords[i,0], neighcoords[i,1] ] <= prev:
			# current site is not but was a neighbour of the other site 
			neighsmat[ neighcoords[i,0], neighcoords[i,1] ] -= 1
			
		#elif surfacemat[ neighcoords[i,0], neighcoords[i,1] ] <= currh and surfacemat[ neighcoords[i,0], neighcoords[i,1] ] <= prev:
			# current site was and is a neighbour of the other site 
		#elif surfacemat[ neighcoords[i,0], neighcoords[i,1] ] > currh and surfacemat[ neighcoords[i,0], neighcoords[i,1] ] > prev:
			# current site was not and is not a neighbour of the other site 
			
	# Update the number of neighbours at the site of interest (x,y).
	neighsmat[x,y] = siteneighs
	
	# Update the tally of neighbours
	neighstally[0] = np.sum( neighstally == 1 )
	neighstally[1] = np.sum( neighstally == 2 )
	neighstally[2] = np.sum( neighstally == 3 )
	neighstally[3] = np.sum( neighstally == 4 )
	neighstally[4] = np.sum( neighstally == 5 )
	
	
	return neighsmat, neighstally


#@contract( neighstally='ndarray', f='int,>=1,<=5', returns='float' )
def calcWm( neighstally, f = 5 ):
	
	"""
	Calculate the total migration rate. 
	
	If default value for f is used (f=5), then this function is equation 3-15 of Shabnam's PhD thesis.
	If another value for f is provided (f>=1,f<5), then this function is being used
	to select one of five classes for performing a migration event (see page 27 of 
	Shabnam's PhD thesis, the sentences immediately following equation 3-12).
	
	For efficiency purposes, it is better to use this function with the default f value.
	"""
	
	# preallocate the array for the storage of the values of the rates of migration events
	Pm = np.ndarray( f, 'float' )
	
	# populate the preallocated array with calculated rates of migration events
	for i in range(f):
		Pm[i] = calcPm( i+1 )
		
	# Calculate the total migration rate
	Wm = np.sum( neighstally[0:f]*Pm )
	
	return Wm


#@contract( neighstally='ndarray', returns='float' )
def calcWd( neighstally ):
	
	"""
	Calculate the total desorption rate. 
	"""
	
	# preallocate the array for storing the values of desorption event rates
	Pd = np.ndarray( 5, 'float' )
	
	# populate the preallocated array with the calculated rates of desorption events
	for i in range(f):
		Pd[i] = calcPd( i+1 )
		
	# Calculate the total desorption rate
	Wd = np.sum( neighstally*Pd )
	
	return Wd


#@contract( n='int,>=1,<=5', returns='float' )
def calcPd( n ):
	
	"""
	Calculate the individual rates of desorption events for each number 
	of neighbours for an atom (each atom can have from 1 to 5 neighbours).
	This function is equation 3-9 of Shabnam's PhD thesis.
	"""
	
	nu0 = kd0 * np.exp( -Ed / (R*T) )
	
	Pd = nu0 * np.exp( -n * E / (R*T) )
	
	return Pd


#@contract( n='int,>=1,<=5', returns='float' )
def calcPm( n ):
	
	"""
	Calculate the individual rates of migration events for each number 
	of neighbours for an atom (each atom can have from 1 to 5 neighbours).
	This function is equation 3-11 of Shabnam's PhD thesis.
	"""
	
	A = (Ed - Em)/(R*T) # equation 3-12 of Shabnam's PhD thesis
	
	Pm = A * calcPd( n )
	
	return Pm


#@contract( N='int,>0', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', Wa='float', Wd='float', Wm='float', dtkmc='float', returns='float' )
def runSolidOnSolidKMC( N, surfacemat, neighsmat, neighstally, dtkmc, Wa, Wd, Wm ):
	
	"""
	Microscale model - solid-on-solid with adsorption, migration and desorption.
	"""
	
	# Use Wa, Wd and Wm to choose between adsorption, desorption or migration (a/d/m)
	Wtotal_inv = np.power( Wa+Wd+Wm, -1. )

	# Get a random number on the half-open interval [1e-53,1.0) to decide which event to perform
	# 1e-53 is practically zero
	zeta = np.random.uniform( low=1e-53, high=1.0 )

	# In an if/elif/else structure, as soon as one of the conditions is true, the code under the 
	# condition is executed and all other conditions are skipped.

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


	# Increment the KMC timestep (equation 3-16 of Shabnam's PhD thesis)
	# 1e-53 is practically zero, but prevents np.log(sigma) from giving "-inf" for an answer
	sigma = np.random.uniform( low=1e-53, high=1.0 )
	dtkmc += -np.log( sigma ) * Wtotal_inv    # equation 3-16
	
	return dtkmc


#@contract( f_eta_2_0='float', returns='ndarray' )
def calcFluidFlowSS(f_eta_2_0):
	
	"""
	Solve the fluid flow conservation equation at steady state (equation 3-1 in Shabnam's PhD thesis).
	f_eta_2_0  -  the initial value of the second derivative at the first boundary.
	
	This page has been helpful: http://www.physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html
	"""
	
	# Parameter values
	rho_b = 1.0
	rho = 1.0
	params = [rho_b, rho] # pack the parameter values for the ODE solver
	
	# Initial values of the dependent variable and its derivatives
	f0 = 0.0 # known value of the dependent variable at the first boundary 
	f_eta_0 = 0.0 # known value of the first derivative at the first boundary
	
	fvars = [ f0, f_eta_0, f_eta_2_0 ] # pack the dependent variable values for the ODE solver
	
	# Make an array for the independent variable
	etaStop = 1.0
	etaInc = 0.01
	eta = np.arange( 0., etaStop, etaInc )
	
	# Use the ode solver imported from scipy.integrate
	fvalues = odeint( FluidFlowSS, fvars, eta, args=(params,)  )
	
	# Return the calculated values of the dependent variable and its first two derivatives
	return fvalues


#@contract( f_eta_2_0='float', returns='float' )
def residualFluidFlowSS(f_eta_2_0):
	
	"""
	Solve the fluid flow conservation equation at steady state (equation 3-1 in Shabnam's PhD thesis).
	There are three known boundary conditions (dependent variable and its first derivative are 
	known at the first boundary, and the first derivative is known at the second boundary).
	Since the second derivative at the first boundary is not known, it has to be solved for 
	numerically by casting this problem as a roots finding problem.
	
	f_eta_2_0  -  the guess at the initial value of the second derivative at the first boundary
	This page has been helpful: http://www.physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html
	"""
	
	# Parameter values
	rho_b = 1.0
	rho = 1.0
	params = [rho_b, rho] # pack the parameter values for the ODE solver
	
	# Initial values of the dependent variable and its derivatives
	f0 = 0.0 # known value of the dependent variable at the first boundary 
	f_eta_0 = 0.0 # known value of the first derivative at the first boundary
	
	fvars = [ f0, f_eta_0, f_eta_2_0 ] # pack the dependent variable values for the ODE solver
	
	# Make an array for the independent variable
	etaStop = 1.0
	etaInc = 0.01
	eta = np.arange( 0., etaStop, etaInc )
	
	# Use the ode solver imported from scipy.integrate
	fvalues = odeint( FluidFlowSS, fvars, eta, args=(params,)  )
	
	# Return the difference between the known value of the first derivative at the second boundary and its estimated value
	return 1.0 - fvalues[-1,1]


#@contract( fvars='list', eta='ndarray', params='list', returns='list' )
def FluidFlowSS( fvars, eta, params ):
	
	"""
	The fluid flow conservation equation at steady state - equation 3-1 
	of Shabnam's PhD thesis at steady state.
	This page has been helpful: http://www.physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html
	"""
	
	f, f_eta, f_eta_2 = fvars # unpack the dependent variable, and its first and second derivatives with respect to eta
	
	rho_b, rho = params # unpack the parameter values
	
	derivs = [ f_eta,
			   f_eta_2,
			   -f * f_eta_2 - 0.5 * ( rho_b/rho - f_eta**2. ) ]
	
	return derivs

