# 2015-Nov-17
# File with the functions called by main.py


from contracts import contract
import numpy as np


@contract( N='int,>0', x='int,>=0,<N', y='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray,ndarray' )
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


@contract( N='int,>0', x='int,>=0,<N', y='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray,ndarray' )
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


@contract( N='int,>0', xi='int,>=0,<N', yi='int,>=0,<N', xf='int,>=0,<N', yf='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray,ndarray' )
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


@contract( N='int,>0', callerid='str', x='int,>=0,<N', y='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray' )
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


@contract( neighstally='ndarray', f='int,>=1,<=5', returns='float' )
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


@contract( neighstally='ndarray', returns='float' )
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


@contract( n='int,>=1,<=5', returns='float' )
def calcPd( n ):
	
	"""
	Calculate the individual rates of desorption events for each number 
	of neighbours for an atom (each atom can have from 1 to 5 neighbours).
	This function is equation 3-9 of Shabnam's PhD thesis.
	"""
	
	nu0 = kd0 * np.exp( -Ed / (R*T) )
	
	Pd = nu0 * np.exp( -n * E / (R*T) )
	
	return Pd


@contract( n='int,>=1,<=5', returns='float' )
def calcPm( n ):
	
	"""
	Calculate the individual rates of migration events for each number 
	of neighbours for an atom (each atom can have from 1 to 5 neighbours).
	This function is equation 3-11 of Shabnam's PhD thesis.
	"""
	
	A = (Ed - Em)/(R*T) # equation 3-12 of Shabnam's PhD thesis
	
	Pm = A * calcPd( n )
	
	return Pm

