from sosmodel import *
from matplotlib import pyplot as plt

def tic():
    # Homemade version of matlab tic and toc 
    # http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"
     

# create class instances (objects)
thinfilm = ThinFilm( 30 ) # the number of sites along an edge of the square film must be provided
gaslayer = GasLayer()

#print gaslayer.xprofile
#print gaslayer.xgrow

''' Calculate the dimensionless stream function '''

# Solve the Fluid Flow conservation equation at steady state (equation 3-1 of Shabnam's PhD thesis).
# fsolve will find the correct value of the 2nd derivative of the stream function at eta = 0 boundary
# in an iterative fashion. On every iteration of fsolve, the stream function values will be updated
# within calcFluidFlowSS and stored as self.f (or gaslayer.f for external access); the last iteration 
# of fsolve will leave self.f in the most up-to-date state.

fsolve( gaslayer.calcFluidFlowSS, 1.2 ) # provide the initial guess for the 2nd derivative at eta = 0

calc_xgrow_PDE( thinfilm, gaslayer, 0.1 ) # provide the objects and the coupling time

#print gaslayer.xprofile
#print gaslayer.xgrow

#print thinfilm.surfacemat
#print thinfilm.neighsmat
#print thinfilm.neighstally
#print ''

#thinfilm.adsorption_event( 0,0 )

#print thinfilm.surfacemat
#print thinfilm.neighsmat
#print thinfilm.neighstally
#print ''


#print thinfilm.Wa

# find "i" value that matches to where zeta is on this discretized region
zetamod = 1.5

tic()
neighclass = 1
while zetamod > np.sum( np.array([1.,2.,3.,4.,5.])[0:neighclass] * np.array([1.,2.,3.,4.,5.])[0:neighclass] ):
	neighclass += 1
	# this "while" loop will stop as soon as zetamod <= np.sum(...), leaving us with the correct "neighclass" value
toc()

print neighclass
print ''

tic()
neighclass = 0
vari = 0.
while zetamod > vari:
	vari += np.array([1.,2.,3.,4.,5.])[neighclass] * np.array([1.,2.,3.,4.,5.])[neighclass] 
	neighclass += 1
	# this "while" loop will stop as soon as zetamod <= vari, leaving us with the correct "neighclass" value

toc()

print neighclass
print ''
