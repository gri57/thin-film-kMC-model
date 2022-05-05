from sosmodel import *

# create class instances
thinfilm = ThinFilm( 30 )
gaslayer = GasLayer()


''' Calculate the dimensionless stream function '''

# Find the value of the 2nd derivative of the dependent variable in the Fluid Flow conservation equation at the first boundary
correct2ndderivative = fsolve( gaslayer.calcFluidFlowSS, 1.2 )

# Solve the Fluid Flow conservation equation at steady state (equation 3-1 of Shabnam's PhD thesis)
# The stream function values will be updated within calcFluidFlowSS and stored as self.streamfun
_ = gaslayer.calcFluidFlowSS( correct2ndderivative )

calc_xgrow( thinfilm, gaslayer, 0.1 )
