#from tictoc import * # import tic() and toc()
# paste the lines below into IPython
from sosmodel import *
%timeit -n2 -r3 run_simu(30,0.1,10.0) 