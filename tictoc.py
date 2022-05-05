
def tic():
    """ Homemade version of matlab tic function
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    """
    
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    """ Homemade version of matlab toc function
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    """
    
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

