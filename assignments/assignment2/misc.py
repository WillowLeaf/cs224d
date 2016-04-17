##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    r = math.sqrt(6.0 / (m + n))
    A0 = random.uniform(-r, r, (m, n))
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0
