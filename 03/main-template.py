# Import packages. You can import additional packages, if you want
# You can change the way they are imported, e.g import pulp as pl or whatever
# But take care to adapt the solver configuration accordingly.
from pulp import *
import matplotlib.pyplot as plt
import numpy as np

# Use the following solver when submitting to gradescope
solver = COIN_CMD(path="/usr/local/lib/python3.10/dist-packages/pulp/solverdir/cbc/linux/64/cbc",threads=4)
# on your computer, you may need a different path, for example
#solver = COIN_CMD(path="/usr/bin/cbc",threads=16)
# at home, you can also try it with different solvers, e.g. GLPK, or with a different
# number of threads.
# WARNING: your code when run in gradescope should finish within 10 minutes!!!

def bakery():
    # Input file is called ./bakery.txt
    input_filename = './bakery.txt'

    # Use solver defined above as a parameter of .solve() method.
    # e.g., if your LpProblem is called prob, then run
    # prob.solve(solver) to solve it.
    data_np = np.loadtxt(input_filename)
    print(data_np)

    prob = LpProblem("Bakery Schedule", LpMinimize)

    retval = 1

    # Write visualization to the correct file:
    visualization_filename = './visualization.png'

    # retval should be a dictionary such that retval['s_i'] is the starting
    # time of pastry i
    return retval
