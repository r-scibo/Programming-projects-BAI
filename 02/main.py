#%%
import numpy as np
import networkx as nx
# Here you may include additional libraries and define more auxiliary functions:

from pulp import *
from pulp import PULP_CBC_CMD

def find_top_ind(f1):

    top_ind = [0,0]

    found = False
    for i in range(10):
        for j in range(80):
            if f1[i,j] == 0:
                continue

            if j < 2:
                if f1[i,j+2] != 0:
                    top_ind[0] = i
                    top_ind[1] = j
                    found = True
                    break
            elif f1[i,j] != 0:
                top_ind[0] = i
                top_ind[1] = j
                found = True
                break

        if found == True:
            break

    return top_ind

def find_dim_(f_1, top_ind):

    dim = [0,0]

    for j in range(top_ind[0],10):
        if f_1[j,top_ind[1]] != 0:
            dim[0] += 1
    
    for i in range(top_ind[1],80):
        if f_1[top_ind[0],i] != 0:
            dim[1] += 1
            if i == 79:
                for k in range(5):
                    if f_1[top_ind[0],k] != 0:
                        dim[1] += 1
    
    return dim

def robust_read_file(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            # Remove trailing and leading whitespace
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Convert each character (if it is a digit) to an int
            row = [int(c) for c in line if c.isdigit()]
            data.append(row)
    # Optionally, verify all rows have the same length
    if not all(len(row) == len(data[0]) for row in data):
        raise ValueError("Not all rows have the same length")
    return np.array(data)


# This function should return the EMD distances between file1 and file2.
# EMD distance depends on your choice of distance between pixels and
# this will be taken into account during grading.
def comp_dist(file1, file2):
    # Write your code here:
    print(f"Last distance computed was: {file1} - {file2}")
    # Tranform the two given files in np arrays

    # with open(file1, 'r') as f:
        # f1 = np.array([list(map(int, line.strip()))
        #            for line in f if line.strip()])
    f1 = robust_read_file(file1)
    f2 = robust_read_file(file2)

    # with open(file2, 'r') as f:
    #     f2 = np.array([list(map(int, line.strip()))
    #                for line in f if line.strip()])

    # Reshape the files according to rotation and position of first file

    top_ind1 = find_top_ind(f1)
    top_ind2 = find_top_ind(f2)

    lead = 0
    adj_ind1 = (top_ind1[1]-35) % 80
    adj_ind2 = (top_ind2[1]-35) % 80
    if adj_ind2 < adj_ind1:
        lead = top_ind2[1]
    else:
        lead = top_ind1[1]

    f1_first = f1[:,lead:]
    f1_sec = f1[:,:lead]
    f_1 = np.concatenate((f1_first,f1_sec), axis=1)
    
    f2_first = f2[:,lead:]
    f2_sec = f2[:,:lead]
    f_2 = np.concatenate((f2_first,f2_sec), axis=1)

    # Divide by sum, making them prob distributions

    mult = 10**7

    f_1 = f_1 / f_1.sum()  * mult  # multiply to avoid floats for the solving algorithms
    f_1 = f_1.astype(int)
    demand1 = f_1.sum()
    f_2 = f_2 / f_2.sum()  * mult
    f_2 = f_2.astype(int)
    demand2 = f_2.sum()

    # Create directed graph and add vertices, calculating for each its demand

    top_ind_1 = find_top_ind(f_1)
    dim1 = find_dim_(f_1, top_ind_1)

    top_ind_2 = find_top_ind(f_2)
    dim2 = find_dim_(f_2, top_ind_2)

    f_2[top_ind_2[0], top_ind_2[1]] += demand1 - demand2

    DG = nx.DiGraph()

    for i in range(10):
        for j in range(80):
            if f_1[i,j] == 0:
                continue
            else:
                DG.add_node(f"1_{i}_{j}", demand = -f_1[i,j])
    
    for i in range(10):
        for j in range(80):
            if f_2[i,j] == 0:
                continue
            else:
                DG.add_node(f"2_{i}_{j}", demand = f_2[i,j])


    # Add edges, for each calculate the weight (euclidean distance) and set capacity to one

    for i in range(dim1[0]):
        for j in range(dim1[1]):

            for h in range(dim2[0]):
                for k in range(dim2[1]):

                    i1 = top_ind_1[0]+i
                    j1 = top_ind_1[1]+j
                    h2 = top_ind_2[0]+h
                    k2 = top_ind_2[1]+k
                    dist = abs(j1-k2) #round((abs(i1-h2)**2 + abs(j1-k2)**2 )** 0.5)
                    DG.add_edge(f"1_{i1}_{j1}", f"2_{h2}_{k2}", weight=dist, capacity=mult)


    # Use netwrok_simplex method from nx
    flowCost, flowDict = nx.network_simplex(DG)

    distance = flowCost/mult

    # Making connections with past frames very expensive
    adj_ind1 = (top_ind1[1]-35) % 80
    adj_ind2 = (top_ind2[1]-35) % 80
    if adj_ind2 < adj_ind1:
        distance = 200.0

    # And return the EMD distance, it should be float.
    return float(distance)

print(comp_dist('P1.txt','P2.txt'))


# This function should sort the files as described on blackboard.
# P1.txt should be the first one.
def sort_files():
    # If your code cannot handle all files, remove the problematic ones
    # from the following list and sort only those which you can handle.
    # Your code should never fail!
    files = ['P1.txt', 'P2.txt', 'P3.txt', 'P4.txt', 'P5.txt', 'P6.txt', 'P7.txt', 'P8.txt', 'P9.txt', 'P10.txt', 'P11.txt', 'P12.txt', 'P13.txt', 'P14.txt', 'P15.txt']
    # Write your code here:

    #Create LP problem
    prob = LpProblem("Ordering the frames problem", LpMinimize)

    # Create list of variables (pointers) and add it to problem

    pointers = []
    for i in range(1,16):
        for j in range(1,16):
            if i != j:
                pointers.append(f"p_{i}_{j}")

    pointer_vars = LpVariable.dicts("Point", pointers, 0,1, cat = LpBinary)

    # Auxiliary variables for MTZ subtour elimination
    u = {i: LpVariable(f"u_{i}", lowBound=2, upBound=15) for i in range(2, 16)}
    u[1] = LpVariable(f"u_1", lowBound=1, upBound=15)

    # Fix the starting node (e.g., node 1) to have order 1
    prob += u[1] == 1

    # Add obj and constraints
    dist = {}
    for i in range(1,16):
        for j in range(1,16):
            if i != j:
                dist[f"{i}_{j}"] = comp_dist(f"P{i}.txt",f"P{j}.txt")
    prob += (
        lpSum(pointer_vars[f"p_{i}_{j}"] * dist[f"{i}_{j}"] for i in range(1,16) for j in range(1,16) if i!=j), 
        "Total path distance"
    )

    # One outgoing pointer for each
    for i in range(1,16):
        prob += lpSum(pointer_vars[f"p_{i}_{j}"] for j in range(1,16) if i != j) == 1, f"Outgoing constr. on {i}"

    # One incoming pointer for each
    for j in range(1,16):
        prob += lpSum(pointer_vars[f"p_{i}_{j}"] for i in range(1,16) if i != j) == 1, f"Incoming constr. on {j}"

    # MTZ Subtour elimination constraints (only for nodes 2 through n)
    M = 15  
    for i in range(2, 16):
        for j in range(2, 16):
            if i != j:
                prob += u[i] - u[j] + M * pointer_vars[f"p_{i}_{j}"] <= M - 1, f"MTZ_{i}_{j}"

    # Create and solve the problem
    prob.writeLP("MinimumPath.lp")
    prob.solve(PULP_CBC_CMD(timeLimit=8))
    print("Status:", LpStatus[prob.status])

    # Extract relavant variables
    finals = []
    for v in prob.variables():
        if v.name.startswith("Point_p_") and v.varValue == 1.0:
            finals.append(v.name)

    fin_point = {s.split('_')[2]: s.split('_')[3] for s in finals}
    # Create sorted list
    sorted_im = ['1']
    for i in range(0,14):
        sorted_im.append(fin_point[sorted_im[i]])

    print(sorted_im)

    sorted_files = [f"P{i}.txt" for i in sorted_im]

    print(sorted_files)
    # should return sorted list of file names
    return sorted_files

sort_files()

# %%
