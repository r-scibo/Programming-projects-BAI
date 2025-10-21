#%%
import numpy as np
import cvxpy as cp


#######
# DATA, do not change this part!
#######
a=[0.5, -0.5, 0.2, -0.7, 0.6, -0.2, 0.7, -0.5, 0.8, -0.4]
l=[40, 20, 40, 40, 20, 40, 30, 40, 30, 60]
Preq=np.arange(a[0],a[0]*(l[0]+0.5),a[0])
for i in range(1, len(l)):
    Preq=np.r_[ Preq, np.arange(Preq[-1]+a[i],Preq[-1]+a[i]*(l[i]+0.5),a[i]) ]

T = sum(l)

Peng_max = 20.0
Pmg_min = -6.0
Pmg_max = 6.0
eta = 0.1
gamma = 0.1
#####
# End of DATA part
#####

# Implement the following functions
# they should return a dictionary retval such that
# retval['Peng'] is a list of floats of length T such that retval['Peng'][t] = P_eng(t+1) for each t=0,...,T-1
# retval['Pmg'] is a list of floats of length T such that retval['Pmg'][t] = P_mg(t+1) for each t=0,...,T-1
# retval['Pbr'] is a list of floats of length T such that retval['Pbr'][t] = P_br(t+1) for each t=0,...,T-1
# retval['E'] is a list of floats of length T+1 such that retval['E'][t] = E(t+1) for each t=0,...,T

def car(Ebatt_max):
    Peng = cp.Variable(T)     
    Pmg  = cp.Variable(T)    
    Pbr  = cp.Variable(T)      
    E    = cp.Variable(T+1)    

    constraints = []

    # Zero‐battery special case
    if Ebatt_max == 0:
        constraints += [Pmg == 0, E == 0]
        

    for t in range(T):
        # power balance at time t
        constraints += [
            Preq[t] == Peng[t] + Pmg[t] - Pbr[t],
        ]

        # engine & motor bounds
        constraints += [
            Peng[t] >= 0,
            Peng[t] <= Peng_max,
            Pmg[t]  >= Pmg_min,
            Pmg[t]  <= Pmg_max,
            Pbr[t]  >= 0,
        ]

        # battery‐dynamics at t to t+1
        constraints += [
            E[t+1] <= E[t] - Pmg[t] - eta*cp.abs(Pmg[t])
        ]
        
        '''
        At optimality, there’s no reason to leave “slack" in 
        that inequality: if you could have pumped more 
        energy into the battery, you would always do so 
        until you hit one of the other binding constraints.
        Hence the solver will in fact push that inequality to 
        equality wherever it matters, recovering the original 
        dynamics exactly.
        '''

    # battery capacity
    constraints += [
        E >= 0,
        E <= Ebatt_max,
        E[T] == E[0]
    ]
    eps = 1e-4
    obj = cp.Minimize(
        cp.sum(Peng + gamma*cp.square(Peng))
    + eps*cp.sum(cp.pos(-Pmg)))
    
    prob = cp.Problem(obj, constraints)
    prob.solve()

    print("Status:", prob.status)
    print("Optimal objective:", prob.value)

    retval = {
        'Peng': [float(Peng.value[t]) for t in range(T)],
        'Pmg':  [float(Pmg.value[t])  for t in range(T)],
        'Pbr':  [float(Pbr.value[t])  for t in range(T)],
        'E':    [float(E.value[t])     for t in range(T+1)],
    }
    
    return retval


def car_with_battery():
    Ebatt_max = 100.0

    retval = car(Ebatt_max)

    return retval
    

def car_without_battery():
    Ebatt_max = 0
    
    retval = car(Ebatt_max)

    return retval




# %%
