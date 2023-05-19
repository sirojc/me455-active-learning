import numpy as np
import matplotlib.pyplot as plt
import plotting

# Plots LaTeX-Style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

### Belief ###

def update_post(post, meas, pmeas1, loc, gg):
    if meas == 1:
        pmeas = pmeas1
    else:
        pmeas = 1-pmeas1   
    
    # Define possible movements depending whether on border or not
    if loc[-1,0] == 0 and 1 <= loc[-1,1] <= gg-2:
        poss_u = np.array([[1, 0], [0, 1], [0, -1], [0, 0]])
    elif loc[-1,0] == gg-1 and 1 <= loc[-1,1] <= gg-2:
        poss_u = np.array([[-1, 0], [0, 1], [0, -1], [0, 0]])
    elif 1 <= loc[-1,0] <= gg-2 and loc[-1,1] == 0:
        poss_u = np.array([[1, 0], [-1, 0], [0, 1], [0, 0]])
    elif 1 <= loc[-1,0] <= gg-2 and loc[-1,1] == gg-1:
        poss_u = np.array([[1, 0], [-1, 0], [0, -1], [0, 0]])

    elif loc[-1,0] == 0 and loc[-1,1] == 0:
        poss_u = np.array([[1, 0], [0, 1], [0, 0]])
    elif loc[-1,0] == 0 and loc[-1,1] == gg-1:
        poss_u = np.array([[1, 0], [0, -1], [0, 0]])
    elif loc[-1,0] == gg-1 and loc[-1,1] == 0:
        poss_u = np.array([[-1, 0], [0, 1], [0, 0]])
    elif loc[-1,0] == gg-1 and loc[-1,1] == gg-1:
        poss_u = np.array([[-1, 0], [0, -1], [0, 0]])

    else:
        poss_u = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])

    for u in poss_u:
        pr0 = get_pr0(gg, loc, u[0], u[1])
        L = get_L(meas, u[0], u[1])
        post[int(loc[-1,0]+u[0]), int(loc[-1,1]+u[1])] = L * pr0 / pmeas

    post = post / np.sum(post) # normalize
    return post, poss_u

def get_L(meas, i, j):
    if meas == 1:
        if i == 1 or i == -1:
            L = 0.5
        else: # j == 1 or j == -1
            L = 0.01 
    else: # meas == 0 -- TODO: check if likelihoods of meas 1/0 are really the same in vicinity of loc
        if i == 1 or i == -1:
            L = 0.5
        else:
            L = 0.01
    return L

def get_pr0(gg, loc, i, j):
    #num_visit = len(np.where(loc == [loc[-1,0]+i, loc[-1,1]+j])[0]) #error in commented out part
    for k in range(len(loc)):
        if loc[k,0] == loc[-1,0]+i and loc[k,1] == loc[-1,1]+j:
            return 0
    return 1/(gg**2 - len(np.unique(loc, axis=0)))

### u ###

def get_u(post, prior, loc, S, possib_u, gg, meas_prob):
    min_u = np.array([0, 0])
    max_EdS = 0
    for u in possib_u:
        rj = loc[-1]+u
        sim_loc = np.vstack((loc, rj))
        pmeas1 = meas_prob[int(rj[0]),int(rj[1])]
        EdS = get_EdS(post, sim_loc, S, gg, pmeas1)
        #print("Entropy reduction in " + str(u) + " is equal to: " + str(EdS))
        if EdS > max_EdS:
            max_EdS = EdS
            min_u = u
    #print("Chose " + str(min_u))
    return min_u

### Entropy ###

def get_EdS(post, sim_loc, S, gg, pmeas1):
    # Get entropy for hypothetical measurement 1
    post1 = update_post(post, 1, pmeas1, sim_loc, gg)[0]
    S1 = get_S(gg, post1)
    # Get entropy for hypothetical measurement 0
    post0 = update_post(post, 0, pmeas1, sim_loc, gg)[0]
    S0 = get_S(gg, post0)

    return S -  pmeas1 * S1 - (1-pmeas1) * S0 # return the entropy reduction (--> the bigger the better)

def get_S(gg, post):
    S = 0
    for i in range(gg):
        for j in range(gg):
            if post[i,j] != 0:
                S -= post[i,j] * np.log(post[i,j])
    return S


### Main ###

def main():
    ### Initialize
    gg = 25
    for s in range(4):
        prior = np.zeros([gg,gg])
        for i in range(gg):
            for j in range(gg):
                prior[i,j] = 1/(gg**2) # uniform distribution

        S = np.log2(gg**2) # entropy of uniform distribution

        post = prior

        # Start Locations
        door = np.random.randint(3,gg-3,2)
        start = np.random.randint(0,gg,2)
        loc = np.array([start])
        #print(loc)

        # Measurement Ground Truth
        meas_prob = np.zeros([gg,gg])
        for i in range(gg):
            for j in range(gg):
                if (i == door[0] and j == door[1]):
                    meas_prob[i,j] = 1.0
                elif (i == door[0]-1 or i == door[0]+1 and door[1]-1 <= j <= door[1]+1):
                    meas_prob[i,j] = 1.0/2.0
                elif (i == door[0]-2 or i == door[0]+2 and door[1]-2 <= j <= door[1]+2):
                    meas_prob[i,j] = 1.0/3.0
                elif (i == door[0]-3 or i == door[0]+3 and door[1]-3 <= j <= door[1]+3):
                    meas_prob[i,j] = 1.0/4.0
                else:
                    meas_prob[i,j] = 0.01

        ### Search
        print(S)
        i = 0
        while S > 1 and i <1000:
            meas = (np.random.random() < meas_prob[int(loc[-1, 0]),int(loc[-1, 1])])
            post, poss_u = update_post(post, meas, meas_prob[int(loc[-1, 0]),int(loc[-1, 1])], loc, gg)
            S = get_S(gg, post)
            print(S)
            u = get_u(post, prior, loc, S, poss_u, gg, meas_prob) ### # of size (2,1): choose direction with highest expected entropy reduction
            loc = np.vstack((loc, loc[-1] + u))
            prior = post
            #print(loc[-1])
            i += 1

        plotting.plot_trajectory(gg, door, loc, s)
    


if __name__ == "__main__":
    main()

# sometimes Entropy still becomes NaN ?