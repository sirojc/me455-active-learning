import numpy as np
import matplotlib.pyplot as plt
import plotting

# Plots LaTeX-Style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def update_post(post, meas, pmeas1, loc, gg):
    if meas == 1:
        pmeas = pmeas1
    else:
        pmeas = 1-pmeas1   
    
    # Define possible movements depending whether on border or not
    if loc[-1,0] == 0 and 1 <= loc[-1,1] <= gg-2:
        update_loc = np.array([[1, 0], [0, 1], [0, -1]])
    elif loc[-1,0] == gg-1 and 1 <= loc[-1,1] <= gg-2:
        update_loc = np.array([[-1, 0], [0, 1], [0, -1]])
    elif 1 <= loc[-1,0] <= gg-2 and loc[-1,1] == 0:
        update_loc = np.array([[1, 0], [-1, 0], [0, 1]])
    elif 1 <= loc[-1,0] <= gg-2 and loc[-1,1] == gg-1:
        update_loc = np.array([[1, 0], [-1, 0], [0, -1]])

    elif loc[-1,0] == 0 and loc[-1,1] == 0:
        update_loc = np.array([[1, 0], [0, 1]])
    elif loc[-1,0] == 0 and loc[-1,1] == gg-1:
        update_loc = np.array([[1, 0], [0, -1]])
    elif loc[-1,0] == gg-1 and loc[-1,1] == 0:
        update_loc = np.array([[-1, 0], [0, 1]])
    elif loc[-1,0] == gg-1 and loc[-1,1] == gg-1:
        update_loc = np.array([[-1, 0], [0, -1]])

    else:
        update_loc = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    for mov in update_loc:
        pr0 = get_pr0(gg, loc, mov[0], mov[1])
        L = get_L(meas, mov[0], mov[1])
        post[int(loc[-1,0]+mov[0]), int(loc[-1,1]+mov[1])] = L * pr0 / pmeas

    return post, update_loc

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
    #num_visit = len(np.where(loc == [loc[-1,0]+i, loc[-1,1]+j])[0]) #TODO: error in this formulation
    for k in range(len(loc)):
        if loc[k,0] == loc[-1,0]+i and loc[k,1] == loc[-1,1]+j:
            return 0
    return 1/(gg**2 - len(np.unique(loc, axis=0)))

def get_u(post, prior, loc, S, possib_u):
    EdS = float('inf')
    min_u = np.array([0, 0])
    min_EdS = float('inf')
    for u in possib_u:
        EdS = get_EdS(post, prior, loc[-1]+u, S, loc)
        #print(EdS)
        if EdS < min_EdS:
            min_EdS = EdS
            min_u = u
    return min_u

def get_EdS(post, prior, rj, S, loc):
    EdS = prior[int(rj[0]), int(rj[1])]*(-S) + (1-prior[int(rj[0]), int(rj[1])])*get_EdS_cond(post, rj, loc)
    return EdS

def get_EdS_cond(post, rj, loc): #TODO
    EdS_cond = 0
    return EdS_cond

def get_S(gg, post):
    S = 0
    for i in range(gg):
        for j in range(gg):
            if post[i,j] != 0:
                S -= post[i,j] * np.log(post[i,j])
    return S


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
                    meas_prob[i,j] = 1
                elif (i == door[0]-1 or i == door[0]+1 and door[1]-1 <= j <= door[1]+1):
                    meas_prob[i,j] = 1/2
                elif (i == door[0]-2 or i == door[0]+2 and door[1]-2 <= j <= door[1]+2):
                    meas_prob[i,j] = 1/3
                elif (i == door[0]-3 or i == door[0]+3 and door[1]-3 <= j <= door[1]+3):
                    meas_prob[i,j] = 1/4
                else:
                    meas_prob[i,j] = 0.01

        ### Search
        print(S)
        while S != 0:
            meas = (np.random.random() < meas_prob[int(loc[-1, 0]),int(loc[-1, 1])])
            post, update_loc = update_post(post, meas, meas_prob[int(loc[-1, 0]),int(loc[-1, 1])], loc, gg)
            S = get_S(gg, post)
            #print(S)
            u = get_u(post, prior, loc, S, update_loc) ### # of size (2,1): choose direction with highest expected entropy reduction
            loc = np.vstack((loc, loc[-1] + u))
            prior = post
            print(loc[-1])

        plotting.plot_trajectory(gg, door, loc, s)
    


if __name__ == "__main__":
    main()


# S goes to infitinity: Look into values of posterior @ 4 update points