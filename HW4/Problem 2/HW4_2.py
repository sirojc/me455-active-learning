import numpy as np
import matplotlib.pyplot as plt
import plotting
import random

# Plots LaTeX-Style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

### Door Hourglass ###

def init_door(door, gg):
    x = door[0]
    y = door[1]
    
    likelihood = 0.01 * np.ones((gg,gg))
    likelihood[x, y] = 1.0

    # Initialize Hourglass shape
    y_list = [y-1, y+1]
    for i in y_list:
        for j in range(-1, 2):
            likelihood[x+j, i] = 1.0/2.0
    y_list = [y-2, y+2]
    for i in y_list:
        for j in range(-2, 3):
            likelihood[x+j, i] = 1.0/3.0
    y_list = [y-3, y+3]
    for i in y_list:
        for j in range(-3, 4):
            likelihood[x+j, i] = 1.0/4.0
    
    return likelihood

### Belief ###

def update_likelihood(meas, x, y, gg):
    likelihood = 0.01 * np.ones((gg,gg))

    if meas == 1:
        likelihood[x, y] = 1.0
        # Hourglass shape
        y_list = [y-1, y+1]
        for i in y_list:
            for j in range(-1, 2):
                if gg > x+j >= 0 and 0 <= i < gg and likelihood[x+j, i] > 0.001:
                    likelihood[x+j, i] = 1.0/2.0
        y_list = [y-2, y+2]
        for i in y_list:
            for j in range(-2, 3):
                if gg > x+j >= 0 and 0 <= i < gg and likelihood[x+j, i] > 0.001:
                    likelihood[x+j, i] = 1.0/3.0
        y_list = [y-3, y+3]
        for i in y_list:
            for j in range(-3, 4):
                if gg > x+j >= 0 and 0 <= i < gg and likelihood[x+j, i] > 0.001:
                    likelihood[x+j, i] = 1.0/4.0
    
    else: # meas == 0
        likelihood[x, y] = 0

        y_list = [y-1, y+1]
        for i in y_list:
            for j in range(-1, 2):
                if gg > x+j >= 0 and 0 <= i < gg and likelihood[x+j, i] > 0.001:
                    likelihood[x+j, i] = 1.0/2.0
        y_list = [y-2, y+2]
        for i in y_list:
            for j in range(-2, 3):
                if gg > x+j >= 0 and 0 <= i < gg and likelihood[x+j, i] > 0.001:
                    likelihood[x+j, i] = 2.0/3.0
        y_list = [y-3, y+3]
        for i in y_list:
            for j in range(-3, 4):
                if gg > x+j >= 0 and 0 <= i < gg and likelihood[x+j, i] > 0.001:
                    likelihood[x+j, i] = 3.0/4.0

    return likelihood

def update_post(post, likelihood):
    post = post * likelihood

    post = post / np.sum(post) # normalize
    return post

### u ###

def get_options(x, y, gg):
    # Borders
    if x == 0 and 1 <= y <= gg-2:
        u_options = np.array([[1, 0], [0, 1], [0, -1], [0, 0]])
    elif x == gg-1 and 1 <= y <= gg-2:
        u_options = np.array([[-1, 0], [0, 1], [0, -1], [0, 0]])
    elif 1 <= x <= gg-2 and y == 0:
        u_options = np.array([[1, 0], [-1, 0], [0, 1], [0, 0]])
    elif 1 <= x <= gg-2 and y == gg-1:
        u_options = np.array([[1, 0], [-1, 0], [0, -1], [0, 0]])

    # Corners
    elif x == 0 and y == 0:
        u_options = np.array([[1, 0], [0, 1], [0, 0]])
    elif x == 0 and y == gg-1:
        u_options = np.array([[1, 0], [0, -1], [0, 0]])
    elif x == gg-1 and y == 0:
        u_options = np.array([[-1, 0], [0, 1], [0, 0]])
    elif x == gg-1 and y == gg-1:
        u_options = np.array([[-1, 0], [0, -1], [0, 0]])

    else:
        u_options = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])

    return u_options

def get_u(post, path, S, u_options, meas_likelihood):
    gg = len(post[0])
    best_u = np.array([0, 0])
    max_exp_red_S = -float('inf')
    exp_red_S_l = []
    
    for u in u_options: # cycle through all possible u
        rj = path[-1]+u

        exp_red_S = get_exp_red_S(post, rj, S, meas_likelihood)
        exp_red_S_l.append(exp_red_S)

        if exp_red_S > max_exp_red_S: # keep u with highest expected entropy recduction
            max_exp_red_S = exp_red_S
            best_u = u
    print(exp_red_S_l)
    return best_u

### Entropy ###

def get_exp_red_S(post, rj, S, meas_likelihood):
    gg = len(post[0])
    pmeas1 = meas_likelihood[int(rj[0]), int(rj[1])]
    likelihood0, likelihood1 = update_likelihood(0, rj[0], rj[1], gg), update_likelihood(1, rj[0], rj[1], gg)

    post1, post0 = update_post(post, likelihood1), update_post(post, likelihood0)

    S0, S1 = get_S(post0), get_S(post1)

    return pmeas1 * (S - S1) + (1-pmeas1) * (S - S0)

def get_S(prob):
    S = 0
    for x in prob:
        for p in x:
            if p < 1e-8:
                p = 1e-8
            S -= p * np.log(p)
    return S

### Main ###

def main():
    ### Initialize
    gg = 25
    eps = 0.0001

    for sim in range(4):
        # Start Locations
        door = np.random.randint(3,gg-3,2)
        start = np.random.randint(0,gg,2)
        # print(start)

        path = np.array([start])
        
        likelihood = init_door(door, gg)

        post = (1 / (gg**2)) * np.ones((gg,gg))
        S = get_S(post)

        ### Search
        i = 0
        while i < 200:
            if path[-1,0] != door[0] or path[-1,1] != door[1]:
                # Take binary measurement
                meas = int(np.random.random() < likelihood[int(path[-1, 0]),int(path[-1, 1])])
                
                # Update belief
                meas_likelihood = update_likelihood(meas, path[-1, 0], path[-1, 1], gg)

                post = update_post(post, meas_likelihood)
                
                u_options = get_options(path[-1,0], path[-1,1], gg)
                # u_options = [[-1, 0], [0, 1]]
                # Compute Entropy
                # S = get_S(post)
                print(S)

                # Take next step based on maximum expected entropy reduction
                u = get_u(post, path, S, u_options, meas_likelihood)
                # u = random.choice(u_options)
                path = np.vstack((path, path[-1] + u))
                plt.scatter(path[-1,0], path[-1,1])
                plt.draw()
                plt.imshow(post.T)
                plt.pause(0.001)
                # print(path[-1])
            else:
                break
            i += 1

        plotting.plot_trajectory(gg, door, path, sim)
    

if __name__ == "__main__":
    main()