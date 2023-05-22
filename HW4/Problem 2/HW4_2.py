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

def update_post(post, meas, pmeas1, visited, likelihood, r0):
    gg = len(visited[0])
    x = r0[0]
    y = r0[1]

    px1 = 1 / (gg ** 2)
    if meas == 1:
        pmeas = px1
    else:
        pmeas = 1 - px1
    # if meas == 1:
    #     pmeas = pmeas1
    # else:
    #     pmeas = 1-pmeas1 

    for i in range(gg): #TODO
        for j in range(gg):
            pr0 = visited[i,j]
            Lr0 = get_Lr0(likelihood, i, j)
            post[i,j] = Lr0 * pr0 / pmeas

    post = post / np.sum(post) # normalize
    return post

def get_Lr0(likelihood, y, x):
    y_list, x_list = [y + 1, y - 1], [x + 1, x - 1]
    p_list = []
    gg = len(likelihood[0])

    for i in y_list:
        if 0 <= i < gg:
            p_list.append(likelihood[i][x])
    for j in x_list:
        if 0 <= j < gg:
            p_list.append(likelihood[y][j])

    Lr0 = sum(p_list) / len(p_list)
    return Lr0

def update_visited(visited, x, y, sq_unvisited):
    gg = len(visited[0])
    visited[x, y] = 0
    for i in range(gg):
        for j in range(gg):
            if visited[i, j] > 0.0001:
                visited[i, j] = 1 / sq_unvisited
    return visited

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

def get_u(post, path, S, u_options, gg, meas_likelihood, visited, sq_unvisited):
    best_u = np.array([0, 0])
    max_exp_red_S = 0
    
    for u in u_options: # cycle through all possible u
        rj = path[-1]+u

        exp_red_S = get_exp_red_S(post, rj, S, gg, meas_likelihood, visited, sq_unvisited)

        if exp_red_S > max_exp_red_S: # keep u with highest expected entropy recduction
            max_exp_red_S = exp_red_S
            best_u = u
    return best_u

### Entropy ###

def get_exp_red_S(post, rj, S, gg, meas_likelihood, visited, sq_unvisited): # TODO
    pmeas1 = meas_likelihood[int(rj[0]), int(rj[1])]
    likelihood0, likelihood1 = update_likelihood(0, rj[0], rj[1], gg), update_likelihood(1, rj[0], rj[1], gg)
    visited = update_visited(visited, rj[0], rj[1], sq_unvisited - 1)

    post1, post0 = update_post(post, 1, pmeas1, visited, likelihood1, rj), update_post(post, 0, pmeas1, visited, likelihood0, rj)

    S0, S1 = get_S(post0), get_S(post1)

    return pmeas1 * (S1 - S) + (1-pmeas1) * (S0 - S)

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
    eps = 0.1

    for sim in range(4):
        # Start Locations
        door = np.random.randint(3,gg-3,2)
        start = np.random.randint(0,gg,2)
        # print(start)

        path = np.array([start])
        
        likelihood = init_door(door, gg)
        sq_unvisited = gg ** 2
        visited = (1 / (gg**2)) * np.ones((gg,gg))

        post = (1 / (gg**2)) * np.ones((gg,gg))
        S = get_S(post)

        ### Search
        i = 0
        while S > eps and sq_unvisited > 0 and i < 500:
            if path[-1,0] != door[0] or path[-1,1] != door[1]:
                # Take binary measurement
                meas = int(np.random.random() < likelihood[int(path[-1, 0]),int(path[-1, 1])])
                
                # Update belief
                meas_likelihood = update_likelihood(meas, path[-1, 0], path[-1, 1], gg)
                sq_unvisited = gg ** 2 - len(np.unique(path, axis=0))
                visited = update_visited(visited, path[-1, 0], path[-1, 1], sq_unvisited)
                # print(visited)

                post = update_post(post, meas, meas_likelihood[int(path[-1, 0]),int(path[-1, 1])], visited, meas_likelihood, path[-1]) # TODO
                
                u_options = get_options(path[-1,0], path[-1,1], gg)

                # Compute Entropy
                S = get_S(post)
                print(S)

                # Take next step depending on expected entropy reduction
                u = get_u(post, path, S, u_options, gg, meas_likelihood, visited, sq_unvisited) # TODO
                # u = random.choice(u_options)
                path = np.vstack((path, path[-1] + u))
                plt.scatter(path[-1,0], path[-1,1])
                plt.draw()
                plt.pause(0.01)
                plt.imshow(post.T)

                #print(path[-1])
            else:
                break
            i += 1

        plotting.plot_trajectory(gg, door, path, sim)
    

if __name__ == "__main__":
    main()