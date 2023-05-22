import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_trajectory
import random
from PIL import Image
import os, shutil


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
    
    if meas == 1:
        likelihood = 0.01 * np.ones((gg,gg))
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
        likelihood = 0.99 * np.ones((gg,gg))
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

def update_post(visited, likelihood):
    post = visited * likelihood

    post = post / np.sum(post) # normalize
    return post

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

def get_u(post, path, S, u_options, meas_likelihood, meas, visited, sq_unvisited, i):
    gg = len(post[0])
    best_u = np.array([0, 0])
    max_exp_red_S = -float('inf')
    exp_red_S_l = []
    
    for u in u_options: # cycle through all possible u
        rj = path[-1]+u

        exp_red_S = get_exp_red_S(post, rj, S, meas_likelihood, visited, sq_unvisited)
        exp_red_S_l.append(exp_red_S)

        if exp_red_S > max_exp_red_S: # keep u with highest expected entropy recduction
            max_exp_red_S = exp_red_S
            best_u = u
    print('iter={}\tS={}\tz={}\tfsq={}\tu={}\texp_red_S={}'.format(i, S, meas, sq_unvisited, best_u, exp_red_S_l))
    return best_u

### Entropy ###

def get_exp_red_S(post, rj, S, meas_likelihood, visited, sq_unvisited):
    gg = len(post[0])
    pmeas1 = meas_likelihood[int(rj[0]), int(rj[1])]
    likelihood0, likelihood1 = update_likelihood(0, rj[0], rj[1], gg), update_likelihood(1, rj[0], rj[1], gg)
    visited = update_visited(visited, rj[0], rj[1], sq_unvisited - 1)

    post1, post0 = update_post(post, likelihood1), update_post(post, likelihood0)

    S0, S1 = get_S(post0), get_S(post1)

    return np.abs(pmeas1 * (S - S1) + (1-pmeas1) * (S - S0))

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
    
    for sim in range(4):
        belief_plots = []
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

        fig2, ax = plt.subplots()

        ### Search
        i = 0
        while i < 1000:
            if path[-1,0] != door[0] or path[-1,1] != door[1]:
                # Take binary measurement
                meas = int(np.random.random() < likelihood[int(path[-1, 0]),int(path[-1, 1])])
                
                # Update belief
                meas_likelihood = update_likelihood(meas, path[-1, 0], path[-1, 1], gg)
                sq_unvisited = gg ** 2 - len(np.unique(path, axis=0))
                visited = update_visited(visited, path[-1, 0], path[-1, 1], sq_unvisited)

                post = update_post(post, meas_likelihood)
                
                u_options = get_options(path[-1,0], path[-1,1], gg)

                # Compute Entropy
                S = get_S(post)
                
                # Take next step based on maximum expected entropy reduction
                u = get_u(post, path, S, u_options, meas_likelihood, meas, visited, sq_unvisited, i)

                path = np.vstack((path, path[-1] + u))
                
                # Belief gif
                ax.scatter(path[-1,0], path[-1,1])
                #ax.draw()
                ax.imshow(post.T, origin='lower')
                ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                ax.set_title('Run ' + str(sim+1))
                fig2.savefig('./ME455_ActiveLearning/HW4/Problem 2/plots/belief_run' + str(sim+1) + '/belief' + str(i) + '.png')
                belief_plots.append(Image.open('./ME455_ActiveLearning/HW4/Problem 2/plots/belief_run' + str(sim+1) + '/belief' + str(i) + '.png'))

            else:
                break
            i += 1

        # Belief gif
        belief_plots[0].save('./ME455_ActiveLearning/HW4/Problem 2/plots/belief_run' + str(sim+1) + '.gif', save_all=True, append_images=belief_plots[1:], optimize=False, duration=int(len(belief_plots)/10), loop=0)
        plot_trajectory(gg, door, path, sim)
    

if __name__ == "__main__":
    for i in range(1,5):
        for root, dirs, files in os.walk('./ME455_ActiveLearning/HW4/Problem 2/plots/belief_run' + str(i)):
            for file in files:
                os.remove(os.path.join(root, file))
    main()