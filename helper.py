# generic helper functions

import time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import random
import scipy as sp
import scipy.misc


# A generic function timer
def time_func(tag, func):
    t0 = time.time() if tag else None
    ret = func()
    if tag:
        t1 = time.time()
        print('Time for {}: {}'.format(tag, t1 - t0))
    return ret


# Compose a single-argument function n times
def repeat(func, n):
    def ret(x):
        for i in range(n):
            x = func(x)
        return x

    return ret


# E.g. _sort_by_occurrence(np.array([1, 3, 3, 1, 2, 2, 2, 3, 4, 2]))
# Return: array([2, 3, 1, 4])
def sort_by_occurrence(arr):
    u, counts = np.unique(arr, return_counts=True)
    sort_index = counts.argsort()[::-1]
    return u[sort_index]


# # https://en.wikipedia.org/wiki/Von_Neumann_neighborhood
# def _manhattan_neighbors(r=1):
#     neighbors = []
#     for dy in range(-r, r + 1):
#         xx = r - abs(dy)
#         for dx in range(-xx, xx + 1):
#             if dy == 0 and dx == 0:
#                 continue
#             neighbors.append((dy, dx))
#     return neighbors
# https://en.wikipedia.org/wiki/Lennard-Jones_potential

# Color map for grayscale images
_cm_greys = plt.cm.get_cmap('Greys')


# Show image in matplotlib window
def show_image(img, cmap=_cm_greys, title=None, interp=None):
    plt.clf()
    plt.axis('off')
    plt.imshow(img, cmap=cmap, interpolation=interp)
    if title:
        plt.title(title)
    plt.show()


def _lj(r, delta=4):
    return np.power(delta / r, 12) - 2 * np.power(delta / r, 6)


# https://en.wikipedia.org/wiki/Simulated_annealing
def anneal(img, num_steps=1000):
    np.seterr(divide='ignore', invalid='ignore')
    height, width = img.shape
    # TODO: Use RGB for now, just for visualization
    new_img = np.zeros((height, width, 3))
    for i in range(3):
        new_img[:, :, i] = 1 - img.copy()
    positions = []
    for y in range(height):
        for x in range(width):
            if img[y, x] == 1:
                new_img[y, x, 0] = 1
                positions.append((y, x))
    positions = np.array(positions)
    num_positions = positions.shape[0]
    print('{} Positions'.format(num_positions))
    particles = np.ones(num_positions, dtype=bool)
    # plt.ion()
    # show_image(new_img)
    # TODO: Just for testing
    E = 0
    # step_list= []
    # E_list = []
    # for p in range(num_positions):
    #     for q in range(p + 1, num_positions):
    #         E += _lj(la.norm(positions[q] - positions[p]))
    for step in range(num_steps):
        beta = (3 + step / 1000) * 1e-6
        # Choose a position randomly, and invert the state
        p = np.random.randint(num_positions)
        y, x = positions[p]
        # noinspection PyTypeChecker
        delta_energy = np.nansum(
            _lj(la.norm(positions[particles] - positions[p], axis=1)))
        if particles[p]:
            delta_energy = -delta_energy
        if delta_energy < 0:
            accept = True
        else:
            accept = (random.random() < np.exp(-beta * delta_energy))
        if accept:
            E += delta_energy
            particles[p] = not particles[p]
            new_img[y, x, 0] = particles[p]
        if step % 50 == 0:
            print('Step {}. beta {}. E {}'.format(step, beta, E))
            # step_list.append(step)
            # E_list.append(E)
            # show_image(new_img, title=step, interp='none')
            # plt.pause(0.1)
    # plt.ioff()
    # plt.clf()
    # plt.plot(step_list, E_list, '*-')
    # plt.xlabel('step')
    # plt.ylabel('Energy')
    # plt.show()
    return new_img


# Capitalize all characters in a sequence
def canonicalize(seq):
    return seq.upper()


# resize a image to new height and width
def resize(image, height, width):
    return sp.misc.imresize(
        image,
        (height, width)
    )
