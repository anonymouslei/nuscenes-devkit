import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
# Epsilon is the amount of coverage in the set,
# i.e. a real world trajectory is at most 8 meters from a trajectory in this set
# We released the set for epsilon = 2, 4, 8. Consult the paper for more information
# on how this set was created
PATH_TO_EPSILON_8_SET = "/home/jfei/leige/datasets/nuscenes/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"
PATH_TO_EPSILON_4_SET = "/home/jfei/leige/datasets/nuscenes/nuscenes-prediction-challenge-trajectory-sets/epsilon_4.pkl"
PATH_TO_EPSILON_2_SET = "/home/jfei/leige/datasets/nuscenes/nuscenes-prediction-challenge-trajectory-sets/epsilon_2.pkl"
trajectories_8 = pickle.load(open(PATH_TO_EPSILON_8_SET, 'rb'))
trajectories_4 = pickle.load(open(PATH_TO_EPSILON_4_SET, 'rb'))
trajectories_2 = pickle.load(open(PATH_TO_EPSILON_2_SET, 'rb'))

# save them as a list of lists
trajectories_8 = torch.Tensor(trajectories_8)
trajectories_4 = torch.Tensor(trajectories_4)
trajectories_2 = torch.Tensor(trajectories_2)

trajectories_8 = trajectories_8.numpy()
trajectories_4 = trajectories_4.numpy()
trajectories_2 = trajectories_2.numpy()

trajectories_8 = trajectories_8[10,:,:]
plt.plot(trajectories_8[:, 0], trajectories_8[:, 1])
plt.xlim(-45, 42)
plt.ylim(-10, 120)

'''
Figure 2: fixed trajectory set generation 
plt.figure(0)
for i in range(64):
    plt.plot(trajectories_8[i, :, 0], trajectories_8[i, :, 1])
plt.title("epsilon 8")

plt.figure(1)
for i in range(415):
    plt.plot(trajectories_4[i, :, 0], trajectories_4[i, :, 1])
plt.title("epsilon 4")

plt.figure(2)
for i in range(2206):
    plt.plot(trajectories_2[i, :, 0], trajectories_2[i, :, 1])
plt.title("epsilon 2")

'''
plt.show()
