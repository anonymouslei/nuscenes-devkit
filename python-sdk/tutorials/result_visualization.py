from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from nuscenes import NuScenes
# from visualization.visualization_result import Visualization

import sys
print(sys.path)
sys.path.append('/home/jfei/leige/code/nuscenes-devkit/python-sdk/visualization')
sys.path.append('/home/jfei/leige/code/nuscenes-devkit/python-sdk/tutorials')
sys.path.append('/home/jfei/leige/code/nuscenes-devkit/python-sdk')
from visualization.visualization_result import Visualization, MapRepresentation, InstanceRepresentation
import matplotlib.pyplot as plt
import torch

DATAROOT = '/home/jfei/leige/datasets/nuscenes/v1.0_mini_asia'

nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)

# Data Splits for the Prediction Challenge
mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)

# getting past and future data for an agent
helper = PredictHelper(nuscenes)
instance_token, sample_token = mini_train[0].split("_")
annotation = helper.get_sample_annotation(instance_token, sample_token)
future_xy_local = helper.get_future_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=True)
future_xy_global = helper.get_future_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=False)
helper.get_future_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=True, just_xy=False)
sample = helper.get_annotations_for_sample(sample_token)

nusc_map = NuScenesMap(map_name='singapore-onenorth', dataroot=DATAROOT)

# input representation
static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=6)
map_layer_rasterizer = MapRepresentation(helper)
instance_rasterizer = InstanceRepresentation(helper, seconds_of_history=0)
mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

instance_token_img, sample_token_img = 'bc38961ca0ac4b14ab90e547ba79fbb6', '7626dde27d604ac28a0240bdd54eba7a'
# anns = [ann for ann in nuscenes.sample_annotation if ann['instance_token'] == instance_token_img]
# img = mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)
# img = mtp_input_representation.make_static_layers(instance_token_img, sample_token_img)

result_visualization = Visualization(map_layer_rasterizer, instance_rasterizer, Rasterizer)
img = result_visualization.make_static_layers(instance_token_img, sample_token_img)
plt.imshow(img)
plt.show()

# Model Implementations
backbone = ResNetBackbone("resnet50")
mtp = MTP(backbone, num_modes=2)

# Note that the value of num_modes depends on the size of the lattice used for coverNet.
covernet = CoverNet(backbone, num_modes=64)

# the second input is a tensor containing the velocity, acceleration, and heading change rate for the agent
agent_state_vector = torch.Tensor([[helper.get_velocity_for_agent(instance_token_img, sample_token_img),
                                    helper.get_acceleration_for_agent(instance_token_img, sample_token_img),
                                    helper.get_heading_change_rate_for_agent(instance_token_img, sample_token_img)]])
image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

# output has 50 entries
# the first 24 are x,y coordinates (in the agent frame) over the next 6 seconds at 2 Hz for the first mode.
# the second 24 are the x,y coordinates for the second mode
# the last 2 are the logits of the mode probabilities
output = mtp(image_tensor, agent_state_vector)
# print(output)

# CoverNet outputs a probability distribution over the trajectory set(64).
# these are the logits of the probabilities
logits = covernet(image_tensor, agent_state_vector)
# print(logits)

import pickle
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

# print 5 most likely predictions
index = logits.argsort(descending=True)[:,0]
# print(logits.argsort(descending=True)[:5])
# print(trajectories_8[logits.argsort(descending=True)[:5]])
# save them as a list of lists

# trajectories_8 = trajectories_8.numpy()
# trajectories_8 = trajectories_8[10,:,:]
# plt.plot(trajectories_8[:, 0], trajectories_8[:, 1])
# plt.xlim(-45, 42)
# plt.ylim(-10, 120)

# trajectories_8 = trajectories_8.numpy()
# trajectories_8 = trajectories_8[index,:,:]
# plt.plot(trajectories_8[:, 0], trajectories_8[:, 1])
# plt.xlim(-45, 42)
# plt.ylim(-10, 120)
# plt.title("the best trajectory")
# plt.show()
