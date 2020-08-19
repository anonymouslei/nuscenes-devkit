import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils

DATAROOT = '/home/jfei/leige/datasets/nuscenes/v1.0_mini_asia'

nusc_map = NuScenesMap(dataroot=DATAROOT, map_name='singapore-onenorth')

fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=1)

fig, ax = nusc_map.render_record('stop_line', nusc_map.stop_line[14]['token'])

patch_box = (300, 1700, 100, 100)
patch_angle = 0 # Default orientation where North is up
layer_names = ['drivable_area', 'walkway']
canvas_size = (1000, 1000)
map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
print(map_mask[0])

figsize = (12, 4)
fig, ax = nusc_map.render_map_mask(patch_box, 45, layer_names, canvas_size, figsize=figsize, n_row=1)

from nuscenes.nuscenes import NuScenes
nusc = NuScenes('v1.0-mini', dataroot=DATAROOT, verbose=False)

sample_token = nusc.sample[9]['token']
layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
camera_channel = 'CAM_FRONT'
img = nusc_map.render_map_in_image(nusc, sample_token, layer_names=layer_names, camera_channel=camera_channel)
plt.show(img)