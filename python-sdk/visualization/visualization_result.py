# Lei Ge
# 2020.08.14
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction.input_representation.static_layers import change_color_of_binary_mask, correct_yaw, \
    get_patchbox, angle_of_rotation, draw_lanes_in_agent_frame, get_crops, load_all_maps
from nuscenes.prediction.helper import PredictHelper
from nuscenes.prediction.input_representation.agents import reverse_history, default_colors, add_present_time_to_history, \
    get_rotation_matrix, History, get_track_box, convert_to_pixel_coords
from nuscenes.prediction.input_representation.combinators import add_foreground_to_image, Rasterizer
import numpy as np
from functools import reduce
from typing import List, Tuple, Callable, Dict, Any
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

import cv2

import sys
print(sys.path)
Color = Tuple[float, float, float]

def get_instance_location_yaw(annotation: Dict[str, Any]):

    location = annotation['translation'][:2]
    yaw_in_radians = quaternion_yaw(Quaternion(annotation['rotation']))

    return location, yaw_in_radians

def draw_trajectory(image: np.ndarray,
                    annotation: Dict[str, Any],
                    trajecotry: np.ndarray,
                    center_pixels: Tuple[float, float],
                    resolution: float = 0.1
                    ) -> None:
    assert resolution > 0
    instance_location, instance_yaw_in_radius = get_instance_location_yaw(annotation)

    trajecotry_global = trajecotry + instance_location
    row_pixel = []
    column_pixel = []
    for pos_x, pos_y in trajecotry[:, 0], trajecotry[:, 1]:
        continue




def draw_instance_box(center_agent_annotation: Dict[str, Any],
                     center_agent_pixels: Tuple[float, float],
                     agent_history: History,
                     base_image: np.ndarray,
                     trajectory: np.ndarray,
                     resolution: float = 0.1) -> None:
    """
    Draws past sequence of agent boxes on the image.
    :param center_agent_annotation: Annotation record for the agent
        that is in the center of the image.
    :param center_agent_pixels: Pixel location of the agent in the
        center of the image.
    :param agent_history: History for all agents in the scene.
    :param base_image: Image to draw the agents in.
    :param get_color: Mapping from category_name to RGB tuple.
    :param resolution: Size of the image in pixels / meter.
    :return: None.
    """

    # 车辆外参，偏移矩阵，单位为米
    agent_x, agent_y = center_agent_annotation['translation'][:2]

    for instance_token, annotations in agent_history.items():

        if instance_token == center_agent_annotation['instance_token']:
            color = (255, 0, 0)
        else:
            continue

        for i, annotation in enumerate(annotations):

            box = get_track_box(annotation, (agent_x, agent_y), center_agent_pixels, resolution)
            test = np.int0(box)
            cv2.fillPoly(base_image, pts=[np.int0(box)], color=color)

            draw_trajectory(base_image, annotation, trajectory, center_agent_pixels, resolution)

class MapRepresentation():
    """
    Creates a representation of the static map layers where
    the map layers are given a color and rasterized onto a
    three channel image.
    """

    def __init__(self, helper: PredictHelper,
                 layer_names: List[str] = None,
                 colors: List[Color] = None,
                 resolution: float = 0.1, # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25):

        self.helper = helper
        self.maps = load_all_maps(helper)

        if not layer_names:
            layer_names = ['drivable_area', 'ped_crossing', 'walkway']
        self.layer_names = layer_names

        if not colors:
            colors = [(255, 255, 255), (119, 136, 153), (0, 0, 255)]
        self.colors = colors

        self.resolution = resolution
        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right
        self.combinator = Rasterizer()

    def make_layer_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        sample_annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        map_name = self.helper.get_map_name_from_sample_token(sample_token)

        x, y = sample_annotation['translation'][:2]

        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))

        yaw_corrected = correct_yaw(yaw)

        image_side_length = 2 * max(self.meters_ahead, self.meters_behind,
                                    self.meters_left, self.meters_right)
        image_side_length_pixels = int(image_side_length / self.resolution)

        patchbox = get_patchbox(x, y, image_side_length)

        angle_in_degrees = angle_of_rotation(yaw_corrected) * 180 / np.pi

        canvas_size = (image_side_length_pixels, image_side_length_pixels)

        masks = self.maps[map_name].get_map_mask(patchbox, angle_in_degrees, self.layer_names, canvas_size=canvas_size)

        images = []
        for mask, color in zip(masks, self.colors):
            images.append(change_color_of_binary_mask(np.repeat(mask[::-1, :, np.newaxis], 3, 2), color))

        lanes = draw_lanes_in_agent_frame(image_side_length_pixels, x, y, yaw, radius=50,
                                          image_resolution=self.resolution, discretization_resolution_meters=1,
                                          map_api=self.maps[map_name])

        images.append(lanes)

        image = self.combinator.combine(images)

        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind, self.meters_left,
                                       self.meters_right, self.resolution,
                                       int(image_side_length / self.resolution))

        return image[row_crop, col_crop, :]


class InstanceRepresentation():

    """
    Represents the past sequence of agent states as a three-channel
    image with faded 2d boxes.
    """

    def __init__(self, helper: PredictHelper,
                 seconds_of_history: float = 0,
                 frequency_in_hz: float = 2,
                 resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25,
                 color_mapping: Callable[[str], Tuple[int, int, int]] = None):

        self.helper = helper
        self.seconds_of_history = seconds_of_history
        self.frequency_in_hz = frequency_in_hz

        if not resolution > 0:
            raise ValueError(f"Resolution must be positive. Received {resolution}.")

        self.resolution = resolution

        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right

        if not color_mapping:
            color_mapping = default_colors

        self.color_mapping = color_mapping

    def make_representation(self, instance_token: str, sample_token: str, trajectory: np.ndarray) -> np.ndarray:
        # Taking radius around track before to ensure all actors are in image
        buffer = max([self.meters_ahead, self.meters_behind,
                      self.meters_left, self.meters_right]) * 2

        image_side_length = int(buffer / self.resolution)

        # We will center the track in the image
        central_track_pixels = (image_side_length / 2, image_side_length / 2)

        base_image = np.zeros((image_side_length, image_side_length, 3))

        history = self.helper.get_past_for_sample(sample_token,
                                                  self.seconds_of_history,
                                                  in_agent_frame=False,
                                                  just_xy=False)
        history = reverse_history(history)

        present_time = self.helper.get_annotations_for_sample(sample_token)

        history = add_present_time_to_history(present_time, history)

        center_agent_annotation = self.helper.get_sample_annotation(instance_token, sample_token)

        draw_instance_box(center_agent_annotation, central_track_pixels,
                         history, base_image, trajectory, resolution=self.resolution)
        # draw_trajectory()
        plt.imshow(base_image)
        plt.show()

        center_agent_yaw = quaternion_yaw(Quaternion(center_agent_annotation['rotation']))
        rotation_mat = get_rotation_matrix(base_image.shape, center_agent_yaw)

        rotated_image = cv2.warpAffine(base_image, rotation_mat, (base_image.shape[1],
                                                                  base_image.shape[0]))

        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind,
                                       self.meters_left, self.meters_right, self.resolution,
                                       image_side_length)

        return rotated_image[row_crop, col_crop].astype('uint8')


class Combinator():
    """ Combines the StaticLayer and Agent representations into a single one. """

    def combine(self, data: List[np.ndarray]) -> np.ndarray:
        # All images in the dict are the same shape
        image_shape = data[0].shape

        base_image = np.zeros(image_shape).astype("uint8")
        return reduce(add_foreground_to_image, [base_image] + data)


class Visualization:

    def __init__(self, map_layer: MapRepresentation, agent: InstanceRepresentation,
                 combinator: Combinator):
        self.static_layer_rasterizer = map_layer
        self.agent_rasterizer = agent
        self.combinator = combinator

    def combine(self, data: List[np.ndarray]) -> np.ndarray:
        # All images in the dict are the same shape
        image_shape = data[0].shape

        base_image = np.zeros(image_shape).astype("uint8")
        return reduce(add_foreground_to_image, [base_image] + data)

    def make_static_layers(self, instance_token: str, sample_token: str, trajectory: np.ndarray) -> np.ndarray:
        static_layers = self.static_layer_rasterizer.make_layer_representation(instance_token, sample_token)
        # agents = np.zeros(static_layers.shape, dtype=np.uint8)
        agents_1 = self.agent_rasterizer.make_representation(instance_token, sample_token, trajectory)

        # return static_layers
        return self.combine([static_layers, agents_1])

