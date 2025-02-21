import time
import numpy as np
import os
from typing import Dict, List, Set, Tuple
import torch
from torch import Tensor
from copy import deepcopy

import rospy
from cv_bridge import CvBridge

from hydra_stretch_msgs.msg import (
    HydraVisionPacket,
    InstanceViewHeader,
    Mask,
    ObjectLayerInfo,
    ObjectNodeInfo,
)

from react.core.object_node import ObjectNode
from react.core.react_manager import ReactManager
from react.net.embedding_net import get_embedding_model
from react.utils.logger import getLogger
from react.utils.read_data import get_bbox
from react.utils.image import preprocess_image, get_instance_view
from react_ros.utils.visualizer_ros import ReactVisualizerRos


DSG_PATH = "/home/ros/dsg_output/"

logger = getLogger(name=__name__, log_file="react_ros.log")


class ReactRosModule:
    def __init__(self):
        rospy.init_node("react_ros")
        logger.info("Starting ReactRosModule.")
        self._init()

    def _init(self):
        self._init_ros_params()
        self._init_ros()
        self._init_react_cores()

    def _init_ros_params(self):
        # ROS params
        ref_scan_id = rospy.get_param("~ref_scan_id", 0)
        assert isinstance(ref_scan_id, int)
        new_scan_id = rospy.get_param("~new_scan_id", 1)
        assert isinstance(new_scan_id, int)
        self.ref_scan_id = ref_scan_id
        self.new_scan_id = new_scan_id
        self.scene_graph = str(rospy.get_param("~scene_graph", "coffee_room"))
        self.weights = str(
            rospy.get_param(
                "~weights",
                f"/home/ros/models/embeddings/iros25/embedding_{self.scene_graph}.pth",
            )
        )
        self.backbone = str(rospy.get_param("~embedding_backbone", "efficientnet_b2"))
        match_threshold = rospy.get_param("~match_threshold", 2.2)
        assert isinstance(
            match_threshold, float
        ), f"{self.match_threshold} needs to be float"
        self.match_threshold = float(match_threshold)
        self.ref_dsg = str(rospy.get_param("~ref_dsg", f"{self.scene_graph}_1"))
        self.packet_topic = rospy.get_param(
            "~packet_topic", "/yolo_ros_node/vision_packet"
        )
        self.object_topic = rospy.get_param(
            "~object_topic", "/hydra_ros_node/frontend/object_nodes"
        )

    def _init_ros(self):
        # ROS Subscribers
        self.packet_sub = rospy.Subscriber(
            self.packet_topic, HydraVisionPacket, callback=self._packet_callback
        )
        self.object_sub = rospy.Subscriber(
            self.object_topic, ObjectLayerInfo, callback=self._object_callback
        )
        # ROS Publishers
        self.visualizer = ReactVisualizerRos(
            ref_scan_id=self.ref_scan_id, new_scan_id=self.new_scan_id
        )
        # OpenCV Bridge
        self.bridge = CvBridge()

    def _init_react_cores(self):
        self.embedding_model = get_embedding_model(
            weights=str(self.weights), backbone=str(self.backbone)
        )
        self.embedding_library: Dict[int, Tensor] = {}  # {mask_id -> embedding}
        self.manager = ReactManager(
            match_threshold=self.match_threshold, embedding_model=self.embedding_model
        )
        self._load_ref_dsg(ref_dsg=os.path.join(DSG_PATH, self.ref_dsg))

    def _load_ref_dsg(self, ref_dsg: str):
        assert os.path.exists(ref_dsg), f"{ref_dsg} does not exists"
        self.manager.process_dsg(
            scan_id=self.ref_scan_id,
            dsg_path=ref_dsg,
        )
        self.matched_nodes = set()
        logger.info(f"Loaded dsg from {ref_dsg}")
        self.ref_clusters = deepcopy(self.manager._instance_clusters)
        assert len(self.ref_clusters) > 0, "Reference dsg is empty!"

    def _packet_callback(self, msg: HydraVisionPacket):
        map_view_cv = self.bridge.imgmsg_to_cv2(msg.color, desired_encoding="bgr8")
        assert isinstance(map_view_cv, np.ndarray), "Invalid map_view_img"
        masks = msg.masks.masks
        if not masks:
            return
        tensor_list = []
        mask_ids = []
        for m in masks:
            if m.mask_id not in self.embedding_library:
                assert isinstance(m, Mask), f"Receive invalid Mask message"
                mask_cv = self.bridge.imgmsg_to_cv2(m.data, desired_encoding="mono8")
                view_cv = get_instance_view(
                    mask=mask_cv,
                    map_view_img=map_view_cv,
                    mask_bg=True,
                    crop=True,
                    padding=10,
                )
                view_tensor = preprocess_image(view_cv)
                tensor_list.append(view_tensor)
                mask_ids.append(m.mask_id)
        view_tensor_batch = torch.cat(tensor_list, dim=0)
        embedding_batch: Tensor = (
            self.embedding_model(view_tensor_batch.cuda()).detach().cpu()
        )
        for i, mask_id in enumerate(mask_ids):
            self.embedding_library[mask_id] = embedding_batch[i, :]

    def _check_object_node_info(self, object_node_msg: ObjectNodeInfo):
        assert isinstance(
            object_node_msg, ObjectNodeInfo
        ), "Invalid msg in ObjectLayerInfo"
        assert (
            isinstance(object_node_msg.position, Tuple)
            and len(object_node_msg.position) == 3
        ), f"object position is invalid {object_node_msg.position}"
        assert (
            isinstance(object_node_msg.bounding_box.dimensions, Tuple)
            and len(object_node_msg.bounding_box.dimensions) == 3
        ), f"object bbox dim is invalid {object_node_msg.bounding_box.dimensions}"
        assert (
            isinstance(object_node_msg.bounding_box.world_P_center, Tuple)
            and len(object_node_msg.bounding_box.world_P_center) == 3
        ), f"object bbox position is invalid {object_node_msg.bounding_box.world_P_center}"

    def _object_callback(self, msg: ObjectLayerInfo):
        if not msg.nodes:
            return
        # Reset the map updater to original state to re-execute assignments
        self.manager.assign_instance_clusters(instance_clusters=self.ref_clusters)
        online_nodes: Dict[int, ObjectNode] = {}
        for object_node_msg in msg.nodes:
            self._check_object_node_info(object_node_msg)
            instance_view_headers = object_node_msg.instance_view_headers
            if not instance_view_headers:
                continue
            mask_ids = set()
            for view_header in instance_view_headers:
                assert isinstance(
                    view_header, InstanceViewHeader
                ), "Invalid InstanceViewHeader in ObjectNodeInfo"
                mask_ids.add(view_header.mask_id)
            object_node = self.get_object_node_online(
                scan_id=self.new_scan_id,
                mask_ids=mask_ids,
                node_id=object_node_msg.node_id,
                name=object_node_msg.name,
                class_id=object_node_msg.class_id,
                position=np.array(object_node_msg.position),
                bbox_dims=list(object_node_msg.bounding_box.dimensions),
                bbox_pos=list(object_node_msg.bounding_box.world_P_center),
            )
            online_nodes[self.manager.assign_instance_id()] = object_node
        self.manager.init_instance_clusters(
            scan_id=self.new_scan_id, object_nodes=online_nodes
        )
        self.manager.optimize_cluster()
        self.manager.update_position_histories(
            scan_id_old=self.ref_scan_id, scan_id_new=self.new_scan_id
        )
        stamp_now = rospy.Time.now()
        self.visualizer.publish_ref_objects(
            self.manager.get_instance_clusters(), stamp=stamp_now
        )
        self.visualizer.publish_position_updates(
            self.manager.get_instance_clusters(), stamp=stamp_now
        )

    #
    # def _get_embedding(self, mask_ids: Set):
    #     embedding_list = []
    #     for id in mask_ids:
    #         if id in self.embedding_library.keys():
    #             embedding_list.append(self.embedding_library[id][None, :])
    #     embedding = torch.quantile(torch.cat(embedding_list), q=0.5, dim=0)
    #     return embedding

    def get_object_node_online(
        self,
        scan_id: int,
        mask_ids: Set,
        node_id: int,
        name: str,
        class_id: int,
        position: np.ndarray,
        bbox_dims: List,
        bbox_pos: List,
    ) -> ObjectNode:
        assert len(bbox_dims) == 3, f"Bbox dimension is {len(bbox_dims)}, should be 3"
        assert len(bbox_pos) == 3, f"Bbox position is {len(bbox_pos)}, should be 3"
        assert len(position) == 3, f"Object position is {len(position)}, should be 3"
        bbox = get_bbox(dimensions=bbox_dims, position=bbox_pos)
        embedding_list = []
        instance_views = {}
        for id in mask_ids:
            if id in self.embedding_library.keys():
                embedding_list.append(self.embedding_library[id][None, :])
                instance_views[id] = np.empty(
                    []
                )  # Create placeholder to calculate average of embedding for cluster
            else:
                logger.warning(f"Missing embedding id {id}")
        embedding = torch.quantile(torch.cat(embedding_list), q=0.5, dim=0)
        node = ObjectNode(
            scan_id=scan_id,
            node_id=node_id,
            class_id=class_id,
            name=name,
            position=position,
            instance_views=instance_views,
            bbox=bbox,
            embedding=embedding,
        )
        return node


def main():
    ReactRosModule()
    rospy.spin()
