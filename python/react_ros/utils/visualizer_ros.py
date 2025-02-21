import numpy as np
from typing import Dict, List
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from hydra_seg_ros.utils.labels import COCO_COLORS

from react.core.instance_cluster import InstanceCluster
from react.utils.logger import getLogger

import rospy

logger = getLogger(name=__name__, log_file="visualizer_ros.log")


class ReactVisualizerRos:
    def __init__(self, ref_scan_id: int = 0, new_scan_id: int = 1):
        self.ref_objects_pub = rospy.Publisher("~ref_scan", MarkerArray, queue_size=10)
        self.position_updates_pub = rospy.Publisher(
            "~position_updates", MarkerArray, queue_size=10
        )
        self.map_frame_id = str(rospy.get_param("~map_frame_id", "map"))
        self.ref_scan_id = ref_scan_id
        self.new_scan_id = new_scan_id

    def publish_ref_objects(
        self,
        instance_clusters: Dict[int, InstanceCluster],
        stamp: rospy.Time,
    ):
        prev_objects_viz_msg = MarkerArray()
        stamp = stamp
        for cluster in instance_clusters.values():
            self.get_instance_cluster_viz_msg(
                instance_cluster=cluster,
                markers_msg=prev_objects_viz_msg,
                stamp=stamp,
                prev_scan_id=self.ref_scan_id,
                frame_id=self.map_frame_id,
            )
        self.ref_objects_pub.publish(prev_objects_viz_msg)

    def get_instance_cluster_viz_msg(
        self,
        instance_cluster: InstanceCluster,
        markers_msg: MarkerArray,
        stamp: rospy.Time,
        prev_scan_id: int,
        frame_id: str = "map",
        offset_z: float = 3.0,
    ):
        if not markers_msg.markers:
            markers_msg.markers = []
        set_color = [c / 255 for c in COCO_COLORS[instance_cluster.get_class_id()]]
        position_histories = instance_cluster.get_cluster_position_history(
            [prev_scan_id]
        )
        for instance_id, pos in position_histories.items():
            if prev_scan_id not in pos.keys():
                continue
            node_viz_msg = self.create_node_viz_msg(
                color=set_color,
                pos=pos[prev_scan_id],
                stamp=stamp,
                frame_id=frame_id,
                ns="InstanceSet "
                + instance_cluster.get_name()
                + " "
                + str(instance_cluster.cluster_id),
                marker_id=instance_id,
                offset_z=offset_z,
            )
            markers_msg.markers.append(node_viz_msg)

    def create_node_viz_msg(
        self,
        color: List,
        pos: np.ndarray,
        stamp: rospy.Time,
        frame_id: str,
        ns: str,
        marker_id: int,
        offset_z: float,
    ) -> Marker:
        node_viz_msg = Marker()
        node_viz_msg.header.stamp = stamp
        node_viz_msg.header.frame_id = frame_id
        node_viz_msg.ns = ns
        node_viz_msg.id = marker_id
        node_viz_msg.type = Marker.CUBE
        node_viz_msg.action = Marker.ADD
        node_viz_msg.pose.position.x = pos[0]
        node_viz_msg.pose.position.y = pos[1]
        # Move them down to easily visualize them
        node_viz_msg.pose.position.z = pos[2] - offset_z
        node_viz_msg.pose.orientation.x = 0.0
        node_viz_msg.pose.orientation.y = 0.0
        node_viz_msg.pose.orientation.z = 0.0
        node_viz_msg.pose.orientation.w = 1.0
        node_viz_msg.scale.x = 0.2
        node_viz_msg.scale.y = 0.2
        node_viz_msg.scale.z = 0.2
        node_viz_msg.color.r = color[0]
        node_viz_msg.color.g = color[1]
        node_viz_msg.color.b = color[2]
        node_viz_msg.color.a = 0.7
        return node_viz_msg

    def publish_position_updates(
        self,
        instance_clusters: Dict[int, InstanceCluster],
        stamp: rospy.Time,
        offset_z: float = 3.0,
    ):
        self.position_updates_pub.publish(
            self.get_delete_all_markers_msg(stamp=stamp, frame_id=self.map_frame_id)
        )
        position_updates_msg = MarkerArray()
        for cluster in instance_clusters.values():
            self.get_instance_set_position_update_msg(
                instance_cluster=cluster,
                stamp=stamp,
                position_updates_msg=position_updates_msg,
                frame_id=self.map_frame_id,
                ref_scan_id=self.ref_scan_id,
                new_scan_id=self.new_scan_id,
                offset_z=offset_z,
            )
        self.position_updates_pub.publish(position_updates_msg)

    def get_instance_set_position_update_msg(
        self,
        instance_cluster: InstanceCluster,
        stamp: rospy.Time,
        position_updates_msg: MarkerArray,
        frame_id: str,
        ref_scan_id: int,
        new_scan_id: int,
        offset_z: float,
    ) -> MarkerArray:
        if position_updates_msg.markers is None:
            position_updates_msg.markers = []
        for inst_id, ph in instance_cluster.get_cluster_position_history(
            scan_ids=[ref_scan_id, new_scan_id]
        ).items():
            if new_scan_id in ph and ref_scan_id in ph:
                prev_pos = ph[ref_scan_id]
                new_pos = ph[new_scan_id]
                pos_update_msg = self.create_position_updates_msg(
                    stamp=stamp,
                    frame_id=frame_id,
                    ns=f"Instance {instance_cluster.get_name()} {inst_id}",
                    marker_id=instance_cluster.cluster_id,
                    prev_pos=prev_pos,
                    new_pos=new_pos,
                    offset_z=offset_z,
                )
                position_updates_msg.markers.append(pos_update_msg)
        return position_updates_msg

    def create_position_updates_msg(
        self,
        stamp: rospy.Time,
        frame_id: str,
        ns: str,
        marker_id: int,
        prev_pos: np.ndarray,
        new_pos: np.ndarray,
        offset_z: float,
    ) -> Marker:
        pos_update_msg = Marker()
        pos_update_msg.header.stamp = stamp
        pos_update_msg.header.frame_id = frame_id
        pos_update_msg.ns = ns
        pos_update_msg.id = marker_id
        pos_update_msg.type = Marker.ARROW
        pos_update_msg.action = Marker.ADD

        start_point = Point()
        start_point.x = prev_pos[0]
        start_point.y = prev_pos[1]
        start_point.z = prev_pos[2] - offset_z

        end_point = Point()
        end_point.x = new_pos[0]
        end_point.y = new_pos[1]
        end_point.z = new_pos[2]

        pos_update_msg.points = []
        pos_update_msg.points.append(start_point)
        pos_update_msg.points.append(end_point)

        # Set arrow properties (color, scale)
        pos_update_msg.scale.x = 0.05  # Shaft diameter
        pos_update_msg.scale.y = 0.1  # Head diameter
        pos_update_msg.scale.z = 0.1  # Head length

        pos_update_msg.color.r = 0.0
        pos_update_msg.color.g = 0.0
        pos_update_msg.color.b = 0.0
        pos_update_msg.color.a = 1.0  # Alpha (transparency)
        # pos_update_msg.lifetime = rospy.Duration(1)
        return pos_update_msg

    def get_delete_all_markers_msg(
        self, stamp: rospy.Time, frame_id: str
    ) -> MarkerArray:
        marker_del = Marker()
        marker_del.action = marker_del.DELETEALL
        marker_del.header.stamp = stamp
        marker_del.header.frame_id = frame_id

        markers_del = MarkerArray()
        markers_del.markers = [marker_del]

        return markers_del
