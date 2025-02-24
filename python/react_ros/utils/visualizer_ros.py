from tracemalloc import start
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
        offset_z_nodes: float = 3.0,
        offset_z_clusters: float = 6.0,
    ):
        if not markers_msg.markers:
            markers_msg.markers = []
        cluster_color = [c / 255 for c in COCO_COLORS[instance_cluster.get_class_id()]]
        position_histories = instance_cluster.get_cluster_position_history(
            [prev_scan_id]
        )
        inst_ids = []
        positions = []
        for instance_id, pos in position_histories.items():
            if prev_scan_id not in pos.keys():
                continue
            node_viz_msg = self.create_cube_msg(
                stamp=stamp,
                ns="Instance " + instance_cluster.get_name() + " " + str(instance_id),
                frame_id=frame_id,
                marker_id=instance_id,
                pos=pos[prev_scan_id] - np.array([0, 0, offset_z_nodes]),
                color=cluster_color,
            )
            positions.append(pos[prev_scan_id])
            inst_ids.append(instance_id)
            markers_msg.markers.append(node_viz_msg)
        if len(positions) > 0:
            cluster_pos = np.array(positions).mean(axis=0)
            cluster_viz = self.create_cluster_viz_msg(
                color=cluster_color,
                cluster_pos=cluster_pos,
                node_pos=positions,
                stamp=stamp,
                frame_id=frame_id,
                name=instance_cluster.get_name(),
                cluster_id=instance_cluster.cluster_id,
                inst_ids=inst_ids,
                offset_z_nodes=offset_z_nodes,
                offset_z_cluster=offset_z_clusters,
            )
            markers_msg.markers += cluster_viz

    def create_cluster_viz_msg(
        self,
        color: List,
        cluster_pos: np.ndarray,
        node_pos: List[np.ndarray],
        stamp: rospy.Time,
        frame_id: str,
        name: str,
        cluster_id: int,
        inst_ids: List[int],
        offset_z_nodes: float,
        offset_z_cluster: float,
    ) -> List[Marker]:
        cluster_viz = []
        cluster_viz_msg = self.create_cube_msg(
            stamp=stamp,
            frame_id=frame_id,
            ns=f"{name} cluster {cluster_id}",
            marker_id=cluster_id * 10000,
            pos=cluster_pos - np.array([0, 0, offset_z_cluster]),
            color=color,
        )
        cluster_viz.append(cluster_viz_msg)
        cluster_text_msg = self.create_text_msg(
            stamp=stamp,
            frame_id=frame_id,
            ns=f"text {name} cluster {cluster_id}",
            id=cluster_id * 100000,
            text=f"{name} {cluster_id}",
            pos=cluster_pos - np.array([0, 0, offset_z_cluster + 0.5]),
        )
        cluster_viz.append(cluster_text_msg)

        assert len(inst_ids) == len(node_pos)
        for node_p, id in zip(node_pos, inst_ids):
            cluster_node_msg = self.create_arrow_msg(
                stamp=stamp,
                frame_id=frame_id,
                ns=f"{name} cluster {cluster_id} - instance {id}",
                marker_id=cluster_id * 10000 + id,
                start=cluster_pos - np.array([0, 0, offset_z_cluster]),
                end=node_p - np.array([0, 0, offset_z_nodes]),
                scale=[0.01, 0.05, 0.05],
            )
            cluster_viz.append(cluster_node_msg)
        return cluster_viz

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
                pos_update_msg = self.create_arrow_msg(
                    stamp=stamp,
                    frame_id=frame_id,
                    ns=f"Instance {instance_cluster.get_name()} {inst_id}",
                    marker_id=instance_cluster.cluster_id,
                    start=prev_pos - np.array([0, 0, offset_z]),
                    end=new_pos,
                    scale=[0.05, 0.1, 0.1],
                )
                position_updates_msg.markers.append(pos_update_msg)
        return position_updates_msg

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

    def create_cube_msg(
        self,
        stamp: rospy.Time,
        frame_id: str,
        ns: str,
        marker_id: int,
        pos: np.ndarray,
        color: List,
    ) -> Marker:
        cube_msg = Marker()
        cube_msg.header.stamp = stamp
        cube_msg.header.frame_id = frame_id
        cube_msg.ns = ns
        cube_msg.id = marker_id
        cube_msg.type = Marker.CUBE
        cube_msg.action = Marker.ADD
        cube_msg.pose.position.x = pos[0]
        cube_msg.pose.position.y = pos[1]
        cube_msg.pose.position.z = pos[2]
        cube_msg.pose.orientation.x = 0.0
        cube_msg.pose.orientation.y = 0.0
        cube_msg.pose.orientation.z = 0.0
        cube_msg.pose.orientation.w = 1.0
        cube_msg.scale.x = 0.2
        cube_msg.scale.y = 0.2
        cube_msg.scale.z = 0.2
        cube_msg.color.r = color[0]
        cube_msg.color.g = color[1]
        cube_msg.color.b = color[2]
        cube_msg.color.a = 0.7
        return cube_msg

    def create_arrow_msg(
        self,
        stamp: rospy.Time,
        frame_id: str,
        ns: str,
        marker_id: int,
        start: np.ndarray,
        end: np.ndarray,
        scale: List = [0.05, 0.1, 0.1],
    ):

        arrow_msg = Marker()
        arrow_msg.header.stamp = stamp
        arrow_msg.header.frame_id = frame_id
        arrow_msg.ns = ns
        arrow_msg.id = marker_id
        arrow_msg.type = Marker.ARROW
        arrow_msg.action = Marker.ADD

        start_point = Point()
        start_point.x = start[0]
        start_point.y = start[1]
        start_point.z = start[2]

        end_point = Point()
        end_point.x = end[0]
        end_point.y = end[1]
        end_point.z = end[2]

        arrow_msg.points = []
        arrow_msg.points.append(start_point)
        arrow_msg.points.append(end_point)

        # Set arrow properties (color, scale)
        arrow_msg.scale.x = scale[0]  # Shaft diameter
        arrow_msg.scale.y = scale[1]  # Head diameter
        arrow_msg.scale.z = scale[2]  # Head length

        arrow_msg.color.r = 0.0
        arrow_msg.color.g = 0.0
        arrow_msg.color.b = 0.0
        arrow_msg.color.a = 1.0  # Alpha (transparency)
        return arrow_msg

    def create_text_msg(
        self,
        stamp: rospy.Time,
        frame_id: str,
        ns: str,
        id: int,
        text: str,
        pos: np.ndarray,
    ) -> Marker:

        text_msg = Marker()
        text_msg.header.stamp = stamp
        text_msg.header.frame_id = frame_id
        text_msg.ns = ns
        text_msg.id = id
        text_msg.type = Marker.TEXT_VIEW_FACING
        text_msg.text = text
        text_msg.pose.position.x = pos[0]
        text_msg.pose.position.y = pos[1]
        # Move them down to easily visualize them
        text_msg.pose.position.z = pos[2]
        text_msg.scale.z = 0.6
        text_msg.color.r = 0
        text_msg.color.g = 0
        text_msg.color.b = 0
        text_msg.color.a = 1
        return text_msg
