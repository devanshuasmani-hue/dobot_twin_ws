#!/usr/bin/env python3
"""Floating text labels so the simulated and the physical twin can never be confused in RViz.

Publishes visualization_msgs/MarkerArray on /twin/labels. Each instance label can be
extended with a status string received on /<namespace>/twin_status (std_msgs/String),
e.g. 'CONNECTED' / 'NO DATA' (used by the physical-robot bridge in Phase 3).
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

DEFAULTS = {
    'sim': ('SIMULATION', (0.20, 0.55, 1.00)),
    'real': ('PHYSICAL ROBOT', (1.00, 0.55, 0.10)),
}


class InstanceLabels(Node):
    def __init__(self):
        super().__init__('instance_labels')
        self.declare_parameter('instances', ['sim'])
        self.declare_parameter('root_link', 'magician_root_link')
        self.declare_parameter('height', 0.42)
        self.instances = list(self.get_parameter('instances').value)
        self.status = {i: '' for i in self.instances}
        for i in self.instances:
            self.create_subscription(String, f'/{i}/twin_status', lambda m, i=i: self.status.__setitem__(i, m.data), 10)
        self.pub = self.create_publisher(MarkerArray, '/twin/labels', 1)
        self.create_timer(0.5, self.tick)

    def tick(self):
        arr = MarkerArray()
        for n, i in enumerate(self.instances):
            text, rgb = DEFAULTS.get(i, (i.upper(), (1.0, 1.0, 1.0)))
            m = Marker()
            m.header.frame_id = f'{i}/{self.get_parameter("root_link").value}'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = 'twin_label', n
            m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            m.pose.position.z = float(self.get_parameter('height').value)
            m.pose.orientation.w = 1.0
            m.scale.z = 0.035
            m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 1.0)
            m.text = text + (f'\n[{self.status[i]}]' if self.status[i] else '')
            arr.markers.append(m)
        self.pub.publish(arr)


def main():
    rclpy.init()
    rclpy.spin(InstanceLabels())


if __name__ == '__main__':
    main()
