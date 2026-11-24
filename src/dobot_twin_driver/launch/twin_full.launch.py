"""Simulated twin + physical twin side by side, with the physical-robot bridge and both control panels.

  ros2 launch dobot_twin_driver twin_full.launch.py                        # real robot (port auto-detected)
  ros2 launch dobot_twin_driver twin_full.launch.py port:=/dev/ttyUSB0
  ros2 launch dobot_twin_driver twin_full.launch.py fake:=true             # no hardware: built-in simulated Dobot
  ... overlay:=true   # draw both on top of each other (sim translucent)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('dobot_twin_driver'), 'config', 'driver.yaml')

    model = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('dobot_twin_description'), 'launch', 'twin_model.launch.py'])),
        launch_arguments={'instance': 'both', 'gui': 'false', 'overlay': LaunchConfiguration('overlay'),
                          'rviz': LaunchConfiguration('rviz')}.items())

    sim = Node(package='dobot_twin_kinematics', executable='twin_sim_controller', namespace='sim',
               name='twin_sim_controller', output='screen',
               parameters=[{'limits_profile': ParameterValue(LaunchConfiguration('limits_profile'), value_type=str),
                            'frame_id': 'sim/magician_root_link'}])
    sim_panel = Node(package='dobot_twin_kinematics', executable='twin_control_panel', namespace='sim',
                     name='twin_control_panel', output='screen', condition=IfCondition(LaunchConfiguration('panel')))

    bridge = Node(package='dobot_twin_driver', executable='dobot_bridge', namespace='real', name='dobot_bridge',
                  output='screen',
                  parameters=[cfg, {'port': ParameterValue(LaunchConfiguration('port'), value_type=str),
                                    'fake': ParameterValue(LaunchConfiguration('fake'), value_type=bool),
                                    'limits_profile': ParameterValue(LaunchConfiguration('limits_profile'), value_type=str),
                                    'frame_id': 'real/magician_root_link'}])
    real_panel = Node(package='dobot_twin_driver', executable='twin_real_panel', namespace='real',
                      name='twin_real_panel', output='screen', condition=IfCondition(LaunchConfiguration('panel')))

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='', description='serial port; empty = auto-detect'),
        DeclareLaunchArgument('fake', default_value='false', choices=['true', 'false'],
                              description='use the built-in simulated Dobot instead of hardware'),
        DeclareLaunchArgument('limits_profile', default_value='conservative'),
        DeclareLaunchArgument('overlay', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('panel', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        model, sim, sim_panel, bridge, real_panel,
    ])
