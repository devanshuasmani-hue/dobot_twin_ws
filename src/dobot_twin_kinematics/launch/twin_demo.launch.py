"""One-command Phase 2 demo: RViz + simulated twin + kinematics controller + control panel.

  ros2 launch dobot_twin_kinematics twin_demo.launch.py
  ros2 launch dobot_twin_kinematics twin_demo.launch.py limits_profile:=datasheet panel:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('dobot_twin_description'), 'launch', 'twin_model.launch.py'])),
        launch_arguments={'instance': 'sim', 'gui': 'false', 'rviz': LaunchConfiguration('rviz')}.items())

    controller = Node(
        package='dobot_twin_kinematics', executable='twin_sim_controller', namespace='sim',
        name='twin_sim_controller', output='screen',
        parameters=[{'limits_profile': ParameterValue(LaunchConfiguration('limits_profile'), value_type=str),
                     'frame_id': 'sim/magician_root_link',
                     'tcp_speed': LaunchConfiguration('tcp_speed'),
                     'joint_speed_deg': LaunchConfiguration('joint_speed_deg')}])

    panel = Node(package='dobot_twin_kinematics', executable='twin_control_panel', namespace='sim',
                 name='twin_control_panel', output='screen', condition=IfCondition(LaunchConfiguration('panel')))

    return LaunchDescription([
        DeclareLaunchArgument('limits_profile', default_value='conservative',
                              description='conservative | urdf | datasheet (see config/limits.yaml)'),
        DeclareLaunchArgument('tcp_speed', default_value='0.10', description='Cartesian line speed [m/s]'),
        DeclareLaunchArgument('joint_speed_deg', default_value='90.0', description='Joint speed limit [deg/s]'),
        DeclareLaunchArgument('panel', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        model, controller, panel,
    ])
