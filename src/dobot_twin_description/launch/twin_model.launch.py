"""Dobot Magician digital twin - model layer (Phase 1).

Starts one or two *strictly separated* robot-model instances built from the unmodified
dobot_description xacro (4-DOF + gripper):

  /sim/...   simulated twin    TF frames  sim/<link>     joint states  /sim/joint_states
  /real/...  physical twin     TF frames  real/<link>    joint states  /real/joint_states

  ros2 launch dobot_twin_description twin_model.launch.py                   # sim + slider GUI + RViz
  ros2 launch dobot_twin_description twin_model.launch.py instance:=both    # sim and real side by side
  ros2 launch dobot_twin_description twin_model.launch.py instance:=both overlay:=true   # co-located

Without a joint-state source the movable joints are not posed: use gui:=true (default for sim)
or the Phase 2 / Phase 3 nodes.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

WORLD = 'world'
ROOT = 'magician_root_link'


def _instances(context):
    which = LaunchConfiguration('instance').perform(context)
    return ['sim', 'real'] if which == 'both' else [which]


def _setup(context):
    insts = _instances(context)
    overlay = LaunchConfiguration('overlay').perform(context).lower() == 'true'
    spacing = float(LaunchConfiguration('spacing').perform(context))
    gui = LaunchConfiguration('gui').perform(context).lower() == 'true'
    model = LaunchConfiguration('model')
    actions = []

    for n, ns in enumerate(insts):
        # side by side (sim left, real right) unless overlaid
        y = 0.0 if (overlay or len(insts) == 1) else (spacing / 2 if ns == 'sim' else -spacing / 2)
        description = ParameterValue(
            Command([FindExecutable(name='xacro'), ' ', model,
                     ' DOF:=4', ' tool:=gripper', ' use_camera:=false']), value_type=str)

        actions.append(Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            namespace=ns, name='robot_state_publisher', output='screen',
            parameters=[{'robot_description': description, 'frame_prefix': f'{ns}/'}]))

        actions.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'{ns}_world_anchor',
            arguments=['--x', '0', '--y', str(y), '--z', '0',
                       '--yaw', '0', '--pitch', '0', '--roll', '0',
                       '--frame-id', WORLD, '--child-frame-id', f'{ns}/{ROOT}']))

        if ns == 'sim' and gui:
            actions.append(Node(
                package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
                namespace=ns, name='joint_state_publisher_gui', output='screen'))

    actions.append(Node(
        package='dobot_twin_description', executable='instance_labels.py',
        name='instance_labels', output='screen',
        parameters=[{'instances': insts, 'root_link': ROOT}]))
    return actions


def generate_launch_description():
    share = FindPackageShare('dobot_twin_description')
    return LaunchDescription([
        DeclareLaunchArgument('instance', default_value='sim', choices=['sim', 'real', 'both'],
                              description='Which twin instance(s) to show.'),
        DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false'],
                              description='Slider GUI for the sim instance (Phase 1 manual posing).'),
        DeclareLaunchArgument('overlay', default_value='false', choices=['true', 'false'],
                              description='Co-locate sim and real instead of placing them side by side.'),
        DeclareLaunchArgument('spacing', default_value='0.45', description='Side-by-side spacing [m].'),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rvizconfig', default_value=PathJoinSubstitution([share, 'rviz', 'twin.rviz'])),
        DeclareLaunchArgument('model', default_value=PathJoinSubstitution(
            [FindPackageShare('dobot_description'), 'model', 'magician_standalone.urdf.xacro']),
            description='Original (unmodified) dobot_description xacro.'),
        OpaqueFunction(function=_setup),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen',
             arguments=['-d', LaunchConfiguration('rvizconfig')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
