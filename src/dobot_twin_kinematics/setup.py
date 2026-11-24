from glob import glob

from setuptools import setup

package_name = 'dobot_twin_kinematics'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/docs', glob('docs/*.md')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dobot Twin',
    maintainer_email='u23me181@med.svnit.ac.in',
    description='Kinematics, limits and simulated motion control for the Dobot Magician digital twin',
    license='BSD',
    entry_points={
        'console_scripts': [
            'twin_sim_controller = dobot_twin_kinematics.sim_node:main',
            'twin_control_panel = dobot_twin_kinematics.panel_node:main',
        ],
    },
)
