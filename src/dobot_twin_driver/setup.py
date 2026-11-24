from glob import glob

from setuptools import setup

package_name = 'dobot_twin_driver'

setup(
    name=package_name,
    version='0.3.0',
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
    description='Physical Dobot Magician bridge for the digital twin (serial protocol, guarded motion, calibration)',
    license='BSD',
    entry_points={
        'console_scripts': [
            'dobot_bridge = dobot_twin_driver.bridge_node:main',
            'twin_real_panel = dobot_twin_driver.real_panel_node:main',
            'twin_calibrate = dobot_twin_driver.calibrate_cli:main',
        ],
    },
)
