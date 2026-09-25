from setuptools import find_packages, setup

package_name = 'inmoov_memory'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artur',
    maintainer_email='fedjukevitsh@gmail.com',
    description='Social memory node for InMoov robot (SQLite + ROS2 service)',
    license='GPL-3.0-only',
    entry_points={
        'console_scripts': [
            'memory_node = inmoov_memory.memory_node:main',
        ],
    },
)
