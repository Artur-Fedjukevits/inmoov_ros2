from setuptools import find_packages, setup

package_name = 'inmoov_cognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/behavior_manager.launch.py',
            'launch/inmoov_full.launch.py',
            'launch/telegram_bridge.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/telegram_params.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artur',
    maintainer_email='fedjukevitsh@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'behavior_manager_node = inmoov_cognition.behavior_manager_node:main',
            'identity_manager_node = inmoov_cognition.identity_manager_node:main',
            'llm_node              = inmoov_cognition.llm_node:main',
            'openhab_bridge_node   = inmoov_cognition.openhab_bridge_node:main',
            'telegram_bridge_node  = inmoov_cognition.telegram_bridge_node:main',
        ],
    },
)
