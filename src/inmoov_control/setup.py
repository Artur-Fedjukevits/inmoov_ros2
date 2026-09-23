from setuptools import setup

package_name = 'inmoov_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_data={package_name: ['face_expressions_calibration.json']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='Artur Fedjukevits',
    maintainer_email='fedjukevitsh@gmail.com',
    description='InMoov motion control: batch serial protocol replacing xicro for servo control.',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arduino_right_node         = inmoov_control.arduino_right_node:main',
            'arduino_left_node          = inmoov_control.arduino_left_node:main',
            'joint_state_publisher      = inmoov_control.joint_state_publisher:main',
            'face_expressions_node      = inmoov_control.face_expressions_node:main',
            'face_expression_calibrator = inmoov_control.face_expression_calibrator:main',
        ],
    },
)
