from setuptools import find_packages, setup

package_name = 'inmoov_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artur',
    maintainer_email='fedjukevitsh@gmail.com',
    description='Vision pipeline for the InMoov robot: dual-eye face detection/tracking/recognition, emotion, OAK-D',
    license='GPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'face_capture_node      = inmoov_vision.face_capture_node:main',
            'face_detection_node    = inmoov_vision.face_detection_node:main',
            'face_tracker_node      = inmoov_vision.face_tracker_node:main',
            'face_recognition_node  = inmoov_vision.face_recognition_node:main',
            'face_gallery_node      = inmoov_vision.face_gallery_node:main',
            'emotion_recognition_node = inmoov_vision.emotion_recognition_node:main',
            'vision_head_tracker_node = inmoov_vision.vision_head_tracker_node:main',
            'oak_node                = inmoov_vision.oak_node:main',
            'human_detection_node   = inmoov_vision.human_detection_node:main',
            'scene_manager_node     = inmoov_vision.scene_manager_node:main',
        ],
    },
)
