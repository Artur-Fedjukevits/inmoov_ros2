from setuptools import find_packages, setup

package_name = 'inmoov_voice'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/voice.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artur',
    maintainer_email='fedjukevitsh@gmail.com',
    description='Voice pipeline for the InMoov robot: wake word, VAD, STT, sound localization, voice emotion, TTS',
    license='GPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'audio_source_node    = inmoov_voice.audio_source_node:main',
            'wakeword_node        = inmoov_voice.openwakeword_node:main',
            'voice_detector_node  = inmoov_voice.voice_detector_node:main',
            'parakeet_stt_node    = inmoov_voice.parakeet_stt_node:main',
            'tts_node             = inmoov_voice.tts_node:main',
'voice_emotion_node   = inmoov_voice.voice_emotion_node:main',
            'sound_localization_node = inmoov_voice.sound_localization_node:main',
            'diagnose             = scripts.diagnose:main',
        ],
    },
)
