from setuptools import setup

package_name = 'hand_signal_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nels',
    maintainer_email='kornelialukojc@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
              'hand_signal_detector = hand_signal_detector.hand_signal_detector:main',
              'hand_gesture_processor = hand_signal_detector.hand_gesture_processor:main',
        ],
    },
)
