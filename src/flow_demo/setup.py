from setuptools import setup

package_name = 'flow_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Flow demo nodes',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dummy_image_pub = flow_demo.dummy_image_pub:main',
            'frame_replay = flow_demo.frame_replay:main',

        ],
    },
)
