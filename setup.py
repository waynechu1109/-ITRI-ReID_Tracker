from setuptools import setup
import glob

package_name = 'reid_tracker'
launch_files = glob.glob('launch/*.launch.py')

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', launch_files),
    ],
    install_requires=['setuptools', 'gdown',  'torchreid'],
    zip_safe=True,
    maintainer='Brian',
    maintainer_email='brain841102@gmail.com',
    description='ROS 2 Person ReID Node',
    license='MIT',
    extras_require={                        
        "test": ["pytest", "mock"]
    },
    entry_points={
        'console_scripts': [
            'person_reid_node = reid_tracker.person_reid_node:main',
            'person_reid_batch_node = reid_tracker.person_reid_node_batch:main'
        ],
    },
)