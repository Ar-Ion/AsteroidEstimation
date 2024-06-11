from setuptools import find_packages, setup
from itertools import chain
import os

package_name = 'slam'

def copy_dir(name):
    
    base_dir = os.path.join('slam', 'pyslam', name)
    
    for (dirpath, dirnames, files) in os.walk(base_dir):
        for f in files:
            yield (os.path.join('share', dirpath), [os.path.join(dirpath, f)])

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=list(chain(
        [
            ('share/ament_index/resource_index/packages',
                ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),
            ('share/' + package_name + '/pyslam', ['slam/pyslam/config.ini'])
        ], 
        copy_dir("thirdparty"),
        copy_dir("settings")
    )),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arion',
    maintainer_email='arionz@caltech.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node = slam.node:main'
        ],
    },
)
