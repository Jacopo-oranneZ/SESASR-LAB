from setuptools import find_packages, setup

package_name = 'lab05_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='l0dz',
    maintainer_email='andrea.test004@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'avoid_obstacles = lab05_pkg.avoid_obstacles:main',
            'v2_avoid_obstacles = lab05_pkg.v2_avoid_obstacles:main'
        ],
    },
)
