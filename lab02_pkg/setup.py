from setuptools import find_packages, setup

package_name = 'lab02_pkg'

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
<<<<<<< HEAD
    maintainer='l0dz',
    maintainer_email='andrea.test004@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
=======
    maintainer='jacopo',
    maintainer_email='jacopo.zennaro.1@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
>>>>>>> main
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
<<<<<<< HEAD
            'controller = lab02_pkg.controller:main'
=======
>>>>>>> main
        ],
    },
)
