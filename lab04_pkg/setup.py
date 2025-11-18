from setuptools import find_packages, setup

package_name = 'lab04_pkg'

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
    maintainer='filo',
    maintainer_email='pietro.filomeno@studenti.polito.it',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'EKFros = lab04_pkg.EKFros:main',
            'utils = lab04_pkg.utils:main',
            'ekf = lab04_pkg.ekf:main',
            'TASK0_mot = lab04_pkg.TASK0_mot:main',
            'TASK0_sens = lab04_pkg.TASK0_sens:main'
        ],
    },
)
