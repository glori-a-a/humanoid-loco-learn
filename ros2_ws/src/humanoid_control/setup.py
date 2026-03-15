from setuptools import setup

setup(
    name='humanoid_control',
    version='1.0.0',
    packages=['humanoid_control'],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'ilc_node = humanoid_control.ilc_node:main',
        ],
    },
)
