from setuptools import find_packages
from distutils.core import setup

setup(
    name='klavier_rl_gym',
    version='1.0.0',
    author='Jimeng Xu',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='Template RL environments for robots',
    install_requires=['isaacgym',  # preview4
                      'wandb',
                      'tensorboard',
                      'tqdm',
                      'numpy==1.23.5',
                      'opencv-python',
                      'mujoco==3.1.5',
                      'mujoco-python-viewer',
                      'matplotlib',
                      'pybullet',
                      'pynput']
)
