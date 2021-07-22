import os
from setuptools import setup

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

setup(name='rl_tools',
      version='0.1',
      description='Custom rl tools',
      author='Philippe Proctor',
      package_dir={"": "rl_tools"},
      packages=["rl_tools"],
      install_requires=[
        f'gym_rad_search @ file://localhost{PKG_DIR}/gym_rad_search/'
      ],
      python_requires=">=3.6",
      license='MIT',
      zip_safe=False)