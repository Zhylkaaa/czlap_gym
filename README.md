# PyBullet_simulation

Repository contains python package called `czlap_the_robot` that aims to provide easy to use `pybullet` simulation of Czlap-Czlap robot

List of Contents:
* [envs](czlap_the_robot/envs) - module that defines OpenAI Gym environments for our robot
* [robot](czlap_the_robot/robot) - module that defines wrapper on our robot pybullet simulation
* [urdf](czlap_the_robot/urdf) - git submodule with `.urdf`,  `.xacro` and configuration files for simulation

Installation:

for now we recommend installing model in `dev` mode.

So, clone the repository and follow this steps:
```bash
$ cd pybullet_simulation
$ pip install -e .
```