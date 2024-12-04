# hw
pybulletProject

# Autonomous Navigation Simulation System using PyBullet

This project implements an autonomous navigation system for a robot using the PyBullet physics engine. The system is tested in two different environments: a simple scene (Scenario A) and a complex scene (Scenario B). The robot is equipped with sensors to perform path planning, obstacle avoidance, and emergency braking.

## Project Structure

- **`main.py`**: Contains the implementation for the simple scene (Scenario A). The robot performs navigation and obstacle avoidance in a static environment with minimal complexity.
- **`final.py`**: Contains the implementation for the complex scene (Scenario B). The robot performs navigation and obstacle avoidance in a dynamic environment with complex obstacle configurations.

## Requirements

To run this project, the following Python libraries are required:

- `pybullet`
- `opencv-python`
- `numpy`
- `matplotlib`
- `gym`

You can install these libraries using `pip`:

```bash
pip install pybullet opencv-python numpy matplotlib pybullet gym

