#!/bin/bash

colcon build
source install/setup.bash
ros2 run feature_extractor main