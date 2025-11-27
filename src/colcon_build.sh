#!/bin/bash
cd /opt/quadrl_ws

MAKEFLAGS="-j $(( $(nproc) / 2 ))" colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source /opt/quadrl_ws/install/setup.bash
