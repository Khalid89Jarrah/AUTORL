#!/bin/bash
cd /opt/autorl_ws

MAKEFLAGS="-j $(( $(nproc) / 2 ))" colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source /opt/autorl_ws/install/setup.bash
