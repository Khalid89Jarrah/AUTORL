#!/bin/bash
T=$1
sleep 40
while true; do
  gz topic -t /world/default/wrench/persistent -m gz.msgs.EntityWrench -p "entity: {name: 'x500::base_link', type: LINK}, wrench: {torque: {x: $T, y: $T, z: 0.0}}" >/dev/null 2>&1
  sleep 1
done
