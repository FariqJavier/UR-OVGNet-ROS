#!/bin/bash 

xhost +local:docker
# DISPLAY=:0 xhost +

docker exec -it ovgnet_noetic bash
