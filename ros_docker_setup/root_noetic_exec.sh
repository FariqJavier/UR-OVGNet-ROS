#!/bin/bash 

xhost +local:docker             # RUN THIS ON LINUX DISPLAY                             

sudo docker exec -ti --user root ovgnet_noetic bash
