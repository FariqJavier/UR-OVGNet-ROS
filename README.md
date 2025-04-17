### UR-OVGNet-ROS

## Requirements !!!

# Add Assets

```bash
$ mv assets workspace/noetic_src/OVGNet/
```

# Add Checkpoint (OVGANet, GroundingDino, Graspnet)

```bash
$ mv checkpoint workspace/noetic_src/OVGNet/
```

```bash
$ mv checkpoint workspace/noetic_src/graspnet-ros/src/logs/logs_rs/
```

# Add OVGrasping

```bash
$ mv ovgrasping workspace/noetic_src/OVGNet/test_vg/datasets/
```

# Add Tolerance

```bash
$ mv tolerance workspace/noetic_src/OVGNet/graspnet/graspnet/dataset/
```

```bash
$ mv tolerance workspace/noetic_src/graspnet-ros/src/dataset/
```

## Build Docker Images

```bash
$ sudo docker compose build
```

## Start Docker Containers

```bash
$ sudo docker compose up -d
```

## Run Existing Containers

```bash
$ bash noetic_exec.sh
```

## Set up user for accessing realsense camera

```bash
$ sudo wget -O /etc/udev/rules.d/99-realsense-libusb.rules https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
$ sudo udevadm control --reload-rules
$ sudo udevadm trigger
$ sudo usermod -a -G plugdev $USERNAME
```

##  Run Intel RealSense SDK 2.0

For testing if camera is connected or not

```sh
$ sudo realsense-viewer
```

Or run as ROOT user on the container

```sh
$ docker exec -ti --user root ros_noetic bash
$ source devel/setup.bash
```

```sh
$ realsense-viewer
```

```sh
$ roslaunch realsense2_camera rs_camera.launch
```

### NOTES

INPUT CAMERA SIZE: [1280,720] (HARDCODED SIZE AND FIXED BASED ON WORKSPACE SIZE [1280,720])

TODO:

REMOVE UNNECESSARY FOLDER (need to search any files with keyword "/home/lm" or "/media/lm/")

PROBLEM:

Dependency conflict especially using compatibility of CUDA and Torch on the project
