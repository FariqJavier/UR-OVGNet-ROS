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

### NOTES

INPUT CAMERA SIZE: [1280,720] (HARDCODED SIZE AND FIXED BASED ON WORKSPACE SIZE [1280,720])

TODO:

REMOVE UNNECESSARY FOLDER (need to search any files with keyword "/home/lm" or "/media/lm/")

PROBLEM:

Dependency conflict especially using compatibility of CUDA and Torch on the project
