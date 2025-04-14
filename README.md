### UR-OVGNet-ROS

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
