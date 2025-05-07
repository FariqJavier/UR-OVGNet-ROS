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
$ realsense-viewer
```

Testing camera depth

```sh
$ rs-depth-quality
```

Find supported image resolution

```sh
$ rs-enumerate-device
```

##  Run Graspnet ROS Package

Run Realsense custom node:
(Resolution: 848x480, Align Depth: Enabled, Sync: Enabled, Allow No Texture Points: Enabled, PointCloud Ordered: Enabled)

```sh
$ roslaunch graspnet-ros custom_rs_camera.launch 
```

Run image saver:
(Save colored image, depth image in meter, dan camera info)

```sh
$ roslaunch graspnet-ros image_saver.launch
```

Run workspace mask:
(Adjust the mask size according to input image resolution)

```sh
$ roslaunch graspnet-ros generate_workspace_mask.launch
```

Run Graspnet Inference on input image
(Adjust checkpoint model path, image path, image size, and depth format)

```sh
$ roslaunch graspnet-ros ros_oneshot_demo.launch
```

### NOTES

INPUT CAMERA SIZE: [848,480]

TARGET HARUS TERLENTANG, DAN TINGGI OBJEK SAAT TERLENTANG MAKS 10 CM (LEBIH DARI ITU, REALSENSE TIDAK BISA MENDETEKSI DEPTH)

TODO:

REMOVE UNNECESSARY FOLDER (need to search any files with keyword "/home/lm" or "/media/lm/")

PROBLEM:

(Jika GroundingDino sulit menghasilkan inference:)
Grounding Dino punya AP (Average Precision) untuk dataset COCO sebesar 52,5% dan untuk dataset ODinW sebesar 26,1%, dimana dataset coco lebih terstruktur dengan kategori yang sudah ditentukan sedangkan odinw lebih menantang karena dataset yang dibuat untuk open-set objek detection yang belum pernal dikenal sebelumnya.
karena itu kemungkinan besar deteksi objek tidak akan berhasil jika hanya sati kali input
Solusinya akan diambil input setiap 0,1 detik selama 1 detik (10 x input) sehingga setidaknya deteksi objek bisa menghasilkan hasil deteksinya 