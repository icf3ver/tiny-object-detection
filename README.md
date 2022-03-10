# Tiny Object Detection
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![build](https://github.com/ultraviolet8567/tiny-object-detection/actions/workflows/rust.yml/badge.svg)](https://github.com/ultraviolet8567/tiny-object-detection/actions/workflows/rust.yml)

Pi edition. Not production yet.

# Requirements
Pi 8GB (Tested on Pi 4 Model B 8GB)
OpenNI2 capable device capable of both color and depth streams:
 + Hacked Xbox Kinect
 + Carmine 1.09 or 1.08 (Adjust image input size 320 x 240)
 + Intel RealSense (Tested on)

Edgetpu:
 + Google Coral (Testing on)

# Execution
Run with 2GB swapspace (Will be necessary later) </br>
Run with 4GB stacksize for speedy image processing
```RUSTFLAGS="-C link-args=-Wl,-zstack-size=4194304"```
In release mode to avoid a Segmentation fault caused by 
failure to release 2GBs of buffer when dropped before 
reallocation.

# Dependencies
Simply execute the `dependencies.sh` batch file

# Wiring
USB 3.0 &nbsp; Coral TPU <br/>
USB 3.0 &nbsp; Kinect OpenNI2 capable device

# License
[Apache 2.0](LICENSE.md)
