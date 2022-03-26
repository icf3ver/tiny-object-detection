# Tiny Object Detection
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![build](https://github.com/littleTitan/tiny-object-detection/actions/workflows/rust.yml/badge.svg)](https://github.com/littleTitan/tiny-object-detection/actions/workflows/rust.yml)

Pi edition. Not production yet.

# Requirements
Pi 8GB (Tested on Pi 4 Model B 8GB)

OpenNI2 capable device capable of both color and depth streams:
 + Hacked Xbox Kinect
 + Carmine 1.09 or 1.08 (Adjust image input size 320 x 240)
 + Intel RealSense (Tested on) (Dependencies not yet included in dependencies.sh)

Edgetpu: Google Coral USB (Testing on)

# Dependencies
Simply execute the `dependencies.sh` batch file

# Wiring (Tested)
USB 3.0 &nbsp; Coral TPU <br/>
USB 3.0 &nbsp; Kinect OpenNI2 capable device

Pi ICE Tower CPU Cooling Fan <br/>
Pi 4 Model B Heat Sink Cooling Fan mounted on coral tpu. 

Note: Use unthrottled edgetpu runtime at your own risk.
Note: I am not libable for any of your damages. (Use at your own risk)

# License
[Apache 2.0](LICENSE.md)
