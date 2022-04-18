# Tiny Object Detection
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![build](https://github.com/littleTitan/tiny-object-detection/actions/workflows/rust.yml/badge.svg)](https://github.com/littleTitan/tiny-object-detection/actions/workflows/rust.yml)

Pi edition. Not production yet.

# Competition version
Unfortunately I misconfigured the train and am only just barely able to trust the masks and model as an instance segmentation model. There is no way I will be able to repair this in the time I have left so instead I have tried to work around it for the time being.

Make sure to `export DISPLAY=:0` if there is no screen.
Bug: run without glsl shader and then with to get it to work ?!

# Requirements
Pi 8GB (Tested on Pi 4 Model B 8GB)

OpenNI2 capable device capable of both color and depth streams:
 + Hacked Xbox Kinect
 + Carmine 1.09 or 1.08 (Adjust image input size 320 x 240)
 + Intel RealSense (Tested on) (Dependencies not yet included in dependencies.sh)

Edgetpu: Google Coral USB (Testing on)

# Dependencies
Simply execute the `dependencies.sh` batch file

Make sure to set
```
framebuffer_width=1080
framebuffer_height=720
hdmi_force_hotplug=1
hdmi_group=1
hdmi_mode=16
hdmi_drive=2
```
in `/boot/config.txt`.

# Wiring (Tested)
USB 3.0 &nbsp; Coral TPU <br/>
USB 3.0 &nbsp; Kinect OpenNI2 capable device

Pi ICE Tower CPU Cooling Fan <br/>
Pi 4 Model B Heat Sink Cooling Fan mounted on coral tpu. 

Note: Use unthrottled edgetpu runtime at your own risk.
Note: I am not liable for any of your damages. (Use at your own risk)

# License
[Apache 2.0](LICENSE.md)
