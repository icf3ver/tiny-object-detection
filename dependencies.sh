#!/bin/bash
## Install dependancies:
##  + libedgetpu
##  + Vulakan
##  + OpenNI2
##
## TODO size estimate

# Check root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root."
    exit 1
fi

# upgrade
echo "Updating packages..."; apt-get update
echo "Upgrading packages..."; apt-get full-upgrade

# edgetpu dependencies
# + Test lib: https://github.com/google-coral/pycoral.git
# + Procedure: hard reboot without coral then hard reboot with coral
# + website: https://coral.ai/docs/accelerator/get-started/#3-run-a-model-on-the-edge-tpu
echo "Installing clang ..."; apt-get install -y clang

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update

sudo apt install -y libedgetpu-std
sudo apt install -y libedgetpu-dev

echo "Installing edgetpu library and header files ..."
git clone https://coral.googlesource.com/edgetpu
cd edgetpu
git checkout release-chef
# TODO clean

cp libedgetpu/libedgetpu_arm64_throttled.so /usr/lib/*-linux-gnu*/libedgetpu.so >> /dev/null
cp libedgetpu/edgetpu.h /usr/include/edgetpu.h >> /dev/null
cd ..

# vulkan dependencies
echo "Installing Vulkan Packages ..."
apt-get install -y libxcb-randr0-dev libxrandr-dev
apt-get install -y libxcb-xinerama0-dev libxinerama-dev libxcursor-dev
apt-get install -y libxcb-cursor-dev libxkbcommon-dev xutils-dev
apt-get install -y xutils-dev libpthread-stubs0-dev libpciaccess-dev
apt-get install -y libffi-dev x11proto-xext-dev libxcb1-dev libxcb-*dev
apt-get install -y libssl-dev libgnutls28-dev x11proto-dri2-dev
apt-get install -y x11proto-dri3-dev libx11-dev libxcb-glx0-dev
apt-get install -y libx11-xcb-dev libxext-dev libxdamage-dev libxfixes-dev
apt-get install -y libva-dev x11proto-randr-dev x11proto-present-dev
apt-get install -y libclc-dev libelf-dev mesa-utils
apt-get install -y libvulkan-dev libvulkan1 libassimp-dev
apt-get install -y libdrm-dev libxshmfence-dev libxxf86vm-dev libunwind-dev
apt-get install -y libwayland-dev wayland-protocols
apt-get install -y libwayland-egl-backend-dev
apt-get install -y valgrind libzstd-dev vulkan-tools
apt-get install -y git build-essential bison flex ninja-build

VERSION_CODENAME=$(cat /etc/os-release | grep -o 'VERSION_CODENAME.*' | cut -f2- -d=)

if [[ $VERSION_CODENAME == 'buster' ]]; then
    apt-get install -y python-mako vulkan-utils
elif [[ $VERSION_CODENAME == 'bullseye' ]]; then
    apt-get install -y python3-mako
fi

# OpenNI2 dependencies
sudo apt-get install -y libusb-1.0-0-dev
sudo apt-get install -y libudev-dev
sudo apt-get install -y freeglut3-dev
sudo apt-get install -y doxygen
sudo apt-get install -y graphviz

# install Java
sudo apt install -y openjdk-8-jdk

# install cmake
sudo apt install cmake

# Build for Arm
git clone https://github.com/occipital/OpenNI2
cd OpenNI2
find . -name Makefile -exec sed -i 's/CFLAGS += -Wall/CFLAGS += -Wall -mfloat-abi=hard/g' {} \;

make
sudo ./Packaging/Linux/install.sh

# Adding current user to video group
echo "Add user to video group so they may access the camera (requires reboot)"
sudo usermod -a -G video pi

# Save env variables
sudo mkdir /lib/libOpenNI2
echo "export OPENNI2_REDIST /lib/libOpenNI2" >> ~/.bashrc

echo "A reboot is required for some changes to take effect."

exit 0
