#!/bin/bash

## Install dependancies:
##  + libedgetpu
##  + Vulkan
##  + OpenNI2
##
## TODO size estimate

# Check root

if [ "$EUID" -ne 0 ]; then # TODO remove
    echo "Please run as root."
    exit 1
fi

# upgrade
echo "Updating packages..."; sudo apt-get update -y
echo "Upgrading packages..."; sudo apt-get full-upgrade -y

if [[ $GITHUB_ACTIONS ]]; then
   CROSSTC="."
else
   CROSSTC="arm-linux-gnueabihf"
fi

# edgetpu dependencies
# + Test lib: https://github.com/google-coral/pycoral.git
# + Procedure: hard reboot without coral then hard reboot with coral
# + website: https://coral.ai/docs/accelerator/get-started/#3-run-a-model-on-the-edge-tpu
echo "Installing clang ..."; sudo apt-get install -y clang
echo "Installing curl ..."; sudo apt install -y curl

sudo apt-get update
sudo apt-get full-upgrade

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt update
sudo apt full-upgrade
sudo apt update

#sudo apt install -y libedgetpu1-max
#sudo apt install -y libedgetpu-dev

echo "Installing edgetpu library and header files ..."

# compile for edgetpu with --min_runtime_version 13
$(
    sudo rm /usr/lib/$CROSSTC/libedgetpu.so
    sudo rm /usr/lib/$CROSSTC/libedgetpu.so.1
    sudo rm /usr/lib/$CROSSTC/libedgetpu.so.1.0

    git clone https://github.com/google-coral/edgetpu
    cd edgetpu

    sudo cp libedgetpu/edgetpu.h /usr/include/edgetpu.h
    sudo cp libedgetpu/edgetpu_c.h /usr/include/edgetpu_c.h

    sudo cp libedgetpu/direct/armv7a/libedgetpu.so.* /usr/lib/$CROSSTC
    cd /usr/lib/$CROSSTC
    sudo ln libedgetpu.so.1.0 libedgetpu.so
)

# Vulkan Dependencies
echo "Installing Vulkan Dependencies ..."
sudo apt-get install -y libxcb-randr0-dev libxrandr-dev
sudo apt-get install -y libxcb-xinerama0-dev libxinerama-dev libxcursor-dev
sudo apt-get install -y libxcb-cursor-dev libxkbcommon-dev xutils-dev
sudo apt-get install -y xutils-dev libpthread-stubs0-dev libpciaccess-dev
sudo apt-get install -y libffi-dev x11proto-xext-dev libxcb1-dev libxcb-*dev
sudo apt-get install -y libssl-dev libgnutls28-dev x11proto-dri2-dev
sudo apt-get install -y x11proto-dri3-dev libx11-dev libxcb-glx0-dev
sudo apt-get install -y libx11-xcb-dev libxext-dev libxdamage-dev libxfixes-dev
sudo apt-get install -y libva-dev x11proto-randr-dev x11proto-present-dev
sudo apt-get install -y libclc-dev libelf-dev mesa-utils
sudo apt-get install -y libvulkan-dev libvulkan1 libassimp-dev
sudo apt-get install -y libdrm-dev libxshmfence-dev libxxf86vm-dev libunwind-dev
sudo apt-get install -y libwayland-dev wayland-protocols
sudo apt-get install -y libwayland-egl-backend-dev
sudo apt-get install -y valgrind libzstd-dev vulkan-tools
sudo apt-get install -y git build-essential bison flex ninja-build

VERSION_CODENAME=$(cat /etc/os-release | grep -o 'VERSION_CODENAME.*' | cut -f2- -d=)

if [[ $VERSION_CODENAME == 'buster' ]]; then
    sudo apt-get install -y python-mako vulkan-utils
elif [[ $VERSION_CODENAME == 'bullseye' ]]; then
    sudo apt-get install -y python3-mako
fi

echo "Installing Vulkan ..."
# remove old versions first
sudo rm -rf /home/pi/mesa_vulkan
# install meson
sudo apt purge meson -y
sudo pip3 install meson
# install mako
sudo pip3 install mako
# install v3dv
cd ~
git clone -b 20.3 https://gitlab.freedesktop.org/mesa/mesa.git mesa_vulkan
# build v3dv (± 30 min)
cd mesa_vulkan
CFLAGS="-mcpu=cortex-a72 -mfpu=neon-fp-armv8" \
CXXFLAGS="-mcpu=cortex-a72 -mfpu=neon-fp-armv8" \
meson --prefix /usr \
 -D platforms=x11 \
 -D vulkan-drivers=broadcom \
 -D dri-drivers= \
 -D gallium-drivers=kmsro,v3d,vc4 \
 -D buildtype=release build
sudo ninja -C build -j4
ninja -C build install
echo "TIP: Check your driver using \"glxinfo -B\""

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
if ! [[ $GITHUB_ACTIONS ]]; then
   find . -name Makefile -exec sed -i 's/CFLAGS += -Wall/CFLAGS += -Wall -mfloat-abi=hard/g' {} \;
fi

make
sudo ./Packaging/Linux/install.sh
cp ./Bin/*-Release/libOpenNI2.so /usr/lib/$CROSSTC/
cp -r ./Bin/*-Release/OpenNI2 /usr/lib/$CROSSTC/

# Adding current user to video group
echo "Add user to video group so they may access the camera (requires reboot)"
sudo usermod -a -G video pi

# Save env variables
sudo mkdir /lib/libOpenNI2
echo 'export OPENNI2_REDIST="/lib/libOpenNI2"' >> ~/.bashrc
echo 'export OPENNI2_REDIST64="/lib/libOpenNI2"' >> ~/.bashrc

echo "A reboot is required for some changes to take effect."

exit 0
