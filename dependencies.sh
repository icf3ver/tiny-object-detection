#!/bin/bash

## Install dependancies:
##  + libedgetpu
##  + Vulakan
##
## TODO size estimate

# Check for root privlages
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root."
    exit 1
fi

# check for updates
echo "Updating packages..."; apt-get update
echo "Upgrading packages..."; apt-get upgrade

# fetch and build libedgetpu
echo "Installing edgetpu library and header files ..."
git clone https://coral.googlesource.com/edgetpu
cd edgetpu
git checkout release-chef
cp libedgetpu/libedgetpu_arm64_throttled.so /usr/lib/*-linux-gnu/libedgetpu.so >> /dev/null
cp libedgetpu/edgetpu.h /usr/include/edgetpu.h >> /dev/null
cd ..

# install vulkan dependencies
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

exit 0
