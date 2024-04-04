#!/bin/bash
# Connect to ADB and execute the commands to unmount, remove, create, and remount the USB drive
adb shell "su -c '\
if mount | grep /mnt/usbdrive > /dev/null; then \
    umount -l /mnt/usbdrive; \
fi; \
umount /dev/sdg1 2>/dev/null; \
rm -rf /mnt/usbdrive; \
mkdir -p /mnt/usbdrive; \
mount /dev/sdg1 /mnt/usbdrive'"

# Run the Docker container inside VOXL using adb shell
adb shell "docker run --device=/dev/bus/usb/002/003:/dev/bus/usb/002/003 -v /mnt/usbdrive/event:/usr/src/app/metadata/event -it voxlevent"
echo "Docker container voxlevent started inside VOXL with USB device mapping and volume mount"
