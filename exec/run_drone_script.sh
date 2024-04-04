#!/bin/bash

# Get the tilt angle from the first command line argument
tilt_angle=$1

# Get the current date and time in the required format
datetime=$(date +"%m%d%H%M%Y.%S")

# Connect to ADB and set the date on the Android device
adb shell "su -c 'date $datetime'"
echo "Date set on Android device: $datetime"

# Execute the remote script on the Android device, passing the tilt angle
adb shell "su -c 'sh /run_data_script.sh $tilt_angle'"

# Optional: Display a message
echo "Executed remote_script.sh on Android device with tilt angle: $tilt_angle"
