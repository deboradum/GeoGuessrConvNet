#!/bin/bash
folder=~/Desktop/geoGuessrConvNet/WashingtonDC/
mkdir -p "$folder"

# Find the next available filename
i=1
while [[ -e "$folder/$i.png" ]]; do
    ((i++))
done
screencapture -R440,-880,640,640 "$folder/$i.png"
