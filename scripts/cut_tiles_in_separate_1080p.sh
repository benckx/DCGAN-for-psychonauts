#!/usr/bin/env bash

# view1 (3x3)
mkdir view1
for f in $(ls *.png); do
 echo $f
 convert $f -crop 1920x1080+0+0 view1/$f
done

ffmpeg -framerate 60 -f image2 -i 'view1/%*.png' -c:v libx264 -profile:v high -crf 16 -pix_fmt yuv420p cartoon56_time_cut0006_box0001.mp4
rm -rf view1

# view2 (3x3)
mkdir view2
for f in $(ls *.png); do
 echo $f
 convert $f -crop 1920x1080+1920+0 view2/$f
done

ffmpeg -framerate 60 -f image2 -i 'view2/%*.png' -c:v libx264 -profile:v high -crf 16 -pix_fmt yuv420p cartoon56_time_cut0006_box0002.mp4
rm -rf view2

# view 3 (4x4)
mkdir view3
for f in $(ls *.png); do
 echo $f
 convert $f -crop 1920x1080+3840+0 view3/$f
done

ffmpeg -framerate 60 -f image2 -i 'view3/%*.png' -c:v libx264 -profile:v high -crf 16 -pix_fmt yuv420p cartoon56_time_cut0006_box0003.mp4
rm -rf view3
