#!/usr/bin/env bash

# view1 (3x3)
mkdir view1
for f in $(ls *.png); do
 echo $f
 convert $f -crop 1920x1080+0+0 view1/$f
done

ffmpeg -framerate 30 -f image2 -i 'view1/%*.png' -c:v libx264 -profile:v high -crf 16 -pix_fmt yuv420p shutter_g7d7_e25_3x3_view1.mp4
rm -rf view1

# view2 (3x3)
mkdir view2
for f in $(ls *.png); do
 echo $f
 convert $f -crop 1920x1080+1920+360 view2/$f
done

ffmpeg -framerate 30 -f image2 -i 'view2/%*.png' -c:v libx264 -profile:v high -crf 16 -pix_fmt yuv420p shutter_g7d7_e25_3x3_view2.mp4
rm -rf view2

# view 3 (4x4)
mkdir view3
for f in $(ls *.png); do
 echo $f
 convert $f -crop 2560x1440+0+0 view3/$f
done

ffmpeg -framerate 30 -f image2 -i 'view3/%*.png' -c:v libx264 -profile:v high -crf 16 -pix_fmt yuv420p shutter_g7d7_e25_4x4_view3.mp4
rm -rf view3

# view4 (4x4)
mkdir view4
for f in $(ls *.png); do
 echo $f
 convert $f -crop 2560x1440+640+0 view4/$f
done

ffmpeg -framerate 30 -f image2 -i 'view4/%*.png' -c:v libx264 -profile:v high -crf 16 -pix_fmt yuv420p shutter_g7d7_e25_4x4_view4.mp4
rm -rf view4

