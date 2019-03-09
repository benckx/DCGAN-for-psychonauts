#!/usr/bin/env bash

for f in $(ls *.mp4); do
 echo $f
 ffmpeg -i $f -c:v libx264 -profile:v high -crf 16 -pix_fmt yuv420p re_encoded_$f
done
