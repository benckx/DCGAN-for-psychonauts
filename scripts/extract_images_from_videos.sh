#!/usr/bin/env bash
for f in $(ls *.mp4); do
 ffmpeg -i $f -vf fps=1/10 ${f}_thumb%05d.jpg -hide_banner
done
