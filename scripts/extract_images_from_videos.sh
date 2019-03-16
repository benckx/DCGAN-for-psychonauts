#!/usr/bin/env bash
for f in $(ls *.mp4); do
 # 1/5 -> every 5 sec.
 ffmpeg -i $f -vf fps=1/4 ${f}_thumb%06d.jpg -hide_banner
done
