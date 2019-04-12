#!/usr/bin/env bash
for f in $(ls *.mp4); do
 # 1/5 -> every 5 sec.
 ffmpeg -i $f -vf fps=1/3 -qscale:v 1 ${f}_thumb%09d.jpg -hide_banner
done
