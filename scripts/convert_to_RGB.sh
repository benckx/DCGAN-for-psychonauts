#!/usr/bin/env bash
for f in $(identify -format "%i %[colorspace]\n" *.jpg | grep -v sRGB | awk '{print $1;}'); do
 echo "$f"
 convert "$f" -colorspace sRGB -type truecolor "$f"
done
