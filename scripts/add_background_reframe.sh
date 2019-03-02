#!/usr/bin/env bash

# recenter
for f in *.png; do
 echo "$f"
 convert "$f" -gravity center -background black -extent 1920x1080 -quality 9 "$f"
done

# add file name
# mogrify -font Liberation-Sans -fill white -undercolor '#00000080' -pointsize 26 -gravity NorthEast -annotate +10+10 %t *.png
