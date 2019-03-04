#!/usr/bin/env bash
for f in $(ls *.jpg); do
 NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
 mv $f ${NEW_UUID}.jpg
done