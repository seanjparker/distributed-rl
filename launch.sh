#!/bin/sh

while getopts f:e: flag
do
  case "${flag}" in
    f) framework=${OPTARG};;
    e) epochs=${OPTARG};;
    *) echo "Invalid arguments";;
  esac
done

for i in 2 4 8 16 32; do
  python run.py --fw "$framework" --w "$i" --epochs "$epochs"
done