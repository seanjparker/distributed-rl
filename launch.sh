#!/bin/sh

for i in 2 4 8 16 32; do
  python run.py --fw torch --w "$i" --epochs 200
done