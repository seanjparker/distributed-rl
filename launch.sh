#!/bin/sh

for (( i=2; i<=32; i*=2 )); do
  python run.py --fw torch --w "$i"
done