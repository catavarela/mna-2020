#!/bin/bash

# get path to base of project
BASEOFPROJECT="$(dirname $BASH_SOURCE[0])/.."
# cd to base of project
cd $BASEOFPROJECT

# create data directory and cd to it
mkdir data
cd data

# dowload public face database and uncompress it
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -x -f lfw.tgz

# copy a random one
find lfw -type f | sort -R | tail -1 | xargs -I randimg cp randimg image.jpg