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

# # eliminate all folders less than 4 folders
# for i in $(ls lfw); 
# do
#     if [ 4 -gt $(ls lfw/$i | wc -l) ]
#     then
#         rm -r lfw/$i
#     fi
# done

# copy a random one
find lfw -type f | sort -R | tail -1 | xargs -I randimg cp randimg image.jpg