#!/bin/bash

echo "Downloading dog_cat_segmentation dataset ..."
mkdir data data/cache
wget -c https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz && tar -xzvf annotations.tar.gz -C ./data
wget -c https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz && tar -xzvf images.tar.gz -C ./data
rm -f annotations.tar.gz images.tar.gz