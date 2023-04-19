#!/bin/sh
wget -qN http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data
export CXXFLAGS='-Ofast -ffast-math -flto -std=c++11 -pedantic -Werror -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wno-unused'
g++ $CXXFLAGS -o train train.cpp && time ./train > model.h
g++ $CXXFLAGS -o predict predict.cpp
perl -ne 'push@d,$_}{print$d[rand$#d]' semeion.data | ./predict