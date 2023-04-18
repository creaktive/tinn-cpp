#!/bin/sh
export CXXFLAGS='-Ofast -ffast-math -flto -std=c++11 -pedantic -Werror -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wno-unused'
wget -qN http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data
g++ $CXXFLAGS -o tinn tinn.cpp && time ./tinn