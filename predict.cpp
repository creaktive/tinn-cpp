#include <fstream>
#include <iostream>

#include "tinn.hpp"
#include "model.h"

using namespace std;

struct Data {
  vector<tinn_num> in;
  vector<tinn_num> pd;
};

int main() {
  // This is how you load the neural network.
  auto tinn = Tinn(NIPS, NHID, NOPS, MODEL);
  vector<Data> data;
  for (string line; getline(cin, line);) {
    stringstream stream;
    stream << line;
    vector<tinn_num> in;
    for (size_t i = 0; i < NIPS; i++) {
      tinn_num val;
      stream >> val;
      in.push_back(val);
    }
    Data row;
    row.in = in;
    row.pd = tinn.predict(in);
    data.push_back(row);
  }
  printf("P2\n%d\n%d\n255\n", NIPS + NOPS, static_cast<int>(data.size()));
  for (auto row : data) {
    for (size_t i = 0; i < NIPS; i++)
      printf("%d ", static_cast<unsigned char>(255 * row.in[i]));
    tinn_num h = 0.0;
    for (size_t i = 0; i < NOPS; i++)
      if (h < row.pd[i])
        h = row.pd[i];
    if (h <= 0.9)
      h = -1.0;
    for (size_t i = 0; i < NOPS; i++)
      printf(row.pd[i] == h ? "255 " : "0   ");
  }
  return 0;
}