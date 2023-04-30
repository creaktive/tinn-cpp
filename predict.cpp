#include <fstream>
#include <iostream>

#include "tinn.hpp"
#include "model.h"

using namespace std;

int main() {
  // This is how you load the neural network.
  auto tinn = Tinn(NIPS, NHID, NOPS, MODEL);
  for (string line; getline(cin, line);) {
    stringstream stream;
    stream << line;
    vector<tinn_num> in;
    for (size_t i = 0; i < NIPS; i++) {
      tinn_num val;
      stream >> val;
      in.push_back(val);
    }
    auto pd = tinn.predict(in);
    cout << tinn.dump_vector(pd) << endl;
  }
  return 0;
}