#include <algorithm>
#include <fstream>

#include "tinn.hpp"

using namespace std;

struct Data {
  vector<double> in;
  vector<double> tg;
};

vector<Data> build(const string filename, const size_t nips, const size_t nops) {
  ifstream input(filename);
  if (!input.is_open())
    throw runtime_error("can't open " + filename);
  vector<Data> data;
  string line;
  while (getline(input, line)) {
    stringstream stream;
    stream << line;
    Data row;
    double val;
    for (size_t col = 0; stream >> val; col++)
      if (col < nips)
        row.in.push_back(val);
      else if (col < nips + nops)
        row.tg.push_back(val);
    if (row.in.size() != nips || row.tg.size() != nops)
      throw runtime_error("malformed input");
    data.push_back(row);
  }
  input.close();
  return data;
}

int main() {
  // Input and output size is hard coded here as machine learning
  // repositories usually don't include the input and output size in the data itself.
  const size_t nips = 256;
  const size_t nops = 10;
  // Hyper Parameters.
  // Learning rate is annealed and thus not constant.
  // It can be fine tuned along with the number of hidden layers.
  // Feel free to modify the anneal rate.
  // The number of iterations can be changed for stronger training.
  double rate = 1.0;
  const size_t nhid = 28;
  const double anneal = 0.99;
  const size_t iterations = 128;
  // Load the training set.
  auto data = build("semeion.data", nips, nops);
  // Train, baby, train.
  auto tinn = Tinn(nips, nhid, nops);
  for (size_t i = 0; i < iterations; i++) {
    shuffle(data.begin(), data.end(), tinn.rng);
    double error = 0.0;
    for (size_t j = 0; j < data.size(); j++)
      error += tinn.train(data[j].in, data[j].tg, rate);
    cout << "error " << fixed << error / data.size();
    cout << " :: ";
    cout << "learning rate " << fixed << rate;
    cout << endl;
    rate *= anneal;
  }
  auto model = tinn.save();
  tinn = Tinn(nips, nhid, nops, model);
  auto pd = tinn.predict(data[0].in);
  tinn.print(data[0].tg);
  tinn.print(pd);
  return 0;
}