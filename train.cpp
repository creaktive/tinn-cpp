#include <algorithm>
#include <fstream>
#include <iostream>

#include "tinn.hpp"

using namespace std;

struct Data {
  vector<tinn_num> in;
  vector<tinn_num> tg;
};

const vector<Data> build(const string filename, const size_t nips, const size_t nops) {
  ifstream input(filename);
  if (!input.is_open())
    throw runtime_error("can't open " + filename);
  vector<Data> data;
  for (string line; getline(input, line);) {
    stringstream stream;
    stream << line;
    Data row;
    tinn_num val;
    for (size_t i = 0; stream >> val; i++)
      if (i < nips)
        row.in.push_back(val);
      else if (i < nips + nops)
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
  tinn_num rate = 1.0;
  const size_t nhid = 28;
  const tinn_num anneal = 0.99;
  const size_t iterations = 128;
  // Load the training set.
  auto data = build("semeion.data", nips, nops);
  // Train, baby, train.
  auto tinn = Tinn(nips, nhid, nops);
  for (size_t i = 0; i < iterations; i++) {
    shuffle(data.begin(), data.end(), tinn.rng);
    tinn_num error = 0.0;
    for (size_t j = 0; j < data.size(); j++)
      error += tinn.train(data[j].in, data[j].tg, rate);
    cerr << "error " << fixed << error / data.size();
    cerr << " :: ";
    cerr << "learning rate " << fixed << rate;
    cerr << endl;
    rate *= anneal;
  }
  // This is how you save the neural network.
  auto model = tinn.save();
  cout << "#define NIPS " << nips << endl;
  cout << "#define NHID " << nhid << endl;
  cout << "#define NOPS " << nops << endl;
  cout << "#define MODEL {" << tinn.dump_vector(model, ",") << "}" << endl;
  // This is how you load the neural network.
  tinn = Tinn(nips, nhid, nops, model);
  // Now we do a prediction with the neural network we loaded.
  // Ideally, we would also load a testing set to make the prediction with,
  // but for the sake of brevity here we just reuse the training set from earlier.
  // One data set is picked at random (zero index of input and target arrays is enough
  // as they were both shuffled earlier).
  auto in = data[0].in;
  auto tg = data[0].tg;
  auto pd = tinn.predict(in);
  // Prints target.
  cerr << tinn.dump_vector(tg) << endl;
  // Prints prediction.
  cerr << tinn.dump_vector(pd) << endl;
  return 0;
}