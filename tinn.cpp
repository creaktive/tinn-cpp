#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

class Tinn {
  private:
    // All the weights.
    std::vector<double> w;
    // Hidden to output layer weights.
    std::vector<double> x;
    // Biases.
    std::vector<double> b;
    // Hidden layer.
    std::vector<double> h;
    // Output layer.
    std::vector<double> o;
    // Number of biases - always two - Tinn only supports a single hidden layer.
    const size_t nb = 2;
    // Number of weights.
    size_t nw;
    // Number of inputs.
    size_t nips;
    // Number of hidden neurons.
    size_t nhid;
    // Number of outputs.
    size_t nops;

    // Computes error.
    double err(const double p, const double q) {
      return 0.5 * (p - q) * (p - q);
    }

    // Returns partial derivative of error function.
    double pderr(const double p, const double q) {
      return p - q;
    }

    // Computes total error of target to output.
    double toterr(const std::vector<double> tg) {
      double sum = 0.0;
      for (size_t i = 0; i < nops; i++)
        sum += err(tg[i], o[i]);
      return sum;
    }

    // Activation function.
    double act(const double p) {
      return 1.0 / (1.0 + std::exp(-p));
    }

    // Returns partial derivative of activation function.
    double pdact(const double p) {
      return p * (1.0 - p);
    }

    // Returns floating point random from 0.0 - 1.0.
    double frand() {
      static std::uniform_real_distribution<double> dist(0.0, 1.0);
      return dist(rng);
    }

    void bprop(const std::vector<double> in, const std::vector<double> tg, const double rate) {
      for (size_t i = 0; i < nhid; i++) {
        double sum = 0.0;
        // Calculate total error change with respect to output.
        for (size_t j = 0; j < nops; j++) {
          const double p = pderr(o[j], tg[j]);
          const double q = pdact(o[j]);
          sum += p * q * x[j * nhid + i];
          // Correct weights in hidden to output layer.
          x[j * nhid + i] -= rate * p * q * h[i];
        }
        // Correct weights in input to hidden layer.
        for (size_t j = 0; j < nips; j++)
          w[i * nips + j] -= rate * sum * pdact(h[i]) * in[j];
      }
    }

    void wbrand() {
      for (size_t i = 0; i < nw; i++)
        w[i] = frand() - 0.5;
      for (size_t i = 0; i < nb; i++)
        b[i] = frand() - 0.5;
    }

    void fprop(const std::vector<double> in) {
      // Calculate hidden layer neuron values.
      for (size_t i = 0; i < nhid; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < nips; j++)
          sum += in[j] * w[i * nips + j];
        h[i] = act(sum + b[0]);
      }
      // Calculate output layer neuron values.
      for (size_t i = 0; i < nops; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < nhid; j++)
          sum += h[j] * x[i * nhid + j];
        o[i] = act(sum + b[1]);
      }
    }

  public:
    std::default_random_engine rng;

    Tinn(const size_t _nips, const size_t _nhid, const size_t _nops)
      : nips(_nips), nhid(_nhid), nops(_nops) {
      nw = nhid * (nips + nops);
      w.resize(nw);
      x.resize(nw + nhid * nips);
      b.resize(nb);
      h.resize(nhid);
      o.resize(nops);

      wbrand();
    }

    // Returns an output prediction given an input.
    std::vector<double> predict(const std::vector<double> in) {
      fprop(in);
      return o;
    }

    // Trains a tinn with an input and target output with a learning rate. Returns target to output error.
    double train(const std::vector<double> in, const std::vector<double> tg, const double rate) {
      fprop(in);
      bprop(in, tg, rate);
      return toterr(tg);
    }

    void print(const std::vector<double> arr) {
      std::stringstream stream;
      for (size_t i = 0; i < arr.size(); i++)
        stream << arr[i] << ' ';
      std::cout << stream.str() << std::endl;
    }
};

struct Data {
  std::vector<double> in;
  std::vector<double> tg;
};

std::vector<Data> build(const std::string filename, const size_t nips, const size_t nops);
std::vector<Data> build(const std::string filename, const size_t nips, const size_t nops) {
  std::ifstream input(filename);
  std::vector<Data> data;
  std::string line;
  while (getline(input, line)) {
    std::stringstream stream;
    stream << line;
    Data row;
    size_t col = 0;
    double val;
    while (stream >> val) {
      if (col < nips)
        row.in.push_back(val);
      else if (col < nips + nops)
        row.tg.push_back(val);
      else
        throw std::runtime_error("malformed input");
      col++;
    }
    data.push_back(row);
  }
  input.close();
  return data;
}

int main() {
  // Input and output size is harded coded here as machine learning
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
    std::shuffle(data.begin(), data.end(), tinn.rng);
    double error = 0.0;
    for (size_t j = 0; j < data.size(); j++)
      error += tinn.train(data[j].in, data[j].tg, rate);
    printf("error %.12f :: learning rate %f\n", error / data.size(), rate);
    rate *= anneal;
  }

  auto pd = tinn.predict(data[0].in);
  tinn.print(data[0].tg);
  tinn.print(pd);

  return 0;
}