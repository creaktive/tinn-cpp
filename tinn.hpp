#pragma once

#include <random>
#include <sstream>
#include <vector>

#ifndef tinn_num
typedef float tinn_num;
#endif

class Tinn {
  public:
    std::default_random_engine rng;

    Tinn(const size_t _nips, const size_t _nhid, const size_t _nops, const std::vector<tinn_num> data = {})
      : nips(_nips), nhid(_nhid), nops(_nops) {
      nb = 2;
      nw = nhid * (nips + nops);
      w.resize(nw);
      b.resize(nb);
      h.resize(nhid);
      o.resize(nops);
      if (data.size() != nb + nw) {
        for (size_t i = 0; i < nb; i++)
          b[i] = frand();
        for (size_t i = 0; i < nw; i++)
          w[i] = frand();
      } else {
        for (size_t i = 0; i < nb; i++)
          b[i] = data[i];
        for (size_t i = 0; i < nw; i++)
          w[i] = data[nb + i];
      }
      x = w.data() + nhid * nips;
    }

    const std::vector<tinn_num> save() {
      std::vector<tinn_num> data(nb + nw);
      for (size_t i = 0; i < nb; i++)
        data[i] = b[i];
      for (size_t i = 0; i < nw; i++)
        data[nb + i] = w[i];
      return data;
    }

    // Returns an output prediction given an input.
    const std::vector<tinn_num> predict(const std::vector<tinn_num> in) {
      fprop(in);
      return o;
    }

    // Trains a tinn with an input and target output with a learning rate. Returns target to output error.
    tinn_num train(const std::vector<tinn_num> in, const std::vector<tinn_num> tg, const tinn_num rate) {
      fprop(in);
      bprop(in, tg, rate);
      return toterr(tg);
    }

    // Prints an array of floats. Useful for printing predictions.
    const std::string dump_vector(const std::vector<tinn_num> data, const std::string sep = " ", const std::streamsize prec = 6) {
      std::stringstream stream;
      stream.setf(std::ios::fixed);
      stream.precision(prec);
      size_t end = data.size() - 1;
      for (size_t i = 0; i <= end; i++)
        stream << data[i] << (i == end ? "" : sep);
      return stream.str();
    }

  private:
    // All the weights.
    std::vector<tinn_num> w;
    // Hidden to output layer weights.
    tinn_num *x;
    // Biases.
    std::vector<tinn_num> b;
    // Hidden layer.
    std::vector<tinn_num> h;
    // Output layer.
    std::vector<tinn_num> o;
    // Number of biases - always two - Tinn only supports a single hidden layer.
    size_t nb;
    // Number of weights.
    size_t nw;
    // Number of inputs.
    size_t nips;
    // Number of hidden neurons.
    size_t nhid;
    // Number of outputs.
    size_t nops;

    // Computes error.
    tinn_num err(const tinn_num p, const tinn_num q) {
      return 0.5 * (p - q) * (p - q);
    }

    // Returns partial derivative of error function.
    tinn_num pderr(const tinn_num p, const tinn_num q) {
      return p - q;
    }

    // Computes total error of target to output.
    tinn_num toterr(const std::vector<tinn_num> tg) {
      tinn_num sum = 0.0;
      for (size_t i = 0; i < nops; i++)
        sum += err(tg[i], o[i]);
      return sum;
    }

    // Activation function.
    tinn_num act(const tinn_num p) {
      return 1.0 / (1.0 + std::exp(-p));
    }

    // Returns partial derivative of activation function.
    tinn_num pdact(const tinn_num p) {
      return p * (1.0 - p);
    }

    // Returns random from -0.5 to 0.5.
    tinn_num frand() {
      static std::uniform_real_distribution<tinn_num> dist(-0.5, 0.5);
      return dist(rng);
    }

    void bprop(const std::vector<tinn_num> in, const std::vector<tinn_num> tg, const tinn_num rate) {
      for (size_t i = 0; i < nhid; i++) {
        tinn_num sum = 0.0;
        // Calculate total error change with respect to output.
        for (size_t j = 0; j < nops; j++) {
          const tinn_num p = pderr(o[j], tg[j]);
          const tinn_num q = pdact(o[j]);
          sum += p * q * x[j * nhid + i];
          // Correct weights in hidden to output layer.
          x[j * nhid + i] -= rate * p * q * h[i];
        }
        // Correct weights in input to hidden layer.
        for (size_t j = 0; j < nips; j++)
          w[i * nips + j] -= rate * sum * pdact(h[i]) * in[j];
      }
    }

    void fprop(const std::vector<tinn_num> in) {
      // Calculate hidden layer neuron values.
      for (size_t i = 0; i < nhid; i++) {
        tinn_num sum = 0.0;
        for (size_t j = 0; j < nips; j++)
          sum += in[j] * w[i * nips + j];
        h[i] = act(sum + b[0]);
      }
      // Calculate output layer neuron values.
      for (size_t i = 0; i < nops; i++) {
        tinn_num sum = 0.0;
        for (size_t j = 0; j < nhid; j++)
          sum += h[j] * x[i * nhid + j];
        o[i] = act(sum + b[1]);
      }
    }
};