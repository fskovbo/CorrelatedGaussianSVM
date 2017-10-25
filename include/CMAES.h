#ifndef CMAES_H
#define CMAES_H

#include <armadillo>
#include <functional>
#include <iostream>

using namespace arma;
using namespace std;

class CMAES {
private:
public:
  static void optimize(std::function<double(vec&)> fitness, vec& xmean, size_t lambda, size_t mu, double sigma, size_t maxeval);
};

#endif
