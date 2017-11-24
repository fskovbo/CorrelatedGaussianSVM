#ifndef VARIATIONAL_H
#define VARIATIONAL_H

#include <armadillo>
#include <functional>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>
#include "nlopt.hpp"

#include "System.h"
#include "MatrixElements.h"
#include "CMAES.h"
#include "Utils.h"

using namespace arma;
using namespace std;


class Variational {
private:
  size_t K, n, De;
  mat H, B, shift;
  MatrixElements matElem;
  cube basis;
  mat basisCoefficients;
  vector<vec**> vArrayList;

  static double myvfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
  static double myvfunc_grad(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
  double groundStateEnergy();
  mat generateRandomGaussian(vec& Ameanval, vec& coeffs);

public:
  Variational(System& sys, MatrixElements& matElem);

  double initializeBasis(size_t basisSize);
  vec sweepStochastic(size_t sweeps, size_t trials, vec& Ameanval);
  vec sweepDeterministic(size_t sweeps, size_t Nunique = 3, vec uniquePar = {0,1,2});
  vec sweepDeterministic_grad(size_t sweeps, size_t Nunique = 3, vec uniquePar = {0,1,2});

  vec sweepDeterministicCMAES(size_t sweeps, size_t maxeval);

  void printBasis();
  void printShift();


  static double myvfunc_grad_test(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
  vec sweepDeterministic_grad_test();
};

#endif
