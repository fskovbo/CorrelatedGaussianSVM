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
#include "FigureOfMerits.hpp"

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

  double eigenEnergy(size_t state);

public:
  Variational(System& sys, MatrixElements& matElem);
  mat generateRandomGaussian(vec& Ameanval, vec& coeffs);

  double initializeBasis(size_t basisSize);
  vec sweepStochastic(size_t state, size_t sweeps, size_t trials, vec Ameanval);
  vec sweepStochasticShift(size_t state, size_t sweeps, size_t trials, vec Ameanval, vec maxShift);
  vec sweepDeterministic(size_t state, size_t sweeps, size_t Nunique = 3, vec uniquePar = {0,1,2});
  vec sweepDeterministicShift(size_t state, size_t sweeps, vec maxShift = {1,1,1}, size_t Nunique = 3, vec uniquePar = {0,1,2});
  vec sweepDeterministicCMAES(size_t state, size_t sweeps, size_t maxeval);

  vec addBasisFunction(size_t state, size_t tries, vec startGuess, vec maxShift = {1,1,1}, size_t Nunique = 3, vec uniquePar = {0,1,2});

  void printBasis();
  void printShift();




  static double myvfunc_grad(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
  vec sweepDeterministic_grad(size_t sweeps, size_t Nunique = 3, vec uniquePar = {0,1,2});
  static double myvfunc_grad_test(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);
  vec sweepDeterministic_grad_test();
};

#endif
