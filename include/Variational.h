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

#include "FigureOfMerits.hpp"

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
  vec sweepDeterministic_grad(size_t state, size_t sweeps);

  void printBasis();
  void printShift();
  void printBasisCoeffs();
};

#endif
