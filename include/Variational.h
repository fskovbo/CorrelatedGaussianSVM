#ifndef VARIATIONAL_H
#define VARIATIONAL_H

#include <armadillo>
#include <functional>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

#include "System.h"
#include "MatrixElements.h"
#include "CMAES.h"
#include "Utils.h"

using namespace arma;
using namespace std;


class Variational {
private:
  size_t K, n;
  mat H, B, shift;
  MatrixElements matElem;
  cube basis;
  mat basisCoefficients;
  vector<vec**> vArrayList;

public:
  Variational(System& sys, MatrixElements& matElem);

  double groundStateEnergy();
  mat generateRandomGaussian(vec& Ameanval, vec& coeffs);
  double initializeBasis(size_t basisSize);
  vec sweepStochastic(size_t sweeps, size_t trials, vec& Ameanval);
  vec sweepDeterministic(size_t sweeps, size_t maxeval);


  double addBasisFunctionCMAES(mat A_guess, vec S_guess, size_t state, size_t maxeval);

  void printBasis();
  void printShift();
};

#endif
