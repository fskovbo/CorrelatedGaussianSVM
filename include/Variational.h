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
  size_t K, n, De, Nunique;
  mat H, B, shift;
  vec uniquePar;
  MatrixElements matElem;
  cube basis;
  vector< vector<double> > basisCoefficients;
  vector<vec**> vArrayList;
  vector<vec> vList;

  double eigenEnergy(size_t state);
  mat generateRandomGaussian(vec& Ameanval, vector<double>& coeffs);
  mat updateMatrices(vector<double> x, size_t index, bool shifted, vec& snew);

public:
  Variational(System& sys, MatrixElements& matElem);

  void setUniqueCoordinates(size_t Nunique_, vec uniquePar_);
  double initializeBasis(size_t basisSize);
  vec sweepStochastic(size_t state, size_t sweeps, size_t trials, vec Ameanval);
  vec sweepStochasticShift(size_t state, size_t sweeps, size_t trials, vec Ameanval, vec maxShift);
  vec sweepDeterministicCMAES(size_t state, size_t sweeps, size_t maxeval);
  vec sweepDeterministic_grad(size_t state, size_t sweeps);

  vec fullBasisSearch(size_t state);

  vec sweepDeterministic(size_t state, size_t sweeps, vec shiftBounds = {0,0,0});
  vec sweepDeterministic(size_t state, size_t sweeps, size_t Nf, vec shiftBounds = {0,0,0});

  void printBasis();
  void printShift();
};

#endif
