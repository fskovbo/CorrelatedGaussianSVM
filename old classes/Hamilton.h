#ifndef HAMILTON_H
#define HAMILTON_H

#include <armadillo>
#include <functional>
#include <assert.h>
#include <iostream>
#include <math.h>

#include "MatrixElements.h"
#include "CMAES.h"
#include "Evals.h"
#include "Multidim_min.h"
#include "Utils.h"

using namespace arma;
using namespace std;

class Hamilton {
private:
  size_t K;
  size_t n;
  mat H;
  mat B;
  MatrixElements matElem;

public:
  Hamilton(size_t K, size_t n, MatrixElements matElem);
  mat getH();
  mat getB();
  void buildMatrices(vec& x);
  vec optimizeH(vec& xmean, size_t lambda, size_t mu, double sigma, size_t maxeval);
  void buildstuff(vec& x, mat& H, mat& B);
  
  void expBasis(vec& x, mat& H, mat& B, vec& Amean, double Swidth, size_t state, size_t lambda);
  vec simpleOptimize(mat& Amean, vec& Swidth, vec& Kvec, size_t lambda);
};

#endif
