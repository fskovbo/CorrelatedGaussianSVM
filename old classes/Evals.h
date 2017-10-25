#ifndef EVALS_H
#define EVALS_H

#include <armadillo>
#include <iostream>
#include <assert.h>

using namespace arma;
using namespace std;

class Evals {
public:
  static double Rosenbrock(vec& x);
  static double eigenEnergy(mat& H, mat& B);
  static vec eigenSpectrum(mat& H, mat& B);
};

#endif
