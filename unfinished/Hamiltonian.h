#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <armadillo>

#include "GaussianGenerator.h"
#include "Evals.h"

using namespace arma;
using namespace std;

class Hamiltonian {
private:
  size_t K;
  size_t n;
  mat H;
  mat B;
  MatrixElements matElem;
  cube basis;
  cube shift;
  vec coeffs;
  vec Emin;

public:
  Hamiltonian(size_t K, size_t n, MatrixElements matElem);
  mat getH();
  mat getB();

  vec OptByRound(size_t rounds, size_t lambda, vec& xstart);
};

#endif
