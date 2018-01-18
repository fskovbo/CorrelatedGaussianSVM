#ifndef WAVEFUNCTION_H
#define WAVEFUNCTION_H

#include <armadillo>
#include "System.h"
#include <cmath>
#include <string>

using namespace arma;
using namespace std;


class Wavefunction {
private:
  size_t K, n, De;
  mat shift, U, Ui;
  cube basis;
  vec coeffs;

  double factorial(double x);
  double calculateExptValue(mat& O);
  void calculateOverlap(mat& O, mat& A1, mat& A2, vec& s1, vec& s2, double c1, double c2, double& Oij, double& Bij);
  cube buildPermutations();

public:
  Wavefunction(System& sys, cube& basis, mat& shift, vec& coeffs);
  Wavefunction(System& sys, std::string filename);

  vec RMSdistances();
  double Symmetrization();
};

#endif
