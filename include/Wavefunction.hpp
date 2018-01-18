#ifndef WAVEFUNCTION_H
#define WAVEFUNCTION_H

#include <armadillo>
#include "System.h"
#include "MatrixElements.h"
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
  MatrixElements matElem;

  double factorial(double x);
  cube buildPermutations();
  void Symmetrize(cube& symbasis, mat& symshift);
  void calculateR(mat& R, mat& A1, mat& A2, vec& s1, vec& s2, double& Rij, double& Bij);

public:
  Wavefunction(System& sys, MatrixElements& matElem, cube& basis, mat& shift, vec& coeffs);
  Wavefunction(System& sys, MatrixElements& matElem, std::string filename);

  vec RMSdistances();
  double Symmetrization();
};

#endif
