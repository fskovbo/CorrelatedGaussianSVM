#ifndef MATRIXELEMENTS_H
#define MATRIXELEMENTS_H

#include <armadillo>
#include "System.h"
#include "PotentialStrategy.h"

using namespace arma;
using namespace std;


class MatrixElements {
private:
  size_t n, De, NparA;
  mat Lambda;
  PotentialStrategy& Vstrat;
  vector<vec**> vArrayList;
  vector<vec> vList;

public:
  MatrixElements(System& sys, PotentialStrategy& Vstrat);
  void calculateH(mat& A1, mat& A2, vec& s1, vec& s2, double& Hij, double& Bij);
  void calculateH(mat& A1, mat& A2, vec& s1, vec& s2, double& Hij, double& Bij, vec& Hgrad, vec& Mgrad);
  void calculateH_noShift(mat& A1, mat& A2, double& Hij, double& Bij);
  void calculateH_noShift(mat& A1, mat& A2, double& Hij, double& Bij, vec& Hgrad, vec& Mgrad);
};

#endif
