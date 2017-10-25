#ifndef SINGLEGAUSSPOTENIAL_H
#define SINGLEGAUSSPOTENIAL_H

#include <armadillo>
#include <vector>
#include "PotentialStrategy.h"
#include "Utils.h"
#include "System.h"

using namespace arma;
using namespace std;

class SingleGaussPotential : public PotentialStrategy{
private:
  vec interStr;
  vector<vec**> vArrayList;
  double alpha;
  size_t n;
  bool JacobiCoordinates;

public:
  SingleGaussPotential(System& sys, double interactionStr, double interactionRange);
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB);
};

#endif
