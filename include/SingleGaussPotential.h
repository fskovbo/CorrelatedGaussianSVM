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
  size_t n, De;
  mat lambdamat;

  vector<mat> interactions;

  vec calculateIntStr(vec& masses, double baseStr, double intRange);
  void buildInteractions(mat& invTrans);

public:
  SingleGaussPotential(System& sys, double baseStr = -2.684, double interactionRange = 1.0);
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB);
};

#endif
