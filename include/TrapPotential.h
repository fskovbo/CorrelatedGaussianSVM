#ifndef TRAPPOTENIAL_H
#define TRAPPOTENIAL_H

#include <armadillo>
#include <vector>
#include <iostream>
#include "PotentialStrategy.h"
#include "System.h"

using namespace arma;
using namespace std;

class TrapPotential : public PotentialStrategy{
private:
  size_t n;
  mat Zmat;
  vec masses;

public:
  TrapPotential(System& sys, double trapFreq);
  void updateTrap(double trapFreq);
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB);
};

#endif
