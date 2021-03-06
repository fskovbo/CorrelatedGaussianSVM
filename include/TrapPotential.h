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
  size_t n, De;
  mat Omega, lambdamat;
  double gsEnergy;

public:
  TrapPotential(System& sys);
  TrapPotential(System& sys, vec trapLength);
  void updateTrap(vec trapLength);
  double gsExpectedVal();
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad);
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2,
                                            mat& Binv, double detB, vec& Vgrad_A, vec& Vgrad_s,
                                            cube& Binvgrad, vec& detBgrad);
};

#endif
