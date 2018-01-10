#ifndef REALTRAPPOTENIAL_H
#define REALTRAPPOTENIAL_H

#include <armadillo>
#include <vector>
#include <iostream>
#include "PotentialStrategy.h"
#include "System.h"

using namespace arma;
using namespace std;

class RealTrapPotential : public PotentialStrategy{
private:
  size_t n, De;
  mat OmegaXY, OmegaZ, lambdamat, XYmat, Zmat;
  double trapLengthXY, trapLengthZ;

public:
  RealTrapPotential(System& sys, double trapLengthXY);
  RealTrapPotential(System& sys, double trapLengthXY, double trapLengthZ);
  void updateTrap(double trapLengthZ);
  double gsExpectedVal();
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad);
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2,
                                            mat& Binv, double detB, vec& Vgrad_A, vec& Vgrad_s,
                                            cube& Binvgrad, vec& detBgrad);
};

#endif
