#ifndef COULOMBPOTENIAL_H
#define COULOMBPOTENIAL_H

#include <armadillo>
#include "PotentialStrategy.h"
#include "Utils.h"
#include "fdcube.h"
#include "System.h"

using namespace arma;
using namespace std;

class CoulombPotential : public PotentialStrategy{
private:
  size_t n, De;
  vec Qinter, bvec, cvec;
  fdcube interactions;

  fdcube buildInteractions(vec& alpha, mat& Ui);

public:
  CoulombPotential(System& sys, size_t expansionterms, double range);
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad);
};

#endif
