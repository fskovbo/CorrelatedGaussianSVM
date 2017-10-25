#ifndef DOUBLETRAPPOTENIAL_H
#define DOUBLETRAPPOTENIAL_H

#include <armadillo>
#include "PotentialStrategy.h"

using namespace arma;
using namespace std;

class DoubleTrapPotential : public PotentialStrategy{
private:
  vec omegasqY, omegasqZ;
  cube Q;
public:
  DoubleTrapPotential(vec& omegasqY, vec& omegasqZ, cube& Q);
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB);
};

#endif
