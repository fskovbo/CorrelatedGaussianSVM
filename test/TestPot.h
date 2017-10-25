#ifndef TESTPOT_H
#define TESTPOT_H

#include <armadillo>
#include "PotentialStrategy.h"

using namespace arma;
using namespace std;

class TestPot : public PotentialStrategy{
public:
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2);
};

#endif
