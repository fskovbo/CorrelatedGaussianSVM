#ifndef POTENTIALSTRATEGY_H
#define POTENTIALSTRATEGY_H

#include <armadillo>

using namespace arma;
using namespace std;

class PotentialStrategy {
public:
    virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB) = 0;
    virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB) = 0;
    virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad) = 0;
};

#endif
