#ifndef TESTS_H
#define TESTs_H

#include <armadillo>
#include <assert.h>
#include "fdcube.h"
#include "Utils.h"
#include "JacobiCoordinates.h"
#include "CoulombPotential.h"

using namespace arma;
using namespace std;

class tests {
public:
    static bool testU();
    //static bool testLambda();
    static bool testCoulomb();
};

#endif
