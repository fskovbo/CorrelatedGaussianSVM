#ifndef JACOBICOORDINATES_H
#define JACOBICOORDINATES_H

#include <armadillo>

using namespace arma;
using namespace std;

class JacobiCoordinates {
public:
    static mat buildTransformationMatrix(vec& masses);
    static mat buildReducedMassMatrix(vec& masses);
};


#endif
