#ifndef UTILS_H
#define UTILS_H

#include <armadillo>
#include <assert.h>
#include "fdcube.h"

using namespace arma;
using namespace std;

class Utils {
public:
    static void invExpansion(vec& b, vec& c, double min, double max, int res);
    static fdcube buildInteraction(size_t N, vec& alpha, mat& Ui);
    static vec buildInterStr(vec& masses, double baseStr, double interwidth);
    static vec buildQinter(vec& Q);
    static cube buildTrap(size_t N, mat& Ui);
    static vec buildOmegaSQ(vec& masses, double oscWidth);
    static void translateToBasis(vec& x, cube& A, mat& s, bool diagonal);
    static void translateToList(vec& x, cube& A, mat& s, bool diagonal);
    static void vec2symmetricMat(vec& vector, mat& symMatrix);
    static void symmetricMat2vec(vec& vector, mat& symMatrix);
};

#endif
