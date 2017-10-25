#ifndef GAUSSIANGENERATOR_H
#define GAUSSIANGENERATOR_H

#include <armadillo>
#include <assert.h>

using namespace arma;
using namespace std;

class GaussianGenerator {
public:
    static mat genMatrix(size_t n, double mean);
    static vec genShift(size_t n, double width);
    static double genCoeff(double mean, double width);
};

#endif
