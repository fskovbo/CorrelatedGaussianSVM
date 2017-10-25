#include "GaussianGenerator.h"

mat GaussianGenerator::genMatrix(size_t n, double mean){
  mat A = zeros<mat>(3*n,3*n);
  A.diag() = 1.0/pow(-mean*log(randu<vec>(3*n)),2);
  return A;
}

vec GaussianGenerator::genShift(size_t n, double width){
  return width*randn<vec>(3*n);
}

double GaussianGenerator::genCoeff(double mean, double width){
  vec lol = width*randn<vec>(1);
  return lol(0)+mean;
}
