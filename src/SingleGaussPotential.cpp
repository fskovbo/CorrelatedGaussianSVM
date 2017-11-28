#include "SingleGaussPotential.h"

SingleGaussPotential::SingleGaussPotential(System& sys, double baseStr, double interactionRange)
  : vArrayList(sys.vArrayList), n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {
  alpha = 1.0/pow(interactionRange,2);
  interStr = calculateIntStr(sys.masses,baseStr,interactionRange);
}

vec SingleGaussPotential::calculateIntStr(vec& masses, double baseStr, double intRange){
  interStr = zeros<vec>(n*(n+1)/2);
  size_t count = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      double mu = masses(i)*masses(j)/(masses(i)+masses(j));
      interStr(count) = baseStr/(2.0*mu*intRange*intRange);
      count++;
    }
  }
  return interStr;
}

double SingleGaussPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  vec v = 2.0*A1*s1 + 2.0*A2*s2;

  vec kappavec(n*(n+1)/2);
  vec** vArray;
  double detBp, wBw;
  mat Bpinv;
  size_t count = 0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      detBp = detB;
      Bpinv = Binv;
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        wBw = 1+ alpha*dot((vArray[i][j]),Binv*vArray[i][j]);
        detBp *= wBw;
        Bpinv -= (Binv*vArray[i][j]*(vArray[i][j]).t()*Binv)/wBw;
      }
      kappavec(count) = pow(detBp,-3.0/De/2.0) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 1/4.0 *dot(v,Bpinv*v));
      count++;
    }
  }
  return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);
}

double SingleGaussPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  vec kappavec(n*(n+1)/2);
  vec** vArray;
  double detBp;
  size_t count = 0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      detBp = detB;
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        detBp *= 1+ alpha*dot((vArray[i][j]),Binv*vArray[i][j]);
      }
      kappavec(count) = pow(detBp,-3.0/De/2.0);
      count++;
    }
  }
  return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);
}

double SingleGaussPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad){
  vec kappavec(n*(n+1)/2);
  vec** vArray;
  double detBp, Gtemp;
  size_t count = 0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      detBp = detB;
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        detBp *= 1+ alpha*dot((vArray[i][j]),Binv*vArray[i][j]);
      }
      kappavec(count) = pow(detBp,-3.0/De/2.0);
      Gtemp += kappavec(count)/detBp;
      count++;
    }
  }
  
  Vgrad = -1.5*detBgrad*Gtemp*pow(datum::pi,3.0*n/2.0);
  return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);
}
