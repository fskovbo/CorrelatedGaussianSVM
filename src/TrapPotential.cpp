#include "TrapPotential.h"

TrapPotential::TrapPotential(System& sys)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {

  Xmat = zeros<mat>(De*n,De*n);
  for (size_t i = 0; i < n; i++) {
    Xmat(De*i,De*i) = 1;
  }
  Xmatgrad = zeros<mat>(De*n*(n+1)/2,De*n*(n+1)/2);
  for (size_t i = 0; i < n*(n+1)/2; i++) {
    Xmatgrad(De*i,De*i) = 1;
  }
}

TrapPotential::TrapPotential(System& sys, double trapLength)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {

  Xmat = zeros<mat>(De*n,De*n);
  for (size_t i = 0; i < n; i++) {
    Xmat(De*i,De*i) = 1;
  }
  Xmatgrad = zeros<mat>(De*n*(n+1)/2,De*n*(n+1)/2);
  for (size_t i = 0; i < n*(n+1)/2; i++) {
    Xmatgrad(De*i,De*i) = 1;
  }

  updateTrap(trapLength);
}

void TrapPotential::updateTrap(double trapLength){
  trapLength_cur = trapLength;

  Omega = lambdamat%Xmat * 0.5 * pow(trapLength,-4);
}

double TrapPotential::gsExpectedVal(){
  return 1.5*trace(lambdamat)/De/De/trapLength_cur/trapLength_cur;
}

double TrapPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  return 999999999;
}

double TrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  return pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0) * 1.5/De*trace(Omega*Binv);
}

double TrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad){
  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0);
  vec Mgrad = -1.5/detB *overlap*detBgrad;
  vec Vgrad2(Vgrad);

  for (size_t i = 0; i < De*n*(n+1)/2; i++) {
    Vgrad2(i) = trace(Omega*Binvgrad.slice(i));
  }
  Vgrad = 1.5/De*trace(Omega*Binv)*Xmatgrad*Mgrad + 1.5*Vgrad2*overlap;
  return overlap * 1.5/De*trace(Omega*Binv);
}
