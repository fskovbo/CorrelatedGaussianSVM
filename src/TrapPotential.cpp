#include "TrapPotential.h"

TrapPotential::TrapPotential(System& sys)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {

}

TrapPotential::TrapPotential(System& sys, double trapLength)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {

  updateTrap(trapLength);
}

void TrapPotential::updateTrap(double trapLength){
  trapLength_cur = trapLength;
  mat Xmat = zeros<mat>(De*n,De*n);
  for (size_t i = 0; i < n; i++) {
    Xmat(De*i,De*i) = 1;
  }
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
  double temp = 1.5/De*trace(Omega*Binv);
  vec Vgrad2(Vgrad);
  for (size_t i = 0; i < De*n*(n+1)/2; i++) {
    Vgrad2(i) = trace(Omega*Binvgrad.slice(i));
  }
  Vgrad = temp*Mgrad + 1.5/De*Vgrad2*overlap;
  return overlap*temp;
}
