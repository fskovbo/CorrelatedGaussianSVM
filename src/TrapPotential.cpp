#include "TrapPotential.h"

TrapPotential::TrapPotential(System& sys)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {

  Xmat = zeros<mat>(De*n,De*n);
  for (size_t i = 0; i < n; i++) {
    Xmat(De*i,De*i) = 1;
  }
}

TrapPotential::TrapPotential(System& sys, double trapLength)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {

  Xmat = zeros<mat>(De*n,De*n);
  for (size_t i = 0; i < n; i++) {
    Xmat(De*i,De*i) = 1;
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
  vec v = s1 + s2;
  vec u = 0.5*Binv*v;
  double overlap = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(0.25*dot(v,Binv*v));
  return overlap*(1.5/De*trace(Omega*Binv) + dot(u,Omega*u));
}

double TrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  return pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0) * 1.5/De*trace(Omega*Binv);
}

double TrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad){
  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0);
  vec Mgrad = -1.5/De/detB *overlap*detBgrad;
  vec Vgrad2(Vgrad);
  for (size_t i = 0; i < De*n*(n+1)/2; i++) {
    Vgrad2(i) = trace(Omega*Binvgrad.slice(i));
  }
  Vgrad = 1.5/De*trace(Omega*Binv)*Mgrad + 1.5/De*Vgrad2*overlap;
  return overlap * 1.5/De*trace(Omega*Binv);
}

double TrapPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2,
                                                 mat& Binv, double detB, vec& Vgrad_A, vec& Vgrad_s,
                                                 cube& Binvgrad, vec& detBgrad)
{
  vec v = s1+s2;
  vec u = 0.5*Binv*v;
  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)*exp(0.25*dot(v,Binv*v));

  vec Vgrad2_A(Vgrad_A), ugrad_A(Vgrad_A), Mgrad2_A(Vgrad_A);
  for (size_t i = 0; i < De*n*(n+1)/2; i++) {
    Vgrad2_A(i) = trace(Omega*Binvgrad.slice(i));
    Mgrad2_A(i) = dot(v,Binvgrad.slice(i)*v);
    ugrad_A(i) = dot(u,Omega*Binvgrad.slice(i)*v);
  }
  vec Mgrad_A = 0.25*Mgrad2_A*overlap -1.5/De/detB *overlap*detBgrad;

  Vgrad_A = (1.5/De*trace(Omega*Binv) + dot(u,Omega*u))*Mgrad_A + (1.5/De*Vgrad2_A + ugrad_A)*overlap;
  Vgrad_s = (1.5/De*trace(Omega*Binv) + dot(u,Omega*u))*0.5*Binv*v*overlap + (Omega*Binv*u)*overlap;
  return overlap*(1.5/De*trace(Omega*Binv) + dot(u,Omega*u));
}
