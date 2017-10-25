#include "TrapPotential.h"

TrapPotential::TrapPotential(System& sys, double trapFreq) {
  n = sys.n; // for absolute coordinates n = N
  masses = sys.masses;
  updateTrap(trapFreq);
}

void TrapPotential::updateTrap(double trapFreq){
  //
  //  Build Zmat (PERHAPS ADD REDUCED MASSES AND OMEGA HERE)
  //
  vec wzsingle;
  Zmat = zeros<mat>(3*n,3*n);
  for (size_t i = 0; i < n; i++) {
    wzsingle = zeros<vec>(3*n);
    wzsingle(3*i+2) = 1;
    Zmat += 0.5*masses(i)*trapFreq*trapFreq* wzsingle*wzsingle.t();
  }
}

double TrapPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  // calculate <g|x^t*v*v^t*x|g> , where v*v^t = Zmat
  vec v = 2.0*A1*s1 + 2.0*A2*s2;
  vec u = 0.5*Binv*v;

  double overlap = pow(datum::pi,3.0*n/2.0)/sqrt(detB) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 1/4.0 *dot(v,Binv*v));

  return overlap*(dot(u,Zmat*u) + 0.5*trace(Zmat*Binv));
}

double TrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  // calculate <g|x^t*v*v^t*x|g> , where v*v^t = Zmat
  return pow(datum::pi,3.0*n/2.0)/sqrt(detB) * 0.5*trace(Zmat*Binv);

}
