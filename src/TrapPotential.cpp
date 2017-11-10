#include "TrapPotential.h"

TrapPotential::TrapPotential(System& sys)
  : n(sys.n), lambdamat(sys.lambdamat) {

}

TrapPotential::TrapPotential(System& sys, vec& trapFreq)
  : n(sys.n), lambdamat(sys.lambdamat) {

  updateTrap(trapFreq);
}

void TrapPotential::updateTrap(vec& trapFreq){
  // Omega = zeros<mat>(3*n,3*n);
  // mat Zmat = zeros<mat>(3,3);
  // Zmat(2,2) = 1;
  //
  // for (size_t i = 0; i<n; i++){
  //   int ibegin = 3*i;
  //   int iend = 3*i+2;
  //   for (size_t j = 0; j<n; j++){
  //     int jbegin = 3*j;
  //     int jend = 3*j+2;
  //     for (size_t k = 0; k < n+1; k++) {
  //       int kbegin = 3*k;
  //       int kend = 3*k+2;
  //
  //       mat Uik = U(span(ibegin,iend),span(kbegin,kend));
  //       mat Ujk = U(span(jbegin,jend),span(kbegin,kend));
  //       Omega(span(ibegin,iend),span(jbegin,jend)) += Zmat*0.5*Uik*Ujk*masses(k)*trapFreq(k)*trapFreq(k);
  //     }
  //   }
  // }
  mat Zmat = zeros<mat>(3,3);
  Zmat(2,2) = 1;
  Omega = lambdamat%Zmat * 0.5 * trapFreq(0); // trapFreq(0) = 1/b^4

  cout << Omega << endl;
}

double TrapPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  return 999999999;
}

double TrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  // calculate <g|x^t*v*v^t*x|g> , where v*v^t = Zmat
  return pow(datum::pi,3.0*n/2.0)/sqrt(detB) * 0.5*trace(Omega*Binv);

}
