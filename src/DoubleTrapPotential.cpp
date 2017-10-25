#include "DoubleTrapPotential.h"

DoubleTrapPotential::DoubleTrapPotential(vec& omegasqY, vec& omegasqZ, cube& Q)
: omegasqY(omegasqY), omegasqZ(omegasqZ), Q(Q){

}

double DoubleTrapPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  //midlertidig løsning for m1=m2=m3=m=1
  //omegasq er frontfaktorer
  mat B = A1 + A2;
  vec v = 2.0*A1*s1 + 2.0*A2*s2;
  vec u = 0.5*B.i()*v;

  size_t n = v.n_rows/3;

  double overlap = (pow(datum::pi,3.0*n/2.0)/sqrt(det(B))) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 1/4.0 *dot(v,B.i()*v));

  if ( n == 2) {
    mat Qmat = zeros<mat>(6,6);
    Qmat(1,1) = omegasqY(0);
    Qmat(4,4) = omegasqY(1);
    Qmat(2,2) = omegasqZ(0);
    Qmat(5,5) = omegasqZ(1);
    return overlap*(dot(u,Qmat*u) + 0.5*trace(Qmat*B.i()));
  }

  return 0;

}

double DoubleTrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  //midlertidig løsning for m1=m2=m3=m=1
  //omegasq er frontfaktorer
  mat B = A1 + A2;

  size_t n = A1.n_rows/3;

  double overlap = pow(datum::pi,3.0*n/2.0)/sqrt(det(B));

  if ( n == 2) {
    mat Qmat = zeros<mat>(6,6);
    Qmat(1,1) = omegasqY(0);
    Qmat(4,4) = omegasqY(1);
    Qmat(2,2) = omegasqZ(0);
    Qmat(5,5) = omegasqZ(1);
    return overlap*(0.5*trace(Qmat*B.i()));
  }

  return 0;

}
