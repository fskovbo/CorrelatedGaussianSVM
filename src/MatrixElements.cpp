#include "MatrixElements.h"

MatrixElements::MatrixElements(System& sys, PotentialStrategy& Vstrat)
: Vstrat(Vstrat), n(sys.n), De(sys.De), lambda(sys.lambdamat){

}

void MatrixElements::calculateH(mat& A1, mat& A2, vec& s1, vec& s2, double& Hij, double& Bij){
  mat B = A1 + A2;
  mat Bi = inv_sympd(B);
  double detB = det(B);
  vec v = 2.0*A1*s1 + 2.0*A2*s2;
  vec u = 1/2.0 * Bi*v;
  mat prod12 = A1*0.5*lambda*A2;

  double overlap = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 1/4.0 *dot(v,Bi*v));
  double T = overlap*(6.0/De*trace(prod12*Bi) + 4.0*dot(u-s1,prod12*(u-s2)));
  double V = Vstrat.calculateExpectedPotential(A1, A2, s1, s2, Bi, detB);

  Hij = T+V;
  Bij = overlap;
}

void MatrixElements::calculateH_noShift(mat& A1, mat& A2, double& Hij, double& Bij){
  mat B = A1 + A2;
  mat Bi = inv_sympd(B);
  double detB = det(B);
  mat prod12 = A1*0.5*lambda*A2;

  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0);
  double T = overlap*(6.0/De*trace(prod12*Bi) );
  double V = Vstrat.calculateExpectedPotential_noShift(A1, A2, Bi, detB);

  Hij = T+V;
  Bij = overlap;
}
