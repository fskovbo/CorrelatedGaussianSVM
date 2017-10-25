#include "CoulombPotential.h"

CoulombPotential::CoulombPotential(System& sys, size_t expansionterms, double range) {
  bvec = zeros<vec>(expansionterms);
  cvec = zeros<vec>(expansionterms);
  Utils::invExpansion(bvec,cvec,0.001,range,1e4);

  vec alpha = 1.0/pow(bvec,2);
  interactions = Utils::buildInteraction(sys.N,alpha,sys.Ui);
  Qinter = Utils::buildQinter(sys.charges);
}

double CoulombPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  mat B = A1 + A2;
  vec v = 2.0*A1*s1 + 2.0*A2*s2;

  size_t k = interactions.n_slices();
  size_t p = interactions.n_pieces();
  size_t n = v.n_rows/3;
  mat kappamat(p,k), kappa(3*n,3*n);
  cube Ri(3*n,3*n,k);

  for (size_t i = 0; i < p; i++) {
    Ri = interactions.getPiece(i);
    for (size_t j = 0; j < k; j++) {
      kappa = B + Ri.slice(j);
      kappamat(i,j) = 1.0/(sqrt(det(kappa))) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 1/4.0 *dot(v,kappa.i()*v));
    }
  }

  return pow(datum::pi,3.0*n/2.0)*dot(Qinter,kappamat*cvec);
}

double CoulombPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  mat B = A1 + A2;

  size_t k = interactions.n_slices();
  size_t p = interactions.n_pieces();
  size_t n = A1.n_rows/3;
  mat kappamat(p,k), kappa(3*n,3*n);
  cube Ri(3*n,3*n,k);

  for (size_t i = 0; i < p; i++) {
    Ri = interactions.getPiece(i);
    for (size_t j = 0; j < k; j++) {
      kappa = B + Ri.slice(j);
      kappamat(i,j) = 1.0/(sqrt(det(kappa)));
    }
  }

  return pow(datum::pi,3.0*n/2.0)*dot(Qinter,kappamat*cvec);
}
