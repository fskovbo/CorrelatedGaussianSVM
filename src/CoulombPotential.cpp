#include "CoulombPotential.h"

CoulombPotential::CoulombPotential(System& sys, size_t expansionterms, double range)
: n(sys.n), De(sys.De) {
  bvec = zeros<vec>(expansionterms);
  cvec = zeros<vec>(expansionterms);
  Utils::invExpansion(bvec,cvec,0.001,range,1e4);

  vec alpha = 1.0/pow(bvec,2);
  interactions = buildInteractions(alpha,sys.Ui);
  Qinter = Utils::buildQinter(sys.charges);
}

fdcube CoulombPotential::buildInteractions(vec& alpha, mat& Ui){
  size_t N = n+1;
  size_t nalp = alpha.n_rows;
  fdcube Rcube(n*De,n*De,nalp,N*(N-1)/2);
  size_t interactionNr = 0;

  for (size_t i = 0; i<n; i++){
      size_t ibegin = De*i;
      size_t iend = De*i+(De-1);

      for (size_t j = i+1; j<N; j++){
          size_t jbegin = De*j;
          size_t jend = De*j+(De-1);

          cube interactionCube = zeros<cube>(De*n,De*n,nalp);
          for (size_t k = 0; k < nalp; k++) {
            mat interaction = zeros<mat>(De*N,De*N);
            interaction(span(ibegin,iend),span(ibegin,iend)) = eye(De,De);
            interaction(span(jbegin,jend),span(jbegin,jend)) = eye(De,De);
            interaction(span(ibegin,iend),span(jbegin,jend)) = -eye(De,De);
            interaction(span(jbegin,jend),span(ibegin,iend)) = -eye(De,De);

            interaction *= alpha(k);
            mat RFull = Ui.t()*interaction*Ui;
            interactionCube.slice(k) = RFull(span(0,De*n-1),span(0,De*n-1));
          }

          Rcube.setPiece(interactionNr,interactionCube);
          interactionNr++;
      }
  }
  return Rcube;
}

double CoulombPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  mat B = A1 + A2;
  vec v = 2.0*A1*s1 + 2.0*A2*s2;

  size_t k = interactions.n_slices();
  size_t p = interactions.n_pieces();
  mat kappamat(p,k), kappa(De*n,De*n);
  cube Ri(De*n,De*n,k);

  for (size_t i = 0; i < p; i++) {
    Ri = interactions.getPiece(i);
    for (size_t j = 0; j < k; j++) {
      kappa = B + Ri.slice(j);
      kappamat(i,j) = pow(det(kappa),-3.0/De/2.0) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 1/4.0 *dot(v,kappa.i()*v));
    }
  }

  return pow(datum::pi,3.0*n/2.0)*dot(Qinter,kappamat*cvec);
}

double CoulombPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  mat B = A1 + A2;

  size_t k = interactions.n_slices();
  size_t p = interactions.n_pieces();
  mat kappamat(p,k), kappa(De*n,De*n);
  cube Ri(De*n,De*n,k);

  for (size_t i = 0; i < p; i++) {
    Ri = interactions.getPiece(i);
    for (size_t j = 0; j < k; j++) {
      kappa = B + Ri.slice(j);
      kappamat(i,j) = pow(det(kappa),-3.0/De/2.0);
    }
  }

  return pow(datum::pi,3.0*n/2.0)*dot(Qinter,kappamat*cvec);
}

double CoulombPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad){
  return 9999999;
}

double CoulombPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2,
                                                    mat& Binv, double detB, vec& Vgrad_A, vec& Vgrad_s,
                                                    cube& Binvgrad, vec& detBgrad)
{
  return 999999;
}
