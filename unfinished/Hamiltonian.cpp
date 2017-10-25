#include "Hamiltonian.h"

Hamiltonian::Hamiltonian(size_t K, size_t n, MatrixElements matElem)
: K(K), n(n), matElem(matElem){
  H = zeros<mat>(K,K);
  B = zeros<mat>(K,K);

  basis = zeros<cube>(3*n,3*n,K);
  shift = zeros<cube>(3*n,1,K);
  coeffs = zeros<vec>(K);
  Emin = zeros<vec>(K);

  for (size_t i = 0; i < K; i++) {
    basis.slice(i) = GaussianGenerator::genMatrix(n,1);
    shift.slice(i) = GaussianGenerator::genShift(n,0);
    coeffs(i)      = GaussianGenerator::genCoeff(0,1);
  }

  double Hij, Bij, Ck1, Ck2;
  mat A1 = zeros<mat>(3*n,3*n), A2 = zeros<mat>(3*n,3*n);
  vec s1 = zeros<vec>(3*n), s2 = zeros<vec>(3*n);
  for (size_t i = 0; i < K; i++) {
    A1 = basis.slice(i);
    s1 = shift.slice(i);
    Ck1 = coeffs(i);

    for (size_t j = i; j < K; j++) {
      A2 = basis.slice(j);
      s2 = shift.slice(j);
      Ck2 = coeffs(j);

      matElem.calculateH(A1,A2,s1,s2,Ck1,Ck2,Hij,Bij);
      H(i,j) = Hij;
      H(j,i) = Hij;
      B(i,j) = Bij;
      B(j,i) = Bij;
    }
  }

  Emin = Evals::eigenSpectrum(H,B);
}

mat Hamiltonian::getH(){
  return H;
}

mat Hamiltonian::getB(){
  return B;
}

vec Hamiltonian::OptByRound(size_t rounds, size_t lambda, vec& xstart){

  double Hij, Bij, Ck1, Ck2;
  mat A1 = zeros<mat>(3*n,3*n), A2 = zeros<mat>(3*n,3*n);
  vec s1 = zeros<vec>(3*n), s2 = zeros<vec>(3*n);

  for (size_t round = 0; round < rounds; round++) {

    for (size_t b = 0; b < K; b++) {

      Acur = basis;
      Scur = shift;
      Ccur = coeffs;

      for (int k = 0; k < lambda; k++) {
        Atry.slice(k) = GaussianGenerator::genMatrix(n,MEAN);
        Stry.slice(k) = GaussianGenerator::genMatrix(n,WIDTH);
        Ctry(k)       = GaussianGenerator::genCoeff(MEAN,WIDTH);
        Acur.slice(b) = Atry.slice(k);
        Scur.slice(b) = Stry.slice(k);
        Ccur(b)       = Ctry(k);

        for (int i = 0; i < K; i++) {
          for (int j = i; j < K; j++) {
            if (i == b || j == b) {
              matElem.calculateH(Acur.slice(i),Acur.slice(j),Scur.slice(i),Scur.slice(j),Ccur(i),Ccur(j),Hij,Bij);
              H(i,j) = Hij;
              H(j,i) = Hij;
              B(i,j) = Bij;
              B(j,i) = Bij;
            }
          }
        }

        Btry.slice(k) = B;
        Htry.slice(k) = H;

        mat L = chol(B,"lower");
        EigVal.col(k) = sort(eig_sym(L.i()*H*(L.i()).t()));
      }

      uword index;
      double minVal = EigVal.row(0).min(index);

      if (minVal < Emin(0)) {
        basis.slice(b)  = Atry.slice(index);
        shift.col(b)    = Stry.slice(index);
        coeffs(b)       = Ctry(index);
        H               = Htry.slice(index);
        B               = Btry.slice(index);
        Hbest           = H;
        Bbest           = B;
        Emin            = EigVal.col(index);
      }
      else{
        H               = Hbest;
        B               = Bbest;
      }

    }
  }
}
