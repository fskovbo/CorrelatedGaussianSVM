#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <iostream>
#include <armadillo>
#include <vector>

#include "System.h"
#include "MatrixElements.h"


inline vec NumGradient(System& sys, MatrixElements& elem, mat& A){
  size_t count = 0;
  vec** vA;
  vec v, g(sys.De*sys.n*(sys.n+1)/2);
  double eps = 1e-6;

  for (size_t i = 0; i < sys.n+1; i++) {
    for (size_t j = i+1; j < sys.n+1; j++) {
      for (size_t k = 0; k < sys.De; k++) {
        vA = sys.vArrayList.at(k);
        v  = vA[i][j];

        mat temp = eps*v*v.t();
        mat Ap = A+temp;
        mat Am = A-temp;
        double Hp, Hm, Bp, Bm;
        elem.calculateH_noShift(Ap,Ap,Hp,Bp);
        elem.calculateH_noShift(Am,Am,Hm,Bm);

        g(count) = (Hp/Bp - Hm/Bm)/(2*eps);
        count++;
      }
    }
  }

  return g;
}

inline vec NumGradient(System& sys, MatrixElements& elem, mat& A, vec& s){
  size_t count = 0;
  vec** vA;
  vec v, g(sys.De*sys.n*(sys.n+1)/2 + sys.De*sys.n);
  double eps = 1e-6;

  for (size_t i = 0; i < sys.n+1; i++) {
    for (size_t j = i+1; j < sys.n+1; j++) {
      for (size_t k = 0; k < sys.De; k++) {
        vA = sys.vArrayList.at(k);
        v  = vA[i][j];

        mat temp = eps*v*v.t();
        mat Ap = A+temp;
        mat Am = A-temp;
        double Hp, Hm, Bp, Bm;
        elem.calculateH(Ap,Ap,s,s,Hp,Bp);
        elem.calculateH(Am,Am,s,s,Hm,Bm);

        g(count) = (Hp/Bp - Hm/Bm)/(2*eps);
        count++;
      }
    }
  }
  for (size_t i = 0; i < sys.De*sys.n; i++) {
    vec temp = zeros<vec>(sys.De*sys.n);
    temp(i) = 1;
    vec sp = s+eps*temp;
    vec sm = s-eps*temp;

    double Hp, Hm, Bp, Bm;
    elem.calculateH(A,A,sp,sp,Hp,Bp);
    elem.calculateH(A,A,sm,sm,Hm,Bm);

    g(count) = (Hp/Bp - Hm/Bm)/(2*eps);
    count++;
  }

  return g;
}

inline vec NumGradient(System& sys, MatrixElements& elem, mat& Afixed, mat& Avar){
  size_t count = 0;
  vec** vA;
  vec v, g(sys.De*sys.n*(sys.n+1)/2);
  double eps = 1e-6;

  for (size_t i = 0; i < sys.n+1; i++) {
    for (size_t j = i+1; j < sys.n+1; j++) {
      for (size_t k = 0; k < sys.De; k++) {
        vA = sys.vArrayList.at(k);
        v  = vA[i][j];

        mat temp = eps*v*v.t();
        mat Ap = Avar+temp;
        mat Am = Avar-temp;
        double Hp, Hm, Bp, Bm;
        elem.calculateH_noShift(Afixed,Ap,Hp,Bp);
        elem.calculateH_noShift(Afixed,Am,Hm,Bm);

        g(count) = (Hp/Bp - Hm/Bm)/(2*eps);
        count++;
      }
    }
  }

  return g;
}

inline vec NumGradient(System& sys, MatrixElements& elem, mat& Afixed, mat& Avar, vec& sfixed, vec& svar){
  size_t count = 0;
  vec** vA;
  vec v, g(sys.De*sys.n*(sys.n+1)/2 + sys.De*sys.n);
  double eps = 1e-6;

  for (size_t i = 0; i < sys.n+1; i++) {
    for (size_t j = i+1; j < sys.n+1; j++) {
      for (size_t k = 0; k < sys.De; k++) {
        vA = sys.vArrayList.at(k);
        v  = vA[i][j];

        mat temp = eps*v*v.t();
        mat Ap = Avar+temp;
        mat Am = Avar-temp;
        double Hp, Hm, Bp, Bm;
        elem.calculateH(Afixed,Ap,sfixed,svar,Hp,Bp);
        elem.calculateH(Afixed,Am,sfixed,svar,Hm,Bm);

        g(count) = (Hp/Bp - Hm/Bm)/(2*eps);
        count++;
      }
    }
  }
  for (size_t i = 0; i < sys.De*sys.n; i++) {
    vec temp = zeros<vec>(sys.De*sys.n);
    temp(i) = 1;
    vec sp = svar+eps*temp;
    vec sm = svar-eps*temp;

    double Hp, Hm, Bp, Bm;
    elem.calculateH(Afixed,Avar,sfixed,sp,Hp,Bp);
    elem.calculateH(Afixed,Avar,sfixed,sm,Hm,Bm);

    g(count) = (Hp/Bp - Hm/Bm)/(2*eps);
    count++;
  }

  return g;
}

inline vec NumGradient(System& sys, MatrixElements& elem, cube& basis, size_t index){
  std::vector<mat> dH;
  std::vector<mat> dM;
  size_t count = 0;
  size_t K = basis.n_slices;
  vec** vA;
  vec v, g(sys.De*sys.n*(sys.n+1)/2);
  double eps = 1e-2;
  mat Hp(K,K), Hm(K,K), Bp(K,K), Bm(K,K), H(K,K), B(K,K);
  double Hxyp, Bxyp, Hxym, Bxym, Hxy, Bxy;

  for (size_t i = 0; i < sys.n+1; i++) {
    for (size_t j = i+1; j < sys.n+1; j++) {
      for (size_t k = 0; k < sys.De; k++) {
        vA = sys.vArrayList.at(k);
        v  = vA[i][j];

        mat temp = eps*v*v.t();

        for (size_t x = 0; x < K; x++) {
          mat Ax = basis.slice(x);
          for (size_t y = x; y < K; y++) {
            mat Ay = basis.slice(y);

            elem.calculateH_noShift(Ax,Ay,Hxy,Bxy);
            H(x,y) = Hxy;
            H(y,x) = Hxy;
            B(x,y) = Bxy;
            B(y,x) = Bxy;
          }
        }

        Hp = Hm = H;
        Bp = Bm = B;

        mat Ap = basis.slice(index) + temp;
        mat Am = basis.slice(index) - temp;
        for (size_t x = 0; x < K; x++) {
          mat Ax = basis.slice(x);

          if (x == index) {
            elem.calculateH_noShift(Ap,Ap,Hxyp,Bxyp);
            elem.calculateH_noShift(Am,Am,Hxym,Bxym);
          }
          else{
            elem.calculateH_noShift(Ax,Ap,Hxyp,Bxyp);
            elem.calculateH_noShift(Ax,Am,Hxym,Bxym);
          }
          Hp(x,index) = Hxyp; Hm(x,index) = Hxym;
          Hp(index,x) = Hxyp; Hm(index,x) = Hxym;
          Bp(x,index) = Bxyp; Bm(x,index) = Bxym;
          Bp(index,x) = Bxyp; Bm(index,x) = Bxym;
        }
        dH.emplace_back((Hp-Hm)/(2*eps));
        dM.emplace_back((Bp-Bm)/(2*eps));

        mat Lp(K,K), Lm(K,K);
        vec eigvalp, eigvalm;
        mat eigvecp, eigvecm;
        chol(Lp,Bp,"lower");
        chol(Lm,Bm,"lower");
        eig_sym(eigvalp,eigvecp, Lp.i()*Hp*(Lp.t()).i() );
        eig_sym(eigvalm,eigvecm, Lm.i()*Hm*(Lm.t()).i() );

        g(count) = ( eigvalp(0) - eigvalm(0) )/(2*eps);

        count++;
      }
    }
  }

  return g;
}

inline vec NumGradient(System& sys, MatrixElements& elem, cube& basis, mat& shift, size_t index){
  std::vector<mat> dH;
  std::vector<mat> dM;
  size_t count = 0;
  size_t K = basis.n_slices;
  vec** vA;
  vec v, g(sys.De*sys.n*(sys.n+1)/2 + sys.De*sys.n);
  double eps = 1e-3;
  mat Hp(K,K), Hm(K,K), Bp(K,K), Bm(K,K), H(K,K), B(K,K);
  double Hxyp, Bxyp, Hxym, Bxym, Hxy, Bxy;

  for (size_t i = 0; i < sys.n+1; i++) {
    for (size_t j = i+1; j < sys.n+1; j++) {
      for (size_t k = 0; k < sys.De; k++) {
        vA = sys.vArrayList.at(k);
        v  = vA[i][j];

        mat temp = eps*v*v.t();

        for (size_t x = 0; x < K; x++) {
          mat Ax = basis.slice(x);
          vec sx = shift.col(x);
          for (size_t y = x; y < K; y++) {
            mat Ay = basis.slice(y);
            vec sy = shift.col(y);

            elem.calculateH(Ax,Ay,sx,sy,Hxy,Bxy);
            H(x,y) = Hxy;
            H(y,x) = Hxy;
            B(x,y) = Bxy;
            B(y,x) = Bxy;
          }
        }

        Hp = Hm = H;
        Bp = Bm = B;

        mat Ap = basis.slice(index) + temp;
        mat Am = basis.slice(index) - temp;
        vec sf = shift.col(index);
        for (size_t x = 0; x < K; x++) {
          mat Ax = basis.slice(x);
          vec sx = shift.col(x);

          if (x == index) {
            elem.calculateH(Ap,Ap,sx,sx,Hxyp,Bxyp);
            elem.calculateH(Am,Am,sx,sx,Hxym,Bxym);
          }
          else{
            elem.calculateH(Ax,Ap,sx,sf,Hxyp,Bxyp);
            elem.calculateH(Ax,Am,sx,sf,Hxym,Bxym);
          }
          Hp(x,index) = Hxyp; Hm(x,index) = Hxym;
          Hp(index,x) = Hxyp; Hm(index,x) = Hxym;
          Bp(x,index) = Bxyp; Bm(x,index) = Bxym;
          Bp(index,x) = Bxyp; Bm(index,x) = Bxym;
        }
        dH.emplace_back((Hp-Hm)/(2*eps));
        dM.emplace_back((Bp-Bm)/(2*eps));

        mat Lp(K,K), Lm(K,K);
        vec eigvalp, eigvalm;
        mat eigvecp, eigvecm;
        chol(Lp,Bp,"lower");
        chol(Lm,Bm,"lower");

        eig_sym(eigvalp,eigvecp, Lp.i()*Hp*(Lp.t()).i() );
        eig_sym(eigvalm,eigvecm, Lm.i()*Hm*(Lm.t()).i() );

        g(count) = ( eigvalp(0) - eigvalm(0) )/(2*eps);

        count++;
      }
    }
  }

  for (size_t i = 0; i < sys.De*sys.n; i++) {
    vec temp = zeros<vec>(sys.De*sys.n);
    temp(i) = 1;
    vec sp = shift.col(index) + eps*temp;
    vec sm = shift.col(index) - eps*temp;
    mat Af = basis.slice(index);

    Hp = Hm = H;
    Bp = Bm = B;

    for (size_t x = 0; x < K; x++) {
      mat Ax = basis.slice(x);
      vec sx = shift.col(x);

      if (x == index) {
        elem.calculateH(Ax,Ax,sp,sp,Hxyp,Bxyp);
        elem.calculateH(Ax,Ax,sm,sm,Hxym,Bxym);
      }
      else{
        elem.calculateH(Ax,Af,sx,sp,Hxyp,Bxyp);
        elem.calculateH(Ax,Af,sx,sm,Hxym,Bxym);
      }
      Hp(x,index) = Hxyp; Hm(x,index) = Hxym;
      Hp(index,x) = Hxyp; Hm(index,x) = Hxym;
      Bp(x,index) = Bxyp; Bm(x,index) = Bxym;
      Bp(index,x) = Bxyp; Bm(index,x) = Bxym;
    }
    dH.emplace_back((Hp-Hm)/(2*eps));
    dM.emplace_back((Bp-Bm)/(2*eps));

    mat Lp(K,K), Lm(K,K);
    vec eigvalp, eigvalm;
    mat eigvecp, eigvecm;

    chol(Lp,Bp,"lower");
    chol(Lm,Bm,"lower");
    eig_sym(eigvalp,eigvecp, Lp.i()*Hp*(Lp.t()).i() );
    eig_sym(eigvalm,eigvecm, Lm.i()*Hm*(Lm.t()).i() );

    g(count) = ( eigvalp(0) - eigvalm(0) )/(2*eps);

    count++;
  }

  return g;
}

inline mat buildA(System* sys){
  vec vals = 1e4*randu<vec>(sys->De * sys->n * (sys->n +1)/2);
  vec** vA;
  vec v;
  mat A = zeros<mat>(sys->De * sys->n , sys->De * sys->n);
  size_t count = 0;
  for (size_t i = 0; i < sys->n+1; i++) {
    for (size_t j = i+1; j < sys->n+1; j++) {
      for (size_t k = 0; k < sys->De; k++) {
        vA   = sys->vArrayList.at(k);
        v    = vA[i][j];
        A   += vals(count)*v*v.t();
        count++;
      }
    }
  }
  return A;
}

inline vec builds(System* sys){
  return 1e-1 * randu<vec>(sys->De * sys->n);
}

inline void buildA(System* sys1, System* sys3, mat& A1, mat& A3){
  vec vals = 1e4*randu<vec>(sys1->n * (sys1->n +1)/2);
  vec** vA;
  vec v;
  A1 = zeros<mat>(sys1->De * sys1->n , sys1->De * sys1->n);
  A3 = zeros<mat>(sys3->De * sys3->n , sys3->De * sys3->n);

  size_t count = 0;
  for (size_t i = 0; i < sys3->n+1; i++) {
    for (size_t j = i+1; j < sys3->n+1; j++) {
      for (size_t k = 0; k < sys3->De; k++) {
        vA   = sys3->vArrayList.at(k);
        v    = vA[i][j];
        A3  += vals(count)*v*v.t();
      }
      count++;
    }
  }
  count = 0;
  for (size_t i = 0; i < sys1->n+1; i++) {
    for (size_t j = i+1; j < sys1->n+1; j++) {
      for (size_t k = 0; k < sys1->De; k++) {
        vA   = sys1->vArrayList.at(k);
        v    = vA[i][j];
        A1  += vals(count)*v*v.t();
      }
      count++;
    }
  }
}

#endif
