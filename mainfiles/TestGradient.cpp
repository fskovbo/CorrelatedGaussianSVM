#include <iostream>
#include <armadillo>
#include <time.h>
#include <vector>

#include "System.h"
#include "SingleGaussPotential.h"
#include "MatrixElements.h"
#include "Variational.h"
#include "TrapPotential.h"
#include "PotentialList.h"


double NumGradient(MatrixElements& elem, mat& A, double eps){
  mat temp = zeros<mat>(A.n_rows,A.n_cols);
  temp(0,0) = eps;
  mat Ap = A+temp;
  mat Am = A-temp;
  double Hp, Hm, Bp, Bm;
  elem.calculateH_noShift(Ap,Ap,Hp,Bp);
  elem.calculateH_noShift(Am,Am,Hm,Bm);

  return (Hp/Bp - Hm/Bm)/(2*eps);
}

double NumGradient(System& sys, MatrixElements& elem, PotentialList& Vstrat, mat& A, double eps){
  vec** vA = sys.vArrayList.at(1);
  vec v = vA[0][1];

  mat temp = eps*v*v.t();
  mat Ap = A+temp;
  mat Am = A-temp;
  double Hp, Hm, Bp, Bm;
  elem.calculateH_noShift(Ap,Ap,Hp,Bp);
  elem.calculateH_noShift(Am,Am,Hm,Bm);

  return (Hp/Bp - Hm/Bm)/(2*eps);

  // double detAp = det(Ap+Ap);
  // mat invAp = (Ap+Ap).i();
  // double Vp = Vstrat.calculateExpectedPotential_noShift(Ap,Ap,invAp,detAp);
  // double detAm = det(Am+Am);
  // mat invAm = (Am+Am).i();
  // double Vm = Vstrat.calculateExpectedPotential_noShift(Am,Am,invAm,detAm);
  //
  // return (Vp-Vm)/(2*eps);
}

double NumGradient(System& sys, MatrixElements& elem, mat& A, mat& A2, double eps){
  vec** vA = sys.vArrayList.at(0);
  vec v = vA[0][1];

  mat temp = eps*v*v.t();
  mat Ap = A+temp;
  mat Am = A-temp;
  double Hp, Hm, Bp, Bm, Hij, Bij;
  mat HP(2,2), BP(2,2), HM(2,2), BM(2,2);
  elem.calculateH_noShift(Ap,Ap,Hp,Bp);
  HP(0,0) = Hp; BP(0,0) = Bp;
  elem.calculateH_noShift(Ap,A2,Hij,Bij);
  HP(1,0) = Hij; BP(1,0) = Bij;
  HP(0,1) = Hij; BP(0,1) = Bij;
  elem.calculateH_noShift(A2,A2,Hij,Bij);
  HP(1,1) = Hij; BP(1,1) = Bij;
  cx_vec eigvalP;
  cx_mat eigvecP;
  eig_pair(eigvalP,eigvecP,HP,BP);

  elem.calculateH_noShift(Am,Am,Hm,Bm);
  HM(0,0) = Hm; BM(0,0) = Bm;
  elem.calculateH_noShift(Am,A2,Hij,Bij);
  HM(1,0) = Hij; BM(1,0) = Bij;
  HM(0,1) = Hij; BM(0,1) = Bij;
  elem.calculateH_noShift(A2,A2,Hij,Bij);
  HM(1,1) = Hij; BM(1,1) = Bij;
  cx_vec eigvalM;
  cx_mat eigvecM;
  eig_pair(eigvalM,eigvecM,HM,BM);

  return real(eigvalP(0) - eigvalM(0))/(2*eps);
}

int main() {

  clock_t begin = clock();
  arma_rng::set_seed_random();

  size_t De             = 3;
  vec masses            = {1 , 1 , 1};
  vec charges           = {0 , 0 , 0};
  auto Test             = System(masses,charges,De);

  auto Trap             = TrapPotential(Test);
  auto Gauss            = SingleGaussPotential(Test);
  PotentialList Vstrat  = {&Trap};
  auto elem             = MatrixElements(Test,Vstrat);
  auto ansatz           = Variational(Test,elem);


  Trap.updateTrap(1e-2);


  double Hij,Bij;
  vec Hg, Bg;

  size_t Npts = 1e2;
  mat data(Npts,5);
  vec Avals = linspace(0.01,1000,Npts);
  vec** vArray;

  for (size_t i = 0; i < Npts; i++) {
    mat A = zeros<mat>(Test.n*De,Test.n*De);
    // vec Alist = {Avals(i) , 1 , 1 };
    vec Alist = ones<vec>(9);
    Alist(0) = 5000;
    Alist(3) = 5000;
    Alist(6) = 5000;
    Alist(1) = Avals(i);

    size_t count = 0;
    for (size_t j = 0; j < Test.n+1; j++) {
      for (size_t k = j+1; k < Test.n+1; k++) {
        for (size_t l = 0; l < De; l++) {
          vArray = Test.vArrayList.at(l);
          A += Alist(De*count+l)* (vArray[j][k] * (vArray[j][k]).t());
        }
        count++;
      }
    }

    elem.calculateH_noShift(A,A,Hij,Bij,Hg,Bg);
    // mat B = A+A;
    // double detB = det(B);
    // mat Binv = B.i();
    // vec Vgrad(De*Test.n*(Test.n+1)/2), detBgrad;
    // cube Binvgrad;

    data(i,0) = Avals(i);
    data(i,1) = Hij/Bij - Trap.gsExpectedVal();
    data(i,2) = (2*Hg(1)-Hij/Bij *2* Bg(1))/Bij;
    // data(i,1) = Vstrat.calculateExpectedPotential_noShift(A,A,Binv,detB,Vgrad,Binvgrad,detBgrad);
    // data(i,2) = Vgrad(0);
    data(i,3) = NumGradient(Test,elem,Vstrat,A,1e-1);
    data(i,4) = data(i,2)-data(i,3);
  }

  // mat A2 = zeros<mat>(Test.n*De,Test.n*De);
  // vec A2list = {200,500,1000};
  // for (size_t j = 0; j < Test.n+1; j++) {
  //   for (size_t k = j+1; k < Test.n+1; k++) {
  //     for (size_t l = 0; l < De; l++) {
  //       vArray = Test.vArrayList.at(l);
  //       A2 += A2list(l)* (vArray[j][k] * (vArray[j][k]).t());
  //     }
  //   }
  // }
  //
  //
  // for (size_t i = 0; i < Npts; i++) {
  //   mat A = zeros<mat>(Test.n*De,Test.n*De);
  //   // vec Alist = {Avals(i) , 1 , 1 };
  //   vec Alist = {Avals(i) , Avals(i) ,Avals(i) };
  //
  //   for (size_t j = 0; j < Test.n+1; j++) {
  //     for (size_t k = j+1; k < Test.n+1; k++) {
  //       for (size_t l = 0; l < De; l++) {
  //         vArray = Test.vArrayList.at(l);
  //         A += Alist(l)* (vArray[j][k] * (vArray[j][k]).t());
  //       }
  //     }
  //   }
  //
  //   mat H(2,2), B(2,2), HG(2,2), BG(2,2);
  //
  //   elem.calculateH_noShift(A,A,Hij,Bij,Hg,Bg);
  //   H(0,0) = Hij; B(0,0) = Bij;
  //   HG(0,0) = 2*Hg(0); BG(0,0) = 2*Bg(0);
  //
  //   elem.calculateH_noShift(A,A2,Hij,Bij,Hg,Bg);
  //   H(1,0) = Hij; B(1,0) = Bij;
  //   H(0,1) = Hij; B(0,1) = Bij;
  //   HG(1,0) = Hg(0); BG(1,0) = Bg(0);
  //   HG(0,1) = Hg(0); BG(0,1) = Bg(0);
  //
  //   elem.calculateH_noShift(A2,A2,Hij,Bij,Hg,Bg);
  //   H(1,1) = Hij; B(1,1) = Bij;
  //   HG(1,1) = 2*Hg(0); BG(1,1) = 2*Bg(0);
  //
  //   mat L(2,2);
  //   vec eigval;
  //   mat eigvec;
  //   bool status = chol(L,B,"lower");
  //   eig_sym(eigval,eigvec, L.i()*H*(L.t()).i() );
  //
  //   data(i,0) = Avals(i);
  //   data(i,1) = eigval(0) - Trap.gsExpectedVal();
  //   data(i,2) = dot(eigvec.col(0), (HG-eigval(0)*BG) * eigvec.col(0))/dot(eigvec.col(0), B*eigvec.col(0));
  //   data(i,3) = NumGradient(Test,elem,A,A2,1e-6);
  //   data(i,4) = data(i,2)-data(i,3);
  // }


  data.save("parameterspaceTEST.txt", arma_ascii);

  cout << data << endl;

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
