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

int main() {

  clock_t begin = clock();
  arma_rng::set_seed_random();

  size_t De             = 3;
  vec masses            = {1 , 1};
  vec charges           = {0 , 0};
  auto Test             = System(masses,charges,De);

  auto Trap             = TrapPotential(Test);
  auto Gauss            = SingleGaussPotential(Test);
  PotentialList Vstrat  = {&Gauss};
  auto elem             = MatrixElements(Test,Vstrat);


  Trap.updateTrap(1e-2);


  double Hij,Bij;
  vec Hg, Bg;

  size_t Npts = 1e2;
  mat data(Npts,5);
  vec Avals = linspace(4950,5050,Npts);
  for (size_t i = 0; i < Npts; i++) {
    mat A = Avals(i)*eye<mat>(De,De);
    elem.calculateH_noShift(A,A,Hij,Bij,Hg,Bg);
    data(i,0) = Avals(i);
    data(i,1) = Hij/Bij - Trap.gsExpectedVal();
    data(i,2) = (2*Hg(0)-Hij/Bij *2* Bg(0))/Bij;
    data(i,3) = NumGradient(elem,A,1e-4);
    data(i,4) = data(i,2)-data(i,3);
  }
  // data.save("parameterspace.txt", arma_ascii);

  cout << data << endl;

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
