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


int main() {

  clock_t begin = clock();
  arma_rng::set_seed_random();

  size_t De             = 1;
  vec masses            = {1 , 1};
  vec charges           = {0 , 0};
  auto Test             = System(masses,charges,De);

  auto Trap             = TrapPotential(Test);
  auto Gauss            = SingleGaussPotential(Test);
  PotentialList Vstrat  = {&Gauss,&Trap};

  auto elem             = MatrixElements(Test,Vstrat);


  Trap.updateTrap(1e-2);


  double Hij,Bij;
  vec dumme;

  size_t Npts = 1e6;
  mat data(Npts,2);
  vec Avals = linspace(4950,5050,Npts);
  for (size_t i = 0; i < Npts; i++) {
    mat A = Avals(i)*eye<mat>(De,De);
    elem.calculateH_noShift(A,A,Hij,Bij,dumme,dumme);
    data(i,0) = Avals(i);
    data(i,1) = Hij/Bij - Trap.gsExpectedVal();
  }
  data.save("parameterspace.txt", arma_ascii);

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
