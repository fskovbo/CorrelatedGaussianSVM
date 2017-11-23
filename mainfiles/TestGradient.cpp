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
  PotentialList Vstrat  = {&Trap};

  auto elem             = MatrixElements(Test,Vstrat);


  Trap.updateTrap(1);
  mat A1 = 3*eye<mat>(De,De);
  mat A2 = 2*eye<mat>(De,De);
  double Hij,Bij;
  vec Hgrad,Bgrad;

  elem.calculateH_noShift(A2,A1,Hij,Bij,Hgrad,Bgrad);

  cout << Hgrad << endl;
  cout << Bgrad << endl;

  return 0;
}
