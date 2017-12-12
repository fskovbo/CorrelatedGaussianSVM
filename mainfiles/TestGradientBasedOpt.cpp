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

  size_t De             = 3;
  vec masses            = {1 , 1 , 1};
  vec charges           = {0 , 0 , 0};
  auto Test             = System(masses,charges,De);

  auto Trap             = TrapPotential(Test);
  auto Gauss            = SingleGaussPotential(Test);
  PotentialList Vstrat  = {&Gauss, &Trap};
  auto elem             = MatrixElements(Test,Vstrat);
  auto ansatz           = Variational(Test,elem);

  Trap.updateTrap(1e-2);
  ansatz.initializeBasis(5);

  vec aGuess            = {0.5*1e-2 , 2.5 , 2.5};
  vec res1              = ansatz.sweepStochastic(0,5,1e4,aGuess);
  vec res2              = ansatz.sweepDeterministic_grad(5);

  cout << res2(res2.n_rows-1)-Trap.gsExpectedVal() << endl;

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
