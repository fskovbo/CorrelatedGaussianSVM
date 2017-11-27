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

  vec masses            = {1 , 1};
  vec charges           = {0 , 0};
  auto TwoPart          = System(masses,charges,1);

  auto Gauss            = SingleGaussPotential(TwoPart);
  auto Trap             = TrapPotential(TwoPart);
  PotentialList Vstrat  = {&Trap};

  auto elem             = MatrixElements(TwoPart,Vstrat);
  auto ansatz           = Variational(TwoPart,elem);

  size_t state          = 0;
  size_t Nvals          = 1;
  vec bs                = logspace<vec>(-2,-2,Nvals);
  mat data              = zeros<mat>(Nvals,2);

  for (size_t i = 0; i < Nvals; i++) {
    Trap.updateTrap(bs(i));
    ansatz.initializeBasis(1);

    vec aGuess    = {bs(i) , 2.5 , 2.5};
    vec maxShift  = {0.1*bs(i) , 1 , 1};
    // vec res1      = ansatz.sweepStochastic(state,5,1e2,aGuess);
    // vec res2      = ansatz.sweepDeterministic(state,5,2,{0,1,1});
    vec res1      = ansatz.sweepStochastic(state,5,1e2,aGuess);
    vec res2      = ansatz.sweepDeterministic_grad_test();

    data(i,0)     = bs(i);
    data(i,1)     = res2(res2.n_rows-1) - Trap.gsExpectedVal();
  }

  ansatz.printBasis();

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;
  cout << data << endl;

  return 0;
}
