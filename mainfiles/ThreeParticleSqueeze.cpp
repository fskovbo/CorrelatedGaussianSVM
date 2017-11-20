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

  vec masses            = {1 , 1 , 1};
  vec charges           = {0 , 0 , 0};
  size_t Deff           = 2;
  auto TwoPart          = System(masses,charges,Deff);

  auto Gauss            = SingleGaussPotential(TwoPart);
  auto Trap             = TrapPotential(TwoPart);
  PotentialList Vstrat  = {&Gauss, &Trap};

  auto elem             = MatrixElements(TwoPart,Vstrat);
  auto ansatz           = Variational(TwoPart,elem);


  size_t Nvals          = 10;
  vec bs                = logspace<vec>(-2,2,Nvals);
  mat data              = zeros<mat>(Nvals,2);

  for (size_t i = 0; i < Nvals; i++) {
    Trap.updateTrap(bs(i));
    ansatz.initializeBasis(10);

    vec aGuess    = {bs(i) , 2.5 , 2.5};
    vec res1      = ansatz.sweepStochastic(5,1e2,aGuess);
    vec res2      = ansatz.sweepDeterministic(5);
    double Vexpt  = 1.5*trace(TwoPart.lambdamat)/Deff/Deff/bs(i)/bs(i);

    data(i,0)     = bs(i);
    data(i,1)     = res2(res2.n_rows-1) - Vexpt;
  }

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;
  cout << data << endl;

  return 0;
}
