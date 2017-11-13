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
  auto TwoPart          = System(masses,charges);

  auto Gauss            = SingleGaussPotential(TwoPart);
  auto Trap             = TrapPotential(TwoPart);
  PotentialList Vstrat  = {&Trap};

  auto elem             = MatrixElements(TwoPart,Vstrat);
  auto ansatz           = Variational(TwoPart,elem);


  size_t Nvals          = 6;
  vec bs                = logspace<vec>(-3,3,Nvals);
  mat data              = zeros<mat>(Nvals,4);

  for (size_t i = 0; i < Nvals; i++) {
    Trap.updateTrap(bs(i));
    ansatz.initializeBasis(10);

    vec aGuess    = {2.5 , 2.5 , bs(i)};
    vec res1      = ansatz.sweepStochastic(5,1e4,aGuess);
    vec res2      = ansatz.sweepDeterministic(5,1e4);
    double Vexpt  = 0.5*trace(TwoPart.lambdamat)/3.0/bs(i)/bs(i);
    data(i,0)     = bs(i);
    data(i,1)     = res2(res2.n_rows-1);
    data(i,2)     = Vexpt;
    data(i,3)     = res2(res2.n_rows-1) - Vexpt;
  }

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;
  cout << data << endl;

  return 0;
}
