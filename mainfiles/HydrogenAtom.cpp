#include <iostream>
#include <armadillo>
#include <time.h>

#include "Utils.h"
#include "System.h"
#include "CoulombPotential.h"
#include "Variational.h"


int main() {
  clock_t begin = clock();
  arma_rng::set_seed_random();

  vec masses    = { 1836 , 1 };
  vec charges   = { 1 , -1};

  auto H        = System(masses,charges,3);
  auto Vstrat   = CoulombPotential(H,10,12);
  auto elem     = MatrixElements(H,Vstrat);
  auto ansatz   = Variational(H,elem);
  size_t state  = 0;

  ansatz.initializeBasis(8);
  vec guess = 4.0/datum::pi * ones<vec>(3);
  vec maxshift = 0.1*ones<vec>(3);
  vec res1 = ansatz.sweepStochasticShift(state,5,1e2,guess,maxshift);
  vec res2 = ansatz.sweepDeterministicShift(state,5);

  cout << "Result after stochastic sweep:" << endl << res1 << endl;
  cout << "Result after deterministic sweep:" << endl << res2 << endl;

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
