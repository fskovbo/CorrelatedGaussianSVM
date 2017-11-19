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

  vec masses(3);
  vec charges(3);
  masses << 7296 << 1 << 1;
  charges << 2 << -1 << -1;

  System He = System(masses,charges);
  CoulombPotential Vstrat(He,10,12);
  MatrixElements elem(He,Vstrat);
  Variational ansatz1 = Variational(He,elem);

  ansatz1.initializeBasis(20);
  vec startingGuess = 2.5*ones<vec>(3);
  vec res1 = ansatz1.sweepStochastic(5,1e2,startingGuess);
  vec res2 = ansatz1.sweepDeterministic(5);

  cout << "Result after stochastic sweep:" << endl << res1 << endl;
  cout << "Result after deterministic sweep:" << endl << res2 << endl;

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
