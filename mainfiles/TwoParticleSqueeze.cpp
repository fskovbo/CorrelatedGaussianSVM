#include <iostream>
#include <armadillo>
#include <time.h>
#include <vector>

#include "Utils.h"
#include "System.h"
#include "SingleGaussPotential.h"
#include "MatrixElements.h"
#include "Variational.h"
#include "TrapPotential.h"
#include "PotentialList.h"


int main() {
  clock_t begin = clock();

  arma_rng::set_seed_random();

  vec masses = {10 , 10};
  vec charges = {0 , 0};

  System TwoPart = System(masses,charges);

  vec temptrapOsc = 1e4 * ones<vec>(masses.n_rows);
  SingleGaussPotential Gauss(TwoPart);
  TrapPotential Trap(TwoPart,temptrapOsc);
  PotentialList Vstrat;
  Vstrat.addPotential(&Gauss);
  Vstrat.addPotential(&Trap);
  MatrixElements elem(TwoPart,Vstrat);
  Variational ansatz = Variational(TwoPart,elem);

  ansatz.initializeBasis(4);

  vec startingGuess = 2.5*ones<vec>(3);

  size_t Nvals = 10;
  vec oscs = logspace<vec>(4,-4,Nvals);
  mat data(Nvals,2);
  vec trapOsc;
  for (size_t i = 0; i < Nvals; i++) {
    trapOsc = oscs(i) * ones<vec>(masses.n_rows);
    Trap.updateTrap(trapOsc);
    ansatz.initializeBasis(4);
    vec res1 = ansatz.sweepStochastic(5,1e4,startingGuess);
    vec res2 = ansatz.sweepDeterministic(5,1e4);
    data(i,0) = oscs(i);
    data(i,1) = res2(res2.n_rows-1) - 0.5*0.5*sum(trapOsc);
  }

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;
  cout << data << endl;



  // vec res1 = ansatz.sweepStochastic(5,1e4,startingGuess);
  // vec res2 = ansatz.sweepDeterministic(5,1e4);
  // clock_t end = clock();
  // cout << "Results after stochastic sweep" << endl << res1 - 0.5*sum(trapOsc) << endl;
  // cout << "Results after deterministic sweep" << endl << res2 - 0.5*sum(trapOsc) << endl;
  // cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;



  return 0;
}
