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

  vec masses(2);
  vec charges(2);
  masses << 1 << 1;
  charges << 0 << 0;

  System TwoPart = System(masses,charges,false); //use false for absolute coordinates

  double intStr = -2.684;
  double intRange = 1;
  vec trapOsc = logspace<vec>(4,0,10);
  SingleGaussPotential Gauss(TwoPart,intStr,intRange);
  TrapPotential Trap(TwoPart,trapOsc(0));
  PotentialList Vstrat;
  Vstrat.addPotential(&Gauss);
  Vstrat.addPotential(&Trap);
  MatrixElements elem(TwoPart,Vstrat);
  Variational ansatz = Variational(TwoPart,elem);
  ansatz.initializeBasis(4);

  mat results(trapOsc.n_rows,2);
  vec startingGuess = 2.5*ones<vec>(3);
  ansatz.sweepStochastic(15,1e5,startingGuess);
  for (size_t i = 0; i < trapOsc.n_rows; i++) {
    Trap.updateTrap(trapOsc(i));

    vec res1 = ansatz.sweepStochastic(10,1e4,startingGuess);
    vec res2 = ansatz.sweepDeterministic(10,5*1e3);
    results(i,0) = trapOsc(i);
    results(i,3) = res2(res2.n_rows-1) - 2*0.5* trapOsc(i);
  }

  cout << "Results:" << endl << results << endl;

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;


  return 0;
}
