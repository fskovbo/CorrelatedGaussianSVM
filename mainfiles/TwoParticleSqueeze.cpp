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

  vec masses = {1 , 1};
  vec charges = {0 , 0};

  System TwoPart = System(masses,charges);

  auto Gauss = SingleGaussPotential(TwoPart);
  auto Trap = TrapPotential(TwoPart),

  PotentialList Vstrat;
  Vstrat.addPotential(&Gauss);
  Vstrat.addPotential(&Trap);
  auto elem = MatrixElements(TwoPart,Vstrat);
  auto ansatz = Variational(TwoPart,elem);



  size_t Nvals = 10;
  vec bs = logspace<vec>(-2,2,Nvals);
  mat data(Nvals,2);
  double mui = TwoPart.lambdamat(0,0);
  vec trapOsc;
  for (size_t i = 0; i < Nvals; i++) {
    trapOsc = { pow(bs(i),-4) };
    Trap.updateTrap(trapOsc);
    ansatz.initializeBasis(4);
    vec res1 = ansatz.sweepStochastic(5,1e4,2.5*ones<vec>(3));
    vec res2 = ansatz.sweepDeterministic(5,1e4);
    data(i,0) = bs(i);
    data(i,1) = res2(res2.n_rows-1) - 0.5*mui/bs(i)/bs(i);
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
