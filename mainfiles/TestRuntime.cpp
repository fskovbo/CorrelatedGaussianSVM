#include <iostream>
#include <armadillo>
#include <time.h>
#include <vector>

#include "System.h"
#include "SingleGaussPotential.h"
#include "CoulombPotential.h"
#include "MatrixElements.h"
#include "Variational.h"
#include "TrapPotential.h"
#include "PotentialList.h"


int main() {
  clock_t begin = clock();
  arma_rng::set_seed_random();

  size_t De             = 3;
  vec masses            = {1 , 1 , 1};
  vec charges           = {0,0,0};
  auto Test             = System(masses,charges,De);

  auto Trap             = TrapPotential(Test);
  auto Gauss            = SingleGaussPotential(Test);

  PotentialList Vstrat  = {&Gauss, &Trap};
  auto elem             = MatrixElements(Test,Vstrat);
  auto ansatz           = Variational(Test,elem);
  vec trapdepth         = {1e-2 , 1e2, 1e2};

  size_t state          = 0;
  vec aGuess            = {2*1e-2 , 2.5 , 2.5};
  vec shiftBounds       = {1e-2 * 1e-2 , 1e-2 , 1e-2};

  Trap.updateTrap(trapdepth);
  ansatz.initializeBasis(8);
  // ansatz.setUniqueCoordinates(2,{0,1,1});
  ansatz.setUpdateNumber(2);

  vec warmstart         = ansatz.sweepStochastic(state,5,1e3,aGuess);
  vec res               = ansatz.sweepDeterministic(state,5);

  std::cout << "Energy of state " << state << ": " << res(res.n_rows-1)-Trap.gsExpectedVal() << '\n';
  std::cout << "Energy spectrum: \n" << ansatz.eigenSpectrum()-Trap.gsExpectedVal() << '\n';

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;


  return 0;
}
