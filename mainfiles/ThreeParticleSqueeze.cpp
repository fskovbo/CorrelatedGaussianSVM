#include <iostream>
#include <armadillo>
#include <time.h>
#include <vector>
#include <string>

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
  auto TwoPart          = System(masses,charges,3);

  auto Gauss            = SingleGaussPotential(TwoPart);
  auto Trap             = TrapPotential(TwoPart);
  PotentialList Vstrat  = {&Trap, &Gauss};

  auto elem             = MatrixElements(TwoPart,Vstrat);
  auto ansatz           = Variational(TwoPart,elem);

  size_t state          = 1;
  size_t Nvals          = 20;
  size_t Ntries         = 10;
  size_t K              = 7;
  vec bs                = logspace<vec>(-2.5,2.5,Nvals);


  mat data = zeros<mat>(Nvals,Ntries+1);

  for (size_t i = 0; i < Nvals; i++) {
    Trap.updateTrap(bs(i));
    vec aGuess      = {2*1e-2 , 2.5 , 2.5};
    vec maxShift    = {0.1*bs(i) , 1 , 1};
    data(i,0)       = bs(i);

    for (size_t j = 0; j < Ntries; j++) {
      ansatz.initializeBasis(K);

      try{
        vec res1        = ansatz.sweepStochastic(state,5,1e4,aGuess);
        vec res2        = ansatz.sweepDeterministic(state,10);
        vec lastres     = ansatz.fullBasisSearch(state);

        data(i,j+1)     = lastres(0) - Trap.gsExpectedVal();
      }
      catch (const std::exception& e) {
        data(i,j+1)     = 999;
      }
      std::cout << "Value: " << i << ", try: " << j << '\n';
    }
  }
  std::string name = "ThreePartSqueezeK";
  std::string nametmp;
  nametmp = name + std::to_string(K);
  data.save(nametmp,raw_ascii);

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
