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
  ansatz.setUniqueCoordinates(2,{0,1,1});
  ansatz.setUpdateNumber(2);

  size_t state          = 2;
  size_t Nvals          = 20;
  size_t Ntries         = 10;
  size_t K              = 14;
  vec bs                = logspace<vec>(-2.5,2.5,Nvals);


  mat data(Nvals,2);
  std::string name = "ThreePartSqueezeK" + std::to_string(K);

  for (size_t i = 0; i < Nvals; i++) {
    Trap.updateTrap(bs(i));
    vec aGuess      = {2 * bs(i) , 2.5 , 2.5};
    vec shiftBounds = {1e-2 * bs(i) , 1e-2 , 1e-2};
    double Ebest    = 1e10;
    cube BestBasis;
    mat BestShift;
    std::string ntmp= name + "b" + std::to_string(bs(i));

    for (size_t j = 0; j < Ntries; j++) {
      ansatz.initializeBasis(K);
      double Etry;

      try{
        vec warmstart   = ansatz.sweepStochastic(state,10,1e4,aGuess);
        vec res         = ansatz.sweepDeterministic(state,20);
        Etry            = res(res.n_rows-1)-Trap.gsExpectedVal();
      }
      catch (const std::exception& e) {
        Etry            = 1e11;
      }
      if (Etry < Ebest) {
        Ebest     = Etry;
        BestBasis = ansatz.getBasis();
        BestShift = ansatz.getShift();
      }
      std::cout << "Value: " << i << ", try: " << j << '\n';
    }
    data(i,0)  = bs(i);
    data(i,1)  = Ebest;

    try{
      ansatz.setBasis(BestBasis);
      ansatz.setShift(BestShift);
      bool hassaved = ansatz.saveBasis(state,ntmp);
      std::cout << "Save Status: " << hassaved << '\n';
    }
    catch (const std::exception& e) {
      std::cout << "Error encountered while saving basis!" << '\n';
    }
  }

  data.save(name,raw_ascii);

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
