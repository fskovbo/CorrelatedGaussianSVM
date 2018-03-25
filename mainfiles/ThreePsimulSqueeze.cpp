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

  vec masses            = {133.0/6.0 , 133.0/6.0 , 1};
  vec charges           = {0 , 0 , 0};
  auto TwoPart          = System(masses,charges,3);

  auto Gauss            = SingleGaussPotential(TwoPart,-2.7);
  auto Trap             = TrapPotential(TwoPart);
  PotentialList Vstrat  = {&Trap, &Gauss};

  auto elem             = MatrixElements(TwoPart,Vstrat);
  auto ansatz           = Variational(TwoPart,elem);
  ansatz.setUpdateNumber(2);

  size_t state          = 0;
  size_t Nvals          = 30;
  size_t Ntries         = 3;
  size_t K              = 16;
  vec bsxy              = logspace<vec>(-2.0,3.0,Nvals);

  mat data(Nvals,2);
  std::cout << "Calculating state " << state << " using " << K << " basis functions" << '\n';
  std::string name = "ThreePartSqueezeK" + std::to_string(K);

  ansatz.setUniqueCoordinates(3,{0,1,2});
  for (size_t i = 0; i < Nvals; i++) {
    Trap.updateTrap( { bsxy(i) , bsxy(i) , 1e6 } );
    vec aGuess      = {2.0*bsxy(i) , 2.0*bsxy(i) , 5};
    double Ebest    = 1e10;
    cube BestBasis;
    mat BestShift;
    std::string ntmp= name + "bxy" + std::to_string(bsxy(i));

    for (size_t j = 0; j < Ntries; j++) {
      ansatz.initializeBasis(K);
      double Etry;

      try{
        vec warmstart   = ansatz.sweepStochastic(state,5,1e3,aGuess);
        vec res         = ansatz.sweepDeterministic(state,10);
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
    data(i,0)  = bsxy(i);
    data(i,1)  = Ebest;

    try{
      ansatz.setBasis(BestBasis);
      ansatz.setShift(BestShift);
      bool hassaved = ansatz.saveBasis(state,ntmp);
      std::cout << "State saved with energy " << Ebest << '\n';
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
