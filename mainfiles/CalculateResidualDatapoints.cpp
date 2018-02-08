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

  vec masses            = {10 , 10 , 1};
  vec charges           = {0 , 0 , 0};
  auto TwoPart          = System(masses,charges,3);

  auto Gauss            = SingleGaussPotential(TwoPart);
  auto Trap             = TrapPotential(TwoPart);
  PotentialList Vstrat  = {&Trap, &Gauss};

  auto elem             = MatrixElements(TwoPart,Vstrat);
  auto ansatz           = Variational(TwoPart,elem);
  ansatz.setUpdateNumber(2);
  // ansatz.setUniqueCoordinates(2,{0,1,1});

  size_t state          = 2;
  size_t Ntries         = 3;
  size_t K              = 26;

  std::vector<vec> trapDepths;
  trapDepths.emplace_back(vec({1e3 , 1e3, 1e3}));

  mat data(trapDepths.size(),3);

  std::string name = "ThreePartSqueezeK" + std::to_string(K) + "State" + std::to_string(state);
  std::cout << "Calculating state " << state << " using " << K << " basis functions" << '\n';
  size_t count = 0;
  for (vec& b: trapDepths){
    Trap.updateTrap( b );
    // vec aGuess      = {2.0*b(0) , 2.0*b(1) , 2.5};
    vec aGuess      = {5 , 5 , 5};
    double Ebest    = 1e10;
    cube BestBasis;
    mat BestShift;
    std::string ntmp= name + "bx" + std::to_string(b(0)) + "by" + std::to_string(b(1));

    for (size_t j = 0; j < Ntries; j++) {
      ansatz.initializeBasis(K);
      double Etry;

      try{
        vec warmstart   = ansatz.sweepStochastic(state,10,1e3,aGuess);
        vec res         = ansatz.sweepDeterministic(state,25);
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
    }
    data(count,0)  = b(0);
    data(count,1)  = b(1);
    data(count,2)  = Ebest;
    std::cout << "Final Energy: " << Ebest << '\n';
    count++;

    try{
      // ansatz.setBasis(BestBasis);
      // ansatz.setShift(BestShift);
      // bool hassaved = ansatz.saveBasis(state,ntmp);
      // std::cout << "Save Status: " << hassaved << '\n';
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
