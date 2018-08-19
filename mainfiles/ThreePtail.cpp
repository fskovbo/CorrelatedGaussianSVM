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

  auto Gauss            = SingleGaussPotential(TwoPart,-2.8);
  auto Trap             = TrapPotential(TwoPart);
  PotentialList Vstrat  = {&Trap, &Gauss};

  auto elem             = MatrixElements(TwoPart,Vstrat);
  auto ansatz           = Variational(TwoPart,elem);
  ansatz.setUpdateNumber(2);

  size_t state          = 1;
  size_t Nvals          = 4;
  size_t Ntries         = 20;
  size_t K              = 20;

  vec bsx               = logspace<vec>(2.482758620689655,3.0,Nvals);

  mat datax(2+Ntries,Nvals);

  std::cout << "Calculating state " << state << " using " << K << " basis functions" << '\n';
  std::string name = "ThreePartSqueezeK" + std::to_string(K);

  ansatz.setUniqueCoordinates(3,{0,1,2});
  for (size_t i = 0; i < Nvals; i++) {
    Trap.updateTrap( { bsx(i) , 1e6 , 1e6 } );
    vec aGuess      = {bsx(i) , 5 , 5};
    double Ebest    = 1e10;
    cube BestBasis;
    mat BestShift;
    std::string ntmp= name + "bx" + std::to_string(bsx(i));

    for (size_t j = 0; j < Ntries; j++) {
      ansatz.initializeBasis(K);
      double Etry;

      try{
        vec warmstart   = ansatz.sweepStochastic(state,10,5e3,aGuess);
        vec res         = ansatz.sweepDeterministic(state,10);
        Etry            = res(res.n_rows-1)-Trap.gsExpectedVal();
      }
      catch (const std::exception& e) {
        Etry            = 1e11;
      }
      if (Etry < Ebest) {
        datax(j+2,i) = Etry;
        Ebest     = Etry;
        BestBasis = ansatz.getBasis();
        BestShift = ansatz.getShift();
      }
      std::cout << "x-value: " << i << ", try: " << j << '\n';
    }
    datax(0,i)  = bsx(i);
    datax(1,i)  = Ebest;

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

  std::string namex = name + "X";
  datax.save(namex,raw_ascii);


  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
