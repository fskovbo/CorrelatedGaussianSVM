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
  vec bsx               = logspace<vec>(-2.0,3.0,Nvals);
  vec bsy               = logspace<vec>(-2.0,3.0,Nvals);


  mat datax(Nvals,2);
  mat datay(Nvals,2);
  std::cout << "Calculating state " << state << " using " << K << " basis functions" << '\n';
  std::string name = "ThreePartSqueezeK" + std::to_string(K);

  // ansatz.setUniqueCoordinates(2,{0,1,1});
  ansatz.setUniqueCoordinates(3,{0,1,2});
  for (size_t i = 0; i < Nvals; i++) {
    Trap.updateTrap( { bsx(i) , 1e6 , 1e6 } );
    vec aGuess      = {2.0*bsx(i) , 5 , 5};
    // vec shiftBounds = {1e-2 * bsx(i) , 1e-2 , 1e-2};
    double Ebest    = 1e10;
    cube BestBasis;
    mat BestShift;
    std::string ntmp= name + "bx" + std::to_string(bsx(i));

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
    datax(i,0)  = bsx(i);
    datax(i,1)  = Ebest;

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

  ansatz.setUniqueCoordinates(3,{0,1,2});
  for (size_t i = 0; i < Nvals; i++) {
    Trap.updateTrap( { bsx(0) , bsy(i) , 1e6 } );
    vec aGuess      = {2.0*bsx(0) , 2.0*bsy(i) , 5};
    // vec shiftBounds = {1e-2 * bsx(0) , 1e-2 * bsy(i) , 1e-2};
    double Ebest    = 1e10;
    cube BestBasis;
    mat BestShift;
    std::string ntmp= name + "by" + std::to_string(bsy(i));

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
    datay(i,0)  = bsy(i);
    datay(i,1)  = Ebest;

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

  // mat data = join_vert(datax,datay);

  std::string namex = name + "X";
  std::string namey = name + "Y";
  datax.save(namex,raw_ascii);
  datay.save(namey,raw_ascii);
  // data.save(name,raw_ascii);

  clock_t end = clock();
  cout << "Runtime = " <<  double(end - begin) / CLOCKS_PER_SEC << endl;

  return 0;
}
