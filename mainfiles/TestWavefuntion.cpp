#include <iostream>
#include <armadillo>
#include <string>

#include "System.h"
#include "SingleGaussPotential.h"
#include "MatrixElements.h"
#include "Variational.h"
#include "TrapPotential.h"
#include "PotentialList.h"
#include "Wavefunction.hpp"


int main() {

  vec masses            = {1 , 1 , 1};
  vec charges           = {0 , 0 , 0};
  auto Sys              = System(masses,charges,3);
  auto Gauss            = SingleGaussPotential(Sys);
  auto Trap             = TrapPotential(Sys,0.035697);
  PotentialList Vstrat  = {&Trap, &Gauss};
  auto elem             = MatrixElements(Sys,Vstrat);
  std::string filename  = "ThreePartSqueezeK14b0.035697";
  auto psi              = Wavefunction(Sys,elem,filename);

  std::cout << "RMS distance\n" << psi.RMSdistances() << '\n';
  std::cout << "Symmetry\n" << psi.Symmetrization() << '\n';

  return 0;
}
