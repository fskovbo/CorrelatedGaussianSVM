#include <iostream>
#include <armadillo>
#include <string>

#include "System.h"
#include "Wavefunction.hpp"


int main() {

  vec masses            = {1 , 1 , 1};
  vec charges           = {0 , 0 , 0};
  auto Sys              = System(masses,charges,3);
  std::string filename  = "ThreePartSqueezeK12b0.035697";
  auto psi              = Wavefunction(Sys,filename);

  std::cout << "RMS distance\n" << psi.RMSdistances() << '\n';
  std::cout << "Symmetry\n" << psi.Symmetrization() << '\n';

  return 0;
}
