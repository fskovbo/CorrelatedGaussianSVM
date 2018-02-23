#include <iostream>
#include <armadillo>
#include <string>

#include "System.h"
#include "Wavefunction.hpp"


int main() {

  vec masses            = {1 , 1 , 1};
  vec charges           = {0 , 0 , 0};
  auto Sys              = System(masses,charges,3);

  size_t Nvals          = 25;
  std::vector<int> Ks   = {14,20,26,30};
  std::vector<int> sts  = {0,1,2,3};
  vec bsx               = logspace<vec>(-2.0,2.0,Nvals);
  vec bsy               = logspace<vec>(-2.0,2.0,Nvals);

  mat symmetries(2*Nvals,Ks.size());

  for (size_t j = 0; j < Ks.size(); j++) {
    size_t count = 0;
    auto Kstr = std::to_string(Ks.at(j));
    std::string dirname = "K" + Kstr + "State" + std::to_string(sts.at(j)) + "Double/";
    mat temp(3*masses.n_rows,Nvals);

    for (size_t i = 0; i < Nvals; i++) {
      double b              = bsx(i);
      std::string filename  = dirname + "ThreePartSqueezeK" + Kstr + "bx" + std::to_string(b);
      try {
        auto psi            = Wavefunction(Sys,filename);
        symmetries(i,j)     = psi.Symmetrization();
      }
      catch (const std::exception& e) {
        symmetries(i,j)     = 0;
      }
    }

    for (size_t i = 0; i < Nvals; i++) {
      double b              = bsy(i);
      std::string filename  = dirname + "ThreePartSqueezeK" + Kstr + "by" + std::to_string(b);
      try {
        auto psi              = Wavefunction(Sys,filename);
        symmetries(i+Nvals,j) = psi.Symmetrization();
      }
      catch (const std::exception& e) {
        symmetries(i+Nvals,j) = 0;
      }
    }
  }

  symmetries.save("Symmetries",raw_ascii);


  return 0;
}
