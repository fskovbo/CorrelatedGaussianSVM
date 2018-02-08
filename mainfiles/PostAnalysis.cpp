#include <iostream>
#include <armadillo>
#include <string>

#include "System.h"
#include "Wavefunction.hpp"


int main() {

  vec masses            = {1 , 1 , 1};
  vec charges           = {0 , 0 , 0};
  auto Sys              = System(masses,charges,3);

  int state             = 1;
  size_t Nvals          = 20;
  std::vector<int> Ks   = {12,14,16,18,22,26};
  vec bs                = logspace<vec>(-2.5,2.5,Nvals);

  mat symmetries(Nvals,Ks.size());
  cube RMSdist(3*masses.n_rows,Nvals,Ks.size());
  mat energies(Nvals,Ks.size()+1);
  energies.col(0) = bs;
  size_t count = 0;

  for (int K : Ks){
    std::string dirname = "K" + std::to_string(K) + "State" + std::to_string(state) + "/";
    mat temp(3*masses.n_rows,Nvals);

    for (size_t i = 0; i < Nvals; i++) {
      double b              = bs(i);
      std::string filename  = dirname + "ThreePartSqueezeK" + std::to_string(K) + "b" + std::to_string(b);
      try {
        auto psi              = Wavefunction(Sys,filename);
        symmetries(i,count)   = psi.Symmetrization();
        temp.col(i)           = psi.RMSdistances();
      }
      catch (const std::exception& e) {
        symmetries(i,count)   = 0;
        temp.col(i).zeros();
      }
    }

    std::string filename = dirname + "ThreePartSqueezeK" + std::to_string(K);
    mat etmp;
    etmp.load(filename,raw_ascii);
    energies.col(count+1)= etmp.col(1);
    RMSdist.slice(count) = temp;
    count++;
  }

  symmetries.save("Symmetries",raw_ascii);
  RMSdist.save("RMSdist",raw_ascii);
  energies.save("Energies",raw_ascii);


  return 0;
}
