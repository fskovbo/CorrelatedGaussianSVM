#include "PotentialList.h"

PotentialList::PotentialList(){

}

void PotentialList::addPotential(PotentialStrategy* pot){
  list.push_back(pot);
}

double PotentialList::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  double Vtotal = 0;
  std::vector<PotentialStrategy*>::iterator it = list.begin();

  while (it != list.end()) {
    Vtotal += (*it++)->calculateExpectedPotential(A1,A2,s1,s2,Binv,detB);
  }

  return Vtotal;
}

double PotentialList::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  double Vtotal = 0;
  std::vector<PotentialStrategy*>::iterator it = list.begin();

  while (it != list.end()) {
    Vtotal += (*it++)->calculateExpectedPotential_noShift(A1,A2,Binv,detB);
  }

  return Vtotal;
}
