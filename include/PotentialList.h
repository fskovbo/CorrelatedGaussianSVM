#ifndef POTENTIALLIST_H
#define POTENTIALLIST_H

#include <armadillo>
#include <vector>
#include "PotentialStrategy.h"

using namespace arma;
using namespace std;

class PotentialList : public PotentialStrategy{
private:
  std::vector<PotentialStrategy*> list;
public:
  PotentialList();
  PotentialList(const std::initializer_list<PotentialStrategy*>& initlist);
  void addPotential(PotentialStrategy* pot);
  virtual double calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB);
  virtual double calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB);
};

#endif
