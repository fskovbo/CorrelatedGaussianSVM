#ifndef SYSTEM_H
#define SYSTEM_H

#include <armadillo>
#include <iostream>
#include <math.h>
#include <vector>

using namespace arma;
using namespace std;


class System {
private:
  void setupCoordinates();
  void setupCoordinates2();
  void setupLambdaMatrix();
  void setupvArray();
  void setupvList();

public:
  size_t N, n, De;
  mat U, Ui, lambdamat;
  vec masses, charges;
  vector<vec**> vArrayList;
  vector<vec> vList;

  System(vec& masses, vec& charges, size_t De = 3);

};

#endif
