#ifndef SYSTEM_H
#define SYSTEM_H

#include <armadillo>
#include <iostream>
#include <math.h>
#include <vector>

using namespace arma;
using namespace std;


class System {

public:
  size_t N, n;
  mat U, Ui, lambdamat;
  vec masses, charges;
  vector<vec**> vArrayList;

  System(vec& masses, vec& charges);

};

#endif
