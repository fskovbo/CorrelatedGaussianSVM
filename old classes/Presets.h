#ifndef PRESETS_H
#define PRESETS_H

#include <armadillo>
#include <string>
#include <time.h>
#include "JacobiCoordinates.h"
#include "Utils.h"
#include "CoulombPotential.h"
#include "SingleGaussPotential.h"
#include "TrapPotential.h"
#include "DoubleTrapPotential.h"
#include "PotentialList.h"
#include "MatrixElements.h"
#include "Hamilton.h"
#include "MultithreadDriver.h"
#include "Variational.h"

using namespace arma;
using namespace std;

class Presets {
private:
  static mat data;

public:
  //static mat ThreeParticleSqueezeTwoDimSimul(size_t maxeval);
  static mat HeliumNew(size_t basisSize, size_t maxeval);
  };

#endif
