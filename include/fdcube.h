#ifndef FDCUBE_H
#define FDCUBE_H

#include <armadillo>
#include <iostream>
#include <vector>
#include "assert.h"

using namespace arma;
using namespace std;

class fdcube{
private:
  vector<cube> data;
  size_t rows;
  size_t cols;
  size_t pieces;
  size_t slices;

public:
  fdcube(size_t i, size_t j, size_t k, size_t l);
  fdcube();

  cube getPiece(size_t i);
  void setPiece(size_t i, cube& piece);
  size_t n_pieces();
  size_t n_rows();
  size_t n_cols();
  size_t n_slices();
  void printall();
};

#endif
