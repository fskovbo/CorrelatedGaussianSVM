#include "fdcube.h"

fdcube::fdcube(size_t i, size_t j, size_t k, size_t l){
  slices = k;
  cols = j;
  rows = i;
  pieces = l;

  for (size_t p = 0; p < l; p++) {
    data.push_back(zeros<cube>(i,j,k));
  }
}

fdcube::fdcube(){ }

cube fdcube::getPiece(size_t i){
  assert(i < pieces && i >= 0);
  return data[i];
}

void fdcube::setPiece(size_t i, cube& piece){
  assert(piece.n_slices == slices);
  assert(piece.n_rows == rows);
  assert(piece.n_cols == cols);

  data[i] = piece;
}

size_t fdcube::n_pieces(){
  return pieces;
}

size_t fdcube::n_slices(){
  return slices;
}

size_t fdcube::n_rows(){
  return rows;
}

size_t fdcube::n_cols(){
  return cols;
}

void fdcube::printall(){
  for (size_t i = 0; i < pieces; i++) {
    cout << "[fdcube piece " << i << "]" << endl << data[i] << endl;
  }
}
