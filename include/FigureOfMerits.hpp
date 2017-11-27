#ifndef FIGUREOFMERIT_HPP
#define  FIGUREOFMERIT_HPP

#include "math.h"
#include <vector>

#include "MatrixElements.h"

using namespace arma;
using namespace std;


typedef struct {
    size_t index, n, K, De, Nunique, state;
    vec& uniquePar;
    vector<vec**>& vArrayList;
    mat H, B;
    vector<mat> HG, BG;
    cube& basis;
    MatrixElements& matElem;
} my_function_data;

typedef struct {
    size_t index, n, K, De, Nunique, state;
    vec& uniquePar;
    vector<vec**>& vArrayList;
    mat H, B;
    cube& basis;
    mat& shift;
    MatrixElements& matElem;
} my_function_data_shift;

inline double myvfunc(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_function_data *d = reinterpret_cast<my_function_data*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De, Nunique = d->Nunique, state = d->state;
  vec& uniquePar = d->uniquePar;
  vector<vec**>& vArrayList = d->vArrayList;
  mat H = d->H, B = d->B;
  cube& basis = d->basis;
  MatrixElements& matElem = d->matElem;

  double Hij, Bij;
  mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
  size_t count = 0;
  vec** vArray;

  for (size_t i = 0; i < n+1; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        Atrial += x[Nunique*count+uniquePar(k)] * (vArray[i][j] * (vArray[i][j]).t());
      }
      count++;
    }
  }

  for (size_t j = 0; j < K; j++) {
    if (j == index) {
      matElem.calculateH_noShift(Atrial,Atrial,Hij,Bij);
      H(index,index) = Hij;
      B(index,index) = Bij;
    } else {
      Acurrent = basis.slice(j);

      matElem.calculateH_noShift(Acurrent,Atrial,Hij,Bij);
      H(j,index) = Hij;
      H(index,j) = Hij;
      B(j,index) = Bij;
      B(index,j) = Bij;
    }
  }

  mat L(K,K);
  bool status = chol(L,B,"lower");
  if (status) {
    vec eigs = eig_sym( L.i()*H*(L.t()).i() );
    return eigs(state);
  }
  else{
    return 9999*1e10;
  }
}


inline double myvfunc_shift(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_function_data_shift *d = reinterpret_cast<my_function_data_shift*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De, Nunique = d->Nunique, state = d->state;
  vec& uniquePar = d->uniquePar;
  vector<vec**>& vArrayList = d->vArrayList;
  mat H = d->H, B = d->B;
  cube& basis = d->basis;
  mat& shift = d->shift;
  MatrixElements& matElem = d->matElem;

  double Hij, Bij;
  mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
  vec scurrent(3*n), strial(3*n);
  size_t count = 0;
  vec** vArray;

  for (size_t i = 0; i < n+1; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        Atrial += x[Nunique*count+uniquePar(k)] * (vArray[i][j] * (vArray[i][j]).t());
      }
      count++;
    }
  }

  for (size_t i = 0; i < 3*n; i++) {
    strial(i) = x[i+count*Nunique];
  }

  for (size_t j = 0; j < K; j++) {
    if (j == index) {
      matElem.calculateH(Atrial,Atrial,strial,strial,Hij,Bij);
      H(index,index) = Hij;
      B(index,index) = Bij;
    } else {
      Acurrent = basis.slice(j);
      scurrent = shift.col(j);

      matElem.calculateH(Acurrent,Atrial,scurrent,strial,Hij,Bij);
      H(j,index) = Hij;
      H(index,j) = Hij;
      B(j,index) = Bij;
      B(index,j) = Bij;
    }
  }

  mat L(K,K);
  bool status = chol(L,B,"lower");
  if (status) {
    vec eigs = eig_sym( L.i()*H*(L.t()).i() );
    return eigs(state);
  }
  else{
    return 9999*1e10;
  }
}

inline double myvfunc_shift_add(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_function_data_shift *d = reinterpret_cast<my_function_data_shift*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De, Nunique = d->Nunique, state = d->state;
  vec& uniquePar = d->uniquePar;
  vector<vec**>& vArrayList = d->vArrayList;
  mat H = d->H, B = d->B;
  cube& basis = d->basis;
  mat& shift = d->shift;
  MatrixElements& matElem = d->matElem;

  double Hij, Bij;
  mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
  vec scurrent(3*n), strial(3*n);
  size_t count = 0;
  vec** vArray;

  for (size_t i = 0; i < n+1; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        Atrial += x[Nunique*count+uniquePar(k)] * (vArray[i][j] * (vArray[i][j]).t());
      }
      count++;
    }
  }

  for (size_t i = 0; i < 3*n; i++) {
    strial(i) = x[i+count*Nunique];
  }

  for (size_t j = 0; j < K-1; j++) {
    Acurrent = basis.slice(j);
    scurrent = shift.col(j);

    matElem.calculateH(Acurrent,Atrial,scurrent,strial,Hij,Bij);
    H(j,K-1) = Hij;
    H(K-1,j) = Hij;
    B(j,K-1) = Bij;
    B(K-1,j) = Bij;
  }
  matElem.calculateH(Atrial,Atrial,strial,strial,Hij,Bij);
  H(K-1,K-1) = Hij;
  B(K-1,K-1) = Bij;

  mat L(K,K);
  bool status = chol(L,B,"lower");
  if (status) {
    vec eigs = eig_sym( L.i()*H*(L.t()).i() );
    return eigs(state);
  }
  else{
    return 9999*1e10;
  }
}

#endif