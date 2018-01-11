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
    vector<vec>& vList;
    mat H, B;
    cube& basis;
    MatrixElements& matElem;
    vector<double>& runtimedata;
} my_function_data;

typedef struct {
    size_t index, n, K, De, Nunique, state;
    vec& uniquePar;
    vector<vec**>& vArrayList;
    vector<vec>& vList;
    mat H, B;
    cube& basis;
    mat& shift;
    MatrixElements& matElem;
} my_function_data_shift;

typedef struct {
    size_t n, K, De, Npar, state;
    vector<vec>& vList;
    MatrixElements& matElem;
    vector<double>& runtimedata;
} global_data;

inline double myvfunc_grad(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_function_data *d = reinterpret_cast<my_function_data*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De, state = d->state;
  vector<vec>& vList = d->vList;
  mat H = d->H, B = d->B;
  cube& basis = d->basis;
  MatrixElements& matElem = d->matElem;
  vector<double>& runtimedata = d->runtimedata;


  double Hij, Bij;
  vec Hgrad, Bgrad;
  mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
  size_t count = 0, Npar = De*n*(n+1)/2;
  std::vector<vec> HG(Npar,zeros<vec>(K));
  std::vector<vec> BG(Npar,zeros<vec>(K));

  for (auto& w : vList){
    Atrial += x[count] * w*w.t();
    count++;
  }

  for (size_t j = 0; j < K; j++) {
    if (j == index) {
      matElem.calculateH_noShift(Atrial,Atrial,Hij,Bij,Hgrad,Bgrad);
      H(index,index) = Hij;
      B(index,index) = Bij;
    } else {
      Acurrent = basis.slice(j);

      matElem.calculateH_noShift(Acurrent,Atrial,Hij,Bij,Hgrad,Bgrad);
      H(j,index) = Hij;
      H(index,j) = Hij;
      B(j,index) = Bij;
      B(index,j) = Bij;
    }
    for (size_t l = 0; l < Npar; l++) {
      (HG[l])(j) = Hgrad(l);
      (BG[l])(j) = Bgrad(l);
    }
  }

  mat L(K,K);
  vec eigval;
  mat eigvec;
  bool status = chol(L,B,"lower");
  if (status) {
    eig_sym(eigval,eigvec, L.i()*H*(L.t()).i() );
  }
  else{
    eigval = 9999*1e10*ones<vec>(K);
  }

  vec c = (L.t()).i() * eigvec.col(state);
  double E = eigval(state);
  double norm = dot(c , B*c);

  if (!grad.empty()){
    for (size_t i = 0; i < Npar; i++) {
      grad[i] = 2.0*c(index)*dot(c,(HG[i]-E*BG[i]))/norm;
    }
  }

  if (E < runtimedata.back()) {
    runtimedata.push_back(E);
  }
  else{
    runtimedata.push_back(runtimedata.back());
  }

  return E;
}

inline double gradvfunc_shift(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_function_data_shift *d = reinterpret_cast<my_function_data_shift*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De, state = d->state;
  vector<vec>& vList = d->vList;
  mat H = d->H, B = d->B;
  cube& basis = d->basis;
  mat& shift = d->shift;
  MatrixElements& matElem = d->matElem;

  size_t NparA = De*n*(n+1)/2, NparS = 3*n, Npar = NparA+NparS;
  double Hij, Bij;
  vec Hgrad, Bgrad;
  mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
  vec scurrent(3*n), strial(3*n);
  size_t count = 0;
  std::vector<vec> HG(Npar,zeros<vec>(K));
  std::vector<vec> BG(Npar,zeros<vec>(K));

  for (auto& w : vList){
    Atrial += x[count] * w*w.t();
    count++;
  }
  for (size_t i = 0; i < NparS; i++) {
    strial(i) = x[count];
    count++;
  }

  for (size_t j = 0; j < K; j++) {
    if (j == index) {
      matElem.calculateH(Atrial,Atrial,strial,strial,Hij,Bij,Hgrad,Bgrad);
      H(index,index) = Hij;
      B(index,index) = Bij;
    } else {
      Acurrent = basis.slice(j);
      scurrent = shift.col(j);

      matElem.calculateH(Acurrent,Atrial,scurrent,strial,Hij,Bij,Hgrad,Bgrad);
      H(j,index) = Hij;
      H(index,j) = Hij;
      B(j,index) = Bij;
      B(index,j) = Bij;
    }
    for (size_t l = 0; l < Npar; l++) {
      (HG[l])(j) = Hgrad(l);
      (BG[l])(j) = Bgrad(l);
    }
  }

  mat L(K,K);
  vec eigval;
  mat eigvec;
  double E;
  bool status = chol(L,B,"lower");
  if (status) {
    status = eig_sym(eigval,eigvec, L.i()*H*(L.t()).i() );
    if (status) {

        vec c = (L.t()).i() * eigvec.col(state);
        E = eigval(state);
        double norm = dot(c , B*c);

        if (!grad.empty()){
          for (size_t i = 0; i < Npar; i++) {
            grad[i] = 2.0*c(index)*dot(c,(HG[i]-E*BG[i]))/norm;
          }
        }

    }
    else{
      E = 1e10;
      // cout << H << endl;
      // cout << Acurrent << endl;
      // cout << scurrent << endl;
    }
  }
  else{
    E = 1e10;
    // cout << H << endl;
    // cout << Acurrent << endl;
    // cout << scurrent << endl;
  }

  return E;
}

inline double globalvfunc(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  global_data *d = reinterpret_cast<global_data*>(data);
  size_t n = d->n, K = d->K, De = d->De, Npar = d->Npar, state = d->state;
  vector<vec>& vList = d->vList;
  MatrixElements& matElem = d->matElem;
  vector<double>& runtimedata = d->runtimedata;


  double Hij, Bij;
  mat Ai(De*n,De*n), Aj(De*n,De*n), H(K,K), B(K,K);
  cube basis(De*n,De*n,K);

  for (size_t l = 0; l < K; l++) {
    Ai.zeros();
    size_t count = 0;

    for (auto& w : vList){
      Ai += x[l*De*n*(n+1)/2 + count] * w*w.t();
      count++;
    }
    basis.slice(l) = Ai;
  }

  for (size_t i = 0; i < K; i++) {
    Ai = basis.slice(i);
    for (size_t j = i; j < K; j++) {
      Aj = basis.slice(j);
      matElem.calculateH_noShift(Ai,Aj,Hij,Bij);
      H(j,i) = Hij;
      H(i,j) = Hij;
      B(j,i) = Bij;
      B(i,j) = Bij;
    }
  }


  mat L(K,K);
  bool status = chol(L,B,"lower");
  if (status) {
    vec eigs = eig_sym( L.i()*H*(L.t()).i() );

    if (eigs(state) < runtimedata.back()) {
      runtimedata.push_back(eigs(state));
    }
    else{
      runtimedata.push_back(runtimedata.back());
    }

    return eigs(state);
  }
  else{
    return 9999*1e10;
  }
}

typedef struct {
    size_t index, n, K, De, Nunique, state;
    vec& uniquePar;
    vector<vec>& vList;
    mat H, B;
    cube& basis;
    mat& shift;
    MatrixElements& matElem;
    bool shifted;
} function_data;

inline double fitness(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  function_data *d = reinterpret_cast<function_data*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De, Nunique = d->Nunique, state = d->state;
  vec& uniquePar = d->uniquePar;
  vector<vec>& vList = d->vList;
  mat H = d->H, B = d->B;
  cube& basis = d->basis;
  mat& shift = d->shift;
  MatrixElements& matElem = d->matElem;
  bool shifted = d->shifted;

  double Hij, Bij;
  mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
  vec scurrent(3*n), strial(3*n);
  size_t count = 0;

  size_t c1 = 0, c2 = 0;
  for (vec& w : vList) {
    Atrial += x[Nunique*c1+uniquePar(c2++)]*w*w.t();
    if (c2 == De) { c2 = 0; c1 ++; }
  }

  for (size_t i = 0; i < 3*n; i++) {
    strial(i) = x[i+Nunique*n*(n+1)/2];
  }

  for (size_t j = 0; j < K; j++) {
    if (j == index) {
      if (shifted) { matElem.calculateH(Atrial,Atrial,strial,strial,Hij,Bij); }
      else         { matElem.calculateH_noShift(Atrial,Atrial,Hij,Bij); }
      H(index,index) = Hij;
      B(index,index) = Bij;
    } else {
      Acurrent = basis.slice(j);
      scurrent = shift.col(j);

      if (shifted) { matElem.calculateH(Acurrent,Atrial,scurrent,strial,Hij,Bij); }
      else         { matElem.calculateH_noShift(Acurrent,Atrial,Hij,Bij); }
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

typedef struct {
    size_t index, n, K, De, Nunique, state, Nf;
    vec& uniquePar;
    vector<vec>& vList;
    mat H, B;
    cube& basis;
    mat& shift;
    MatrixElements& matElem;
    bool shifted;
} function_data_test;

inline double fitness_test(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  function_data_test *d = reinterpret_cast<function_data_test*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De, Nunique = d->Nunique, state = d->state, Nf = d->Nf;
  vec& uniquePar = d->uniquePar;
  vector<vec>& vList = d->vList;
  mat H = d->H, B = d->B;
  cube& basis = d->basis;
  mat& shift = d->shift;
  MatrixElements& matElem = d->matElem;
  bool shifted = d->shifted;

  double Hij, Bij;
  mat Ai(De*n,De*n), Aj(De*n,De*n);
  vec si(3*n), sj(3*n);
  cube Atrials(De*n,De*n,Nf);
  mat strials(3*n,Nf);
  size_t count = 0, Npar = Nunique*n*(n+1)/2 + 3*n;

  for (size_t j = 0; j < Nf; j++) {
    mat Atrial = zeros<mat>(De*n,De*n);
    vec strial = zeros<vec>(3*n);
    size_t c1 = 0, c2 = 0;
    for (vec& w : vList) {
      Atrial += x[Npar*j+Nunique*c1+uniquePar(c2++)]*w*w.t();
      if (c2 == De) { c2 = 0; c1 ++; }
    }

    for (size_t i = 0; i < 3*n; i++) {
      strial(i) = x[Npar*j+i+Nunique*n*(n+1)/2];
    }
    Atrials.slice(j) = Atrial;
    strials.col(j)   = strial;
  }

  basis.slices(index,index+Nf-1) = Atrials;
  shift.cols(index,index+Nf-1)   = strials;

  for (size_t i = 0; i < Nf; i++) {
    Ai = basis.slice(index+i);
    si = shift.col(index+i);

    for (size_t j = 0; j < K; j++) {
      Aj = basis.slice(j);
      sj = shift.col(j);

      if (shifted) { matElem.calculateH(Ai,Aj,si,sj,Hij,Bij); }
      else         { matElem.calculateH_noShift(Ai,Aj,Hij,Bij); }
      H(j,index+i) = Hij;
      H(index+i,j) = Hij;
      B(j,index+i) = Bij;
      B(index+i,j) = Bij;
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

#endif
