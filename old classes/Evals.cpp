#include "Evals.h"

double Evals::Rosenbrock(vec& x){
  size_t n = x.n_rows;
  vec a = pow(x(span(0,n-2)),2);
  vec b = pow(x(span(1,n-1)),2);
  vec c = pow(x(span(0,n-2))-1,2);

  return 100*sum(a-b) + sum(c);
}

double Evals::eigenEnergy(mat& H, mat& B){
  size_t K = H.n_rows;

  assert(H.n_cols == K);
  assert(B.n_cols == K);
  assert(B.n_rows == K);

  mat L(K,K);
  bool status = chol(L,B,"lower");

  if (status) {
    try{
      vec eigvals = sort(eig_sym(L.i()*H*(L.t()).i()));
      return eigvals(0);
    }
    catch (const std::exception& e){
      return 9999;
    }

  }
  else{
    return 9999;
  }
}

vec Evals::eigenSpectrum(mat& H, mat& B){
  size_t K = H.n_rows;

  assert(H.n_cols == K);
  assert(B.n_cols == K);
  assert(B.n_rows == K);

  mat L(K,K);
  bool status = chol(L,B,"lower");

  if (status) {
    return sort(eig_sym(L.i()*H*(L.t()).i()));
  }
  else{
    return 99999*ones<vec>(K);
  }
}
