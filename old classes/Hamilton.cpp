#include "Hamilton.h"
#include <time.h>

Hamilton::Hamilton(size_t K, size_t n, MatrixElements matElem)
: K(K), n(n), matElem(matElem){
  H = zeros<mat>(K,K);
  B = zeros<mat>(K,K);
}

mat Hamilton::getH(){
  return H;
}

mat Hamilton::getB(){
  return B;
}

void Hamilton::buildMatrices(vec& x){
  assert(x.n_rows == 6*n*K);

  double Hij, Bij;
  mat A1 = zeros<mat>(3*n,3*n), A2 = zeros<mat>(3*n,3*n);
  vec s1 = zeros<vec>(3*n), s2 = zeros<vec>(3*n);

  for (size_t i = 0; i < K; i++) {
    A1.diag() = x.rows(i*n*3,i*n*3+(3*n-1));
    s1 = x.rows(i*n*3+3*n*K,i*n*3+(3*n-1)+3*n*K);

    for (size_t j = i; j < K; j++) {
      A2.diag() = x.rows(j*n*3,j*n*3+(3*n-1));
      s2 = x.rows(j*n*3+3*n*K,j*n*3+(3*n-1)+3*n*K);

      matElem.calculateH(A1,A2,s1,s2,Hij,Bij);
      H(i,j) = Hij;
      H(j,i) = Hij;
      B(i,j) = Bij;
      B(j,i) = Bij;
    }
  }
}

vec Hamilton::optimizeH(vec& xmean, size_t lambda, size_t mu, double sigma, size_t maxeval){
  function<double(vec&)> eval = [&](vec& x){
    assert(x.n_rows == 6*n*K);
    //if (any(x.rows(0,3*n*K-1) < 0)) return 100.0;
    x.rows(0,3*n*K-1) = abs(x.rows(0,3*n*K-1));
    buildMatrices(x);
    return Evals::eigenEnergy(H,B);
  };

  CMAES::optimize(eval, xmean, lambda, mu, sigma, maxeval);
  Multidim_min::QuasiNewtonMin(eval, xmean, 1e-4, 1e-4, maxeval);


  return Evals::eigenSpectrum(H,B);
}

void Hamilton::buildstuff(vec& x, mat& Hx, mat& Bx){
  size_t Kx = Hx.n_rows;

  double Hij, Bij;
  mat A1 = zeros<mat>(3*n,3*n), A2 = zeros<mat>(3*n,3*n);
  vec s1 = zeros<vec>(3*n), s2 = zeros<vec>(3*n);

  for (size_t i = 0; i < Kx; i++) {
    A1.diag() = x.rows(i*n*3,i*n*3+(3*n-1));
    s1 = x.rows(i*n*3+3*n*Kx,i*n*3+(3*n-1)+3*n*Kx);

    for (size_t j = i; j < Kx; j++) {
      A2.diag() = x.rows(j*n*3,j*n*3+(3*n-1));
      s2 = x.rows(j*n*3+3*n*Kx,j*n*3+(3*n-1)+3*n*Kx);

      matElem.calculateH(A1,A2,s1,s2,Hij,Bij);
      Hx(i,j) = Hij;
      Hx(j,i) = Hij;
      Bx(i,j) = Bij;
      Bx(j,i) = Bij;
    }
  }
}

void Hamilton::expBasis(vec& x, mat& H, mat& B, vec& Amean, double Swidth, size_t state, size_t lambda){
  size_t K = H.n_rows;

  H.resize(K+1,K+1);
  B.resize(K+1,K+1);

  mat Am = repmat(Amean,1,lambda);

  mat Atrial = 1.0/pow(-Am%log(randu<mat>(3*n,lambda)),2);
  mat Strial = Swidth*randn<mat>(3*n,lambda);
  vec Etrial(lambda);
  cube Htrial(K+1,K+1,lambda), Btrial(K+1,K+1,lambda);

  double Hij, Bij;
  mat A1 = zeros<mat>(3*n,3*n), A2 = zeros<mat>(3*n,3*n);
  vec s1 = zeros<vec>(3*n), s2 = zeros<vec>(3*n);

  for (size_t k = 0; k < lambda; k++) {
    A2.diag() = Atrial.col(k);
    s2 = Strial.col(k);

    for (size_t i = 0; i < K; i++) {
      A1.diag() = x.rows(i*n*3,i*n*3+(3*n-1));
      s1 = x.rows(i*n*3+3*n*K,i*n*3+(3*n-1)+3*n*K);

      matElem.calculateH(A1,A2,s1,s2,Hij,Bij);
      H(i,K) = Hij;
      H(K,i) = Hij;
      B(i,K) = Bij;
      B(K,i) = Bij;
    }
    matElem.calculateH(A2,A2,s2,s2,Hij,Bij);
    H(K,K) = Hij;
    B(K,K) = Bij;
    Htrial.slice(k) = H;
    Btrial.slice(k) = B;

    vec eigs = Evals::eigenSpectrum(H,B);
    Etrial(k) = eigs(state);
  }
  uword index = Etrial.index_min();

  H = Htrial.slice(index);
  B = Btrial.slice(index);
  vec Ax, Sx;

  if (K == 0) {
    Ax = Atrial.col(index);
    Sx = Strial.col(index);
  }
  else{
    Ax = join_vert(x.rows(0,3*n*K-1),Atrial.col(index));
    Sx = join_vert(x.rows(3*n*K,6*n*K-1),Strial.col(index));
  }

  x = join_vert(Ax,Sx);
}

vec Hamilton::simpleOptimize(mat& Amean, vec& Swidth, vec& Kvec, size_t lambda){
  size_t states = Kvec.n_rows;
  mat H, B;
  vec x;


  double Sw = Swidth(0);
  vec Am = Amean.col(0);

  clock_t tStart = clock();
  for (size_t i = 0; i < Kvec(0); i++) {
    //expBasis(x,H,B,Am,Sw,0,lambda);
    expBasis(x,H,B,Am,Sw,0,lambda/(i+1));
  }


  printf("expBasis done in: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  buildstuff(x,H,B);
  vec eigs = Evals::eigenSpectrum(H,B);
  cout << "Result: " << endl << eigs.rows(0,states-1) << endl;

  x.randn();

  tStart = clock();
  optimizeH(x,200,50,1e-2,lambda);
  printf("optimizeH done in: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  buildstuff(x,H,B);
  eigs = Evals::eigenSpectrum(H,B);

  return eigs.rows(0,states-1);
}
