#include "Utils.h"

void Utils::invExpansion(vec& b, vec& c, double min, double max, int res){
  assert (b.n_rows == c.n_rows);
  unsigned int n = b.n_rows;

  vec x = linspace<vec>(min,max,res);

  vec lnb = linspace<vec>(log(min),log(max),n);
  b = exp(lnb);

  mat A = zeros<mat>(res,n);

  for (int ix = 0; ix<res; ix++){
      for (size_t ib = 0; ib<n; ib++){
          A(ix,ib) = exp(-pow((x(ix)/b(ib)),2));
      }
  }

  vec y = zeros<vec>(res);
  for (int i = 0; i<res; i++){
      y(i) = 1/x(i);
  }

  c = solve(A,y);

}

fdcube Utils::buildInteraction(size_t N, vec& alpha, mat& Ui){
  size_t n = N-1;
  size_t nalp = alpha.n_rows;
  fdcube Rcube(n*3,n*3,nalp,N*(N-1)/2);
  size_t interactionNr = 0;

  for (size_t i = 0; i<n; i++){
      size_t ibegin = 3*i;
      size_t iend = 3*i+2;

      for (size_t j = i+1; j<N; j++){
          size_t jbegin = 3*j;
          size_t jend = 3*j+2;

          cube interactionCube = zeros<cube>(3*n,3*n,nalp);
          for (size_t k = 0; k < nalp; k++) {
            mat interaction = zeros<mat>(3*N,3*N);
            interaction(span(ibegin,iend),span(ibegin,iend)) = eye(3,3);
            interaction(span(jbegin,jend),span(jbegin,jend)) = eye(3,3);
            interaction(span(ibegin,iend),span(jbegin,jend)) = -eye(3,3);
            interaction(span(jbegin,jend),span(ibegin,iend)) = -eye(3,3);

            interaction *= alpha(k);
            mat RFull = Ui.t()*interaction*Ui;
            interactionCube.slice(k) = RFull(span(0,3*n-1),span(0,3*n-1));
          }

          Rcube.setPiece(interactionNr,interactionCube);
          interactionNr++;
      }
  }

  return Rcube;
}

vec Utils::buildInterStr(vec& masses, double baseStr, double interwidth){
  size_t N = masses.n_rows;
  vec interstr(N*(N-1)/2);
  size_t interactionNr = 0;

  for (size_t i = 0; i < N-1; i++) {
    double mi = masses(i);
    for (size_t j = i+1; j < N; j++) {
      double mj = masses(j);
      double mu = mi*mj/(mi+mj);

      //interstr(interactionNr) = baseStr/(mu*pow(interwidth,2));
      interstr(interactionNr) = baseStr;
      interactionNr++;
    }
  }
  return interstr;
}

vec Utils::buildQinter(vec& Q){
  size_t N = Q.n_elem;
  size_t interactionNr = 0;
  vec Qinter = zeros<vec>(N*(N-1)/2);

  for (size_t i = 0; i<N-1; i++){
      for (size_t j = i+1; j<N; j++){
          Qinter(interactionNr) = Q(i)*Q(j);
          interactionNr++;
      }
  }
  return Qinter;
}

cube Utils::buildTrap(size_t N, mat& Ui){
  //skal ikke bruges
  size_t n = N-1;
  cube Qcube(n*3,n*3,N);

  for (size_t i = 0; i < N; i++) {
    mat Q = zeros<mat>(3*N,3*N);
    Q(3*i+2,3*i+2) = 1;

    mat QFull = Ui.t()*Q*Ui;
    Qcube.slice(i) = QFull(span(0,3*n-1),span(0,3*n-1));
  }
  return Qcube;
}

vec Utils::buildOmegaSQ(vec& masses, double oscWidth){
  //midlertidig løsning for m1=m2=m3=1
  /*
  size_t N = masses.n_rows;
  size_t n = N-1;
  vec omegaSQ(n);

  if ( n == 1) {
    omegaSQ(0) = 1/(4*pow(oscWidth,4));
    return omegaSQ;
  }
  if ( n == 2) {
    omegaSQ(0) = 1/(4*pow(oscWidth,4));
    omegaSQ(1) = 1/(3*pow(oscWidth,4));
    return omegaSQ;
  }
  else {
    omegaSQ = zeros<vec>(n);
  }
*/

  //hardcode AAB system , mA = 130, mB = 6
  size_t N = masses.n_rows;
  size_t n = N-1;
  vec omegaSQ(n);

  if ( n == 1) {
    omegaSQ(0) = 1/(4*pow(oscWidth,4));
    return omegaSQ;
  }
  if ( n == 2) {
    //omegaSQ(0) = 65.0/36.0 /(2*pow(oscWidth,4));
    //omegaSQ(1) = 65.0/399.0 /(2*pow(oscWidth,4));
    omegaSQ(0) = 0.5*65.0*pow(oscWidth,2);
    omegaSQ(1) = 0.5*6.0/(1+3.0/130.0)*pow(oscWidth,2);
    return omegaSQ;
  }
  else {
    omegaSQ = zeros<vec>(n);
  }

  return omegaSQ;
}

void Utils::translateToBasis(vec& x, cube& A, mat& s, bool diagonal){
  size_t K = A.n_slices;
  size_t nD = A.n_cols;

  if (diagonal) {
    assert(x.n_rows == 2*nD*K);

    mat Atemp = zeros<mat>(nD,nD);
    vec stemp = zeros<vec>(nD,1);

    for (size_t i = 0; i < K; i++) {
      Atemp.diag() = x.rows(i*nD,(i+1)*nD-1);
      stemp = x.rows((i+K)*nD,(i+K+1)*nD-1);

      A.slice(i) = Atemp;
      s.col(i) = stemp;
    }

  } else {
    assert(x.n_rows == nD*K*(nD+1));

    mat Atemp;
    vec stemp;

    for (size_t i = 0; i < K; i++) {
      Atemp = x.rows(i*nD*nD,(i+1)*nD*nD-1);
      Atemp.reshape(nD,nD);
      stemp = x.rows((i+K*nD)*nD,(i+K*nD+1)*nD-1);

      A.slice(i) = Atemp;
      s.col(i) = stemp;
    }
  }
}

void Utils::translateToList(vec& x, cube& A, mat& s, bool diagonal){
  size_t K = A.n_slices;
  size_t nD = A.n_cols;

  if (diagonal) {
    assert(x.n_rows == 2*nD*K);

    for (size_t i = 0; i < K; i++) {
      x.rows(i*nD,(i+1)*nD-1) = A.slice(i).diag();
      x.rows((i+K)*nD,(i+K+1)*nD-1) = s.col(i);
    }

  } else {
    assert(x.n_rows == nD*K*(nD+1));

    mat xtemp;

    for (size_t i = 0; i < K; i++) {
      xtemp = A.slice(i);
      xtemp.reshape(nD*nD,1);
      x.rows(i*nD*nD,(i+1)*nD*nD-1) = xtemp;
      x.rows((i+K*nD)*nD,(i+K*nD+1)*nD-1) = s.col(i);
    }
  }

}

void Utils::vec2symmetricMat(vec& vector, mat& symMatrix){
  size_t N = symMatrix.n_cols;
  assert(vector.n_rows == N*(N+1)/2);

  size_t count = 0;
  double val;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = i; j < N; j++) {
      val = vector(count);
      symMatrix(i,j) = val;
      symMatrix(j,i) = val;
      count++;
    }
  }
}

void Utils::symmetricMat2vec(vec& vector, mat& symMatrix){
  size_t N = symMatrix.n_cols;
  assert(vector.n_rows == N*(N+1)/2);

  size_t count = 0;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = i; j < N; j++) {
      vector(count) = symMatrix(i,j);
      count++;
    }
  }
}
