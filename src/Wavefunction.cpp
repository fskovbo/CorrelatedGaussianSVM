#include "Wavefunction.hpp"

Wavefunction::Wavefunction(System& sys, cube& basis, mat& shift, vec& coeffs)
 : K(coeffs.n_rows), n(sys.n), De(sys.De), basis(basis), shift(shift),
   coeffs(coeffs), U(sys.U), Ui(sys.Ui) {

}

Wavefunction::Wavefunction(System& sys, std::string filename)
 : n(sys.n), De(sys.De), U(sys.U), Ui(sys.Ui) {

 bool status;
 mat tmp;
 std::string cn = filename + "_coeffs";
 std::string bn = filename + "_basis";
 std::string sn = filename + "_shift";
 status         = coeffs.load(cn,raw_ascii);
 K              = coeffs.n_rows;
 basis          = zeros<cube>(De*n,De*n,K);
 status         = tmp.load(bn,raw_ascii);
 for (size_t i = 0; i < K; i++) {
   basis.slice(i) = tmp(span(i*De*n,(i+1)*De*n-1),span(0,De*n-1));
 }
 status         = shift.load(sn,raw_ascii);

 if (!status) {
  std::cout << "Loading Error!" << '\n';
 }
}

void Wavefunction::calculateR(mat& R, mat& A1, mat& A2, vec& s1, vec& s2, double& Rij, double& Bij){
  mat B     = A1 + A2;

  mat L     = chol(B,"lower");
  mat temp  = eye<mat>(size(B));
  mat x     = solve(trimatl(L), temp,solve_opts::fast);
  B         = solve(trimatu(L.t()),x,solve_opts::fast); // inplace inverse

  double detB = 1;
  for (size_t i = 0; i < L.n_rows; i++) {
    detB *= L(i,i);
  }
  detB *= detB;

  vec v = s1 + s2;
  vec u = 0.5*B*v;

  Bij   = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(0.25*dot(v,B*v));
  Rij   = Bij*(1.5/De*trace(R*B) + dot(u,R*u));
}

double Wavefunction::factorial(double x){
  return std::tgamma(x+1);
}

cube Wavefunction::buildPermutations(){
  int Nperm = factorial(n+1); //only for N=3
  cube perm( De*n,De*n,Nperm );
  size_t count = 0;

  for (size_t i = 0; i < n; i++) {
    size_t ib = De*i;
    size_t ie = De*(i+1)-1;

    for (size_t j = i+1; j < n+1; j++) {
      size_t jb = De*j;
      size_t je = De*(j+1)-1;

      mat tmp                      = eye(De*(n+1),De*(n+1));
      tmp(span(ib,ie),span(ib,ie)) = zeros<mat>(De,De);
      tmp(span(jb,je),span(jb,je)) = zeros<mat>(De,De);
      tmp(span(ib,ie),span(jb,je)) = eye(De,De);
      tmp(span(jb,je),span(ib,ie)) = eye(De,De);
      tmp                          = U*tmp*Ui;
      perm.slice(count++)          = tmp(span(0,De*n-1),span(0,De*n-1));
    }
  }

  mat tmp = zeros<mat>(De*(n+1),De*(n+1));
  for (size_t i = 0; i < n+1; i++) {
    size_t ib = De*i;
    size_t ie = De*(i+1)-1;
    size_t jb;
    if (i == n) { jb = 0; }
    else        { jb = De*(i+1); }
    size_t je = jb+De-1;

    tmp(span(ib,ie),span(jb,je)) = eye(De,De);
  }

  for (size_t i = 0; i < n+1; i++) {
    mat temp = tmp;
    for (size_t j = 0; j < i; j++) {
      temp = temp*tmp;
    }

    temp                = U*temp*Ui;
    perm.slice(count++) = temp(span(0,De*n-1),span(0,De*n-1));
  }
  return perm;
}

void Wavefunction::Symmetrize(cube& symbasis, mat& symshift){
  cube permutations = buildPermutations();

  for (size_t i = 0; i < K; i++) {
    mat A = basis.slice(i), As = zeros<mat>(De*n,De*n);
    vec s = shift.col(i),   ss = zeros<vec>(3*n);

    for (size_t k = 0; k < permutations.n_slices; k++) {
      mat perm = permutations.slice(k);
      As      += perm.t()*A*perm;
      ss      += perm*s;
    }

    symbasis.slice(i) = As/permutations.n_slices;
    symshift.col(i)   = ss/permutations.n_slices;
  }
}

double Wavefunction::Symmetrization(){
  cube symbasis(basis);
  mat symshift(shift);
  Symmetrize(symbasis,symshift);
  mat Ai, Aj;
  vec si, sj;
  double Rij, Bij;
  mat F = zeros<mat>(De*n,De*n);
  mat S(K,K), B(K,K);

  for (size_t i = 0; i < K; i++) {
    Ai = basis.slice(i);
    si = shift.col(i);
    for (size_t j = 0; j < K; j++) {
      Aj = basis.slice(j);
      sj = shift.col(j);
      calculateR(F,Ai,Aj,si,sj,Rij,Bij);
      B(i,j) = Bij;

      Aj = symbasis.slice(j);
      sj = symshift.col(j);
      calculateR(F,Ai,Aj,si,sj,Rij,Bij);
      S(i,j) = Bij;
    }
  }

  return dot(coeffs,S*coeffs)/dot(coeffs,B*coeffs);
}

vec Wavefunction::RMSdistances(){
  vec RMS(De*(n+1));
  mat Ai, Aj;
  vec si, sj;
  double Rij, Bij;
  mat R(K,K), B(K,K);

  for (size_t k = 0; k < De*(n+1); k++) {
    mat Uir = Ui( span(k,k),span(0,De*n-1) );
    mat F   = Uir.t()*Uir;

    for (size_t i = 0; i < K; i++) {
      Ai = basis.slice(i);
      si = shift.col(i);
      for (size_t j = i; j < K; j++) {
        Aj = basis.slice(j);
        sj = shift.col(j);

        calculateR(F,Ai,Aj,si,sj,Rij,Bij);
        R(i,j) = Rij;
        R(j,i) = Rij;
        B(i,j) = Bij;
        B(j,i) = Bij;
      }
    }
    RMS(k) = dot(coeffs,R*coeffs)/dot(coeffs,B*coeffs);
  }
  return RMS;
}
