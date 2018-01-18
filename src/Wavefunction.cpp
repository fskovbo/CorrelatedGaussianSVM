#include "Wavefunction.hpp"

Wavefunction::Wavefunction(System& sys, cube& basis, mat& shift, vec& coeffs)
 : K(coeffs.n_rows), n(sys.n), De(sys.De), basis(basis), shift(shift), coeffs(coeffs), U(sys.U), Ui(sys.Ui) {

}

Wavefunction::Wavefunction(System& sys, std::string filename)
 : n(sys.n), De(sys.De), U(sys.U), Ui(sys.Ui) {

 bool status;
 std::string cn = filename + "_coeffs";
 std::string bn = filename + "_basis";
 std::string sn = filename + "_shift";
 status         = coeffs.save(cn,raw_ascii);
 status         = basis.save(bn,raw_ascii);
 status         = shift.save(sn,raw_ascii);

 if (!status) {
  std::cout << "Loading Error!" << '\n';
 }
 else{
   K = coeffs.n_rows;
 }
}

double Wavefunction::factorial(double x){
  return std::tgamma(x+1);
}

void Wavefunction::calculateOverlap(mat& O, mat& A1, mat& A2, vec& s1, vec& s2, double c1, double c2, double& Oij, double& Bij){
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
  Oij   = c1*c2*Bij*(1.5/De*trace(O*B) + dot(u,O*u));
}

double Wavefunction::calculateExptValue(mat& O){
  mat Om(K,K), B(K,K);
  mat A1, A2;
  vec s1, s2;
  double c1, c2, Oij, Bij;

  for (size_t i = 0; i < K; i++) {
    A1 = basis.slice(i);
    s1 = shift.col(i);
    c1 = coeffs(i);

    for (size_t j = i; j < K; j++) {
      A2 = basis.slice(j);
      s2 = shift.col(j);
      c2 = coeffs(j);

      calculateOverlap(O,A1,A2,s1,s2,c1,c2,Oij,Bij);
      Om(i,j) = Oij;
      Om(j,i) = Oij;
      B(i,j)  = Bij;
      B(j,i)  = Bij;
    }
  }
  mat L(K,K);
  bool status = chol(L,B,"lower");
  if (status) {
    vec eigs = eig_sym( L.i()*Om*(L.t()).i() );
    return eigs(0);
  }
  else{
    return 9999*1e10;
  }
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
    size_t ib = De*i, ie = De*(i+1)-1, jb;
    if (i = n) { jb = 0; }
    else       { jb = De*(i+1); }
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

vec Wavefunction::RMSdistances(){
  vec RMS(De*(n+1));

  for (size_t i = 0; i < De*(n+1); i++) {
    mat Uir = Ui( span(i,i),span(0,De*n-1) );
    mat F   = Uir.t()*Uir;
    RMS(i)  = calculateExptValue(F);
  }

  return RMS;
}

double Wavefunction::Symmetrization(){
  cube permutations = buildPermutations();
  mat S             = zeros<mat>(K,K);
  mat B(K,K);

  for (size_t k = 0; k < permutations.n_slices; k++) {
    mat perm = permutations.slice(k);
    mat Bp(K,K), I = eye(De*n,De*n);
    mat A1, A2, A2p;
    vec s1, s2, s2p;
    double c1, c2, Oij, Bij;

    for (size_t i = 0; i < K; i++) {
      A1 = basis.slice(i);
      s1 = shift.col(i);
      c1 = coeffs(i);

      for (size_t j = 0; j < K; j++) {
        A2  = basis.slice(j);
        A2p = perm.t()*A2*perm;
        s2  = shift.col(j);
        s2p = perm*s2;
        c2  = coeffs(j);

        calculateOverlap(I,A1,A2,s1,s2,c1,c2,Oij,Bij);
        B(i,j)  = Bij;
        B(j,i)  = Bij;
        calculateOverlap(I,A1,A2p,s1,s2p,c1,c2,Oij,Bij);
        Bp(i,j) = Bij;
        Bp(j,i) = Bij;
      }
    }
    S += Bp;
  }

  mat L(K,K);
  bool status = chol(L,B,"lower");
  if (status) {
    vec eigs = eig_sym( L.i()*S*(L.t()).i() );
    return eigs(0)/permutations.n_slices;
  }
  else{
    return 9999*1e10;
  }
}
