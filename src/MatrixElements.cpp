#include "MatrixElements.h"

MatrixElements::MatrixElements(System& sys, PotentialStrategy& Vstrat)
: Vstrat(Vstrat), n(sys.n), De(sys.De), Lambda(0.5*sys.lambdamat), vArrayList(sys.vArrayList), vList(sys.vList) {

  NparA = vList.size();
}

void MatrixElements::calculateH(mat& A1, mat& A2, vec& s1, vec& s2, double& Hij, double& Bij){
  mat B = A1 + A2;

  mat L = chol(B,"lower");
  mat temp = eye<mat>(size(B));
  mat x = solve(trimatl(L), temp,solve_opts::fast);
  B = solve(trimatu(L.t()),x,solve_opts::fast); // inplace inverse

  double detB = 1;
  for (size_t i = 0; i < L.n_rows; i++) {
    detB *= L(i,i);
  }
  detB *= detB;

  vec v = s1+s2;
  vec u = 0.5*B*v;

  double overlap = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(0.25*dot(v,B*v));
  double T = overlap*(6.0/De*trace(A1*Lambda*A2*B) + dot((s1-2.0*A1*u),Lambda*(s2-2.0*A2*u)));
  double V = Vstrat.calculateExpectedPotential(A1, A2, s1, s2, B, detB);

  Hij = T+V;
  Bij = overlap;
}

void MatrixElements::calculateH_noShift(mat& A1, mat& A2, double& Hij, double& Bij){
  mat B = A1 + A2;

  mat L = chol(B,"lower");
  mat temp = eye<mat>(size(B));
  mat x = solve(trimatl(L), temp,solve_opts::fast);
  B = solve(trimatu(L.t()),x,solve_opts::fast); // inplace inverse

  double detB = 1;
  for (size_t i = 0; i < L.n_rows; i++) {
    detB *= L(i,i);
  }
  detB *= detB;

  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0);
  double T = overlap*(6.0/De*trace(A1*Lambda*A2*B) );
  double V = Vstrat.calculateExpectedPotential_noShift(A1, A2, B, detB);

  Hij = T+V;
  Bij = overlap;
}

void MatrixElements::calculateH_noShift(mat& A1, mat& A2, double& Hij, double& Bij, vec& Hgrad, vec& Mgrad){
  mat B = A1 + A2;

  mat L = chol(B,"lower");
  mat temp = eye<mat>(size(B));
  mat x = solve(trimatl(L), temp,solve_opts::fast);
  B = solve(trimatu(L.t()),x,solve_opts::fast); // inplace inverse

  double detB = 1;
  for (size_t i = 0; i < L.n_rows; i++) {
    detB *= L(i,i);
  }
  detB *= detB;

  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0);

  //
  //  gradient
  //
  cube Bigrad(De*n,De*n,NparA);
  vec detBgrad(NparA), Tgrad(NparA), Vgrad(NparA);
  size_t count = 0;
  for (auto& w : vList){
    Bigrad.slice(count) = -B*w*w.t()*B;
    detBgrad(count) = detB*dot(w,B*w);
    Tgrad(count) = 6.0/De*overlap*trace(A1*Lambda*(A2*Bigrad.slice(count)+w*w.t()*B));
    count++;
  }
  Mgrad = -1.5/De/detB *overlap*detBgrad;
  Tgrad += 6.0/De*trace(A1*Lambda*A2*B)*Mgrad;

  double T = overlap*(6.0/De*trace(A1*Lambda*A2*B) );
  double V = Vstrat.calculateExpectedPotential_noShift(A1, A2, B, detB, Vgrad, Bigrad, detBgrad);

  Hij   = T+V;
  Bij   = overlap;
  Hgrad = Tgrad+Vgrad;

}

void MatrixElements::calculateH(mat& A1, mat& A2, vec& s1, vec& s2, double& Hij, double& Bij, vec& Hgrad, vec& Mgrad){
  mat B = A1 + A2;

  mat L = chol(B,"lower");
  mat temp = eye<mat>(size(B));
  mat x = solve(trimatl(L), temp,solve_opts::fast);
  B = solve(trimatu(L.t()),x,solve_opts::fast); // inplace inverse

  double detB = 1;
  for (size_t i = 0; i < L.n_rows; i++) {
    detB *= L(i,i);
  }
  detB *= detB;

  vec v = s1+s2;
  vec u = 0.5 * B*v;

  double overlap = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(0.25*dot(v,B*v));

  //
  //  gradient
  //
  vec S1 = s1-2.0*A1*u;
  vec S2 = s2-2.0*A2*u;

  cube Bigrad(De*n,De*n,NparA);
  vec detBgrad(NparA), Tgrad_A(NparA), Vgrad_A(NparA), Vgrad_s(De*n), Mgrad_A(NparA);
  size_t count = 0;
  for (auto& w : vList){
    Bigrad.slice(count) = -B*w*w.t()*B;
    detBgrad(count)     = detB*dot(w,B*w);
    Mgrad_A(count)      = 0.25*overlap*dot(v,Bigrad.slice(count)*v);
    Tgrad_A(count)      = 6.0/De*overlap*trace(A1*Lambda*(A2*Bigrad.slice(count)+w*w.t()*B))
                        + overlap*dot(-2.0*A1*Bigrad.slice(count)*v, Lambda*S2)
                        + overlap*dot(S1,Lambda*(-w*w.t()*B*v - A2*Bigrad.slice(count)*v));
    count++;
  }
  double T    = 6.0/De*trace(A1*Lambda*A2*B) + dot(S1,Lambda*S2);
  Mgrad_A    += -1.5/De/detB *overlap*detBgrad;
  Tgrad_A    += T*Mgrad_A;
  vec Mgrad_s = 0.5*B*v*overlap;
  vec Tgrad_s = T*Mgrad_s
              + (Lambda*s1-2.0*Lambda*A1*u-B*A1*Lambda*s2-B*A2*Lambda*s1+2.0*B*A2*Lambda*A1*u+2.0*B*A1*Lambda*A2*u)*overlap;

  T          *= overlap;
  double V    = Vstrat.calculateExpectedPotential(A1, A2, s1, s2, B, detB, Vgrad_A, Vgrad_s, Bigrad, detBgrad);

  Hij         = T+V;
  Bij         = overlap;
  vec Hgrad_A = Tgrad_A + Vgrad_A;
  vec Hgrad_s = Tgrad_s + Vgrad_s;

  Hgrad       = join_vert(Hgrad_A, Hgrad_s);
  Mgrad       = join_vert(Mgrad_A, Mgrad_s);
}
