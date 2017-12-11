#include "MatrixElements.h"

MatrixElements::MatrixElements(System& sys, PotentialStrategy& Vstrat)
: Vstrat(Vstrat), n(sys.n), De(sys.De), lambda(sys.lambdamat), vArrayList(sys.vArrayList) {

}

void MatrixElements::calculateH(mat& A1, mat& A2, vec& s1, vec& s2, double& Hij, double& Bij){
  mat B = A1 + A2;
  mat Bi = inv_sympd(B);
  double detB = det(B);
  vec v = s1+s2;
  vec u = 0.5*Bi*v;
  mat prod12 = A1*0.5*lambda*A2;

  double overlap = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(0.25*dot(v,Bi*v));
  double T = overlap*(6.0/De*trace(A1*0.5*lambda*A2*Bi) + dot(s1-2.0*A1*u,0.5*lambda*(s2-2.0*A2*u)));
  double V = Vstrat.calculateExpectedPotential(A1, A2, s1, s2, Bi, detB);

  Hij = T+V;
  Bij = overlap;
}

void MatrixElements::calculateH_noShift(mat& A1, mat& A2, double& Hij, double& Bij){
  mat B = A1 + A2;
  mat Bi = inv_sympd(B);
  double detB = det(B);
  mat prod12 = A1*0.5*lambda*A2;

  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0);
  double T = overlap*(6.0/De*trace(prod12*Bi) );
  double V = Vstrat.calculateExpectedPotential_noShift(A1, A2, Bi, detB);

  Hij = T+V;
  Bij = overlap;
}

void MatrixElements::calculateH_noShift(mat& A1, mat& A2, double& Hij, double& Bij, vec& Hgrad, vec& Mgrad){
  //
  //  Gradient of A2
  //
  mat B = A1 + A2;
  mat Bi = inv_sympd(B);
  double detB = det(B);
  mat prod12 = A1*0.5*lambda*A2;

  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0);

  //
  //  gradient
  //
  vec** vArray;
  cube Bigrad(De*n,De*n,De*n*(n+1)/2);
  vec detBgrad(De*n*(n+1)/2);
  vec Tgrad2(De*n*(n+1)/2);
  vec Tgrad3(De*n*(n+1)/2);
  size_t count = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray                    = vArrayList.at(k);
        Bigrad.slice(De*count+k)  = -Bi*vArray[i][j]*(vArray[i][j]).t()*Bi;
        detBgrad(De*count+k)      = detB * dot(vArray[i][j],Bi*vArray[i][j]);
        Tgrad2(De*count+k)        = trace(A1*0.5*lambda*A2*Bigrad.slice(De*count+k)); // dB-1/da
        Tgrad3(De*count+k)        = trace(A1*0.5*lambda*vArray[i][j]*(vArray[i][j]).t()*Bi); // dA/da
      }
      count++;
    }
  }
  Mgrad = -1.5/De/detB *overlap*detBgrad;
  vec Tgrad = 6.0/De*trace(A1*0.5*lambda*A2*Bi)*Mgrad + 6.0/De*(Tgrad2+Tgrad3)*overlap;
  vec Vgrad(De*n*(n+1)/2);

  double T = overlap*(6.0/De*trace(prod12*Bi) );
  double V = Vstrat.calculateExpectedPotential_noShift(A1, A2, Bi, detB, Vgrad, Bigrad, detBgrad);

  Hij   = T+V;
  Bij   = overlap;
  Hgrad = Tgrad+Vgrad;

}

void MatrixElements::calculateH(mat& A1, mat& A2, vec& s1, vec& s2, double& Hij, double& Bij, vec& Hgrad, vec& Mgrad){
  mat B = A1 + A2;
  mat Bi = inv_sympd(B);
  double detB = det(B);
  vec v = s1+s2;
  vec u = 0.5 * Bi*v;

  double overlap = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(0.25*dot(v,Bi*v));

  //
  //  gradient
  //
  vec** vArray;
  cube Bigrad_A(De*n,De*n,De*n*(n+1)/2);
  vec detBgrad_A(De*n*(n+1)/2);
  vec Tgrad2_A(De*n*(n+1)/2);
  vec Tgrad3_A(De*n*(n+1)/2);
  vec Tgrad4_A(De*n*(n+1)/2);
  vec Tgrad5_A(De*n*(n+1)/2);
  vec Mgrad2_A(De*n*(n+1)/2);


  size_t count = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray                    = vArrayList.at(k);
        Bigrad_A.slice(De*count+k)= -Bi*vArray[i][j]*(vArray[i][j]).t()*Bi;
        detBgrad_A(De*count+k)    = detB * dot(vArray[i][j],Bi*vArray[i][j]);
        Mgrad2_A(De*count+k)      = dot(v,Bigrad_A.slice(De*count+k)*v);
        Tgrad2_A(De*count+k)      = trace(A1*0.5*lambda*A2*Bigrad_A.slice(De*count+k)); // dB-1/da
        Tgrad3_A(De*count+k)      = trace(A1*0.5*lambda*vArray[i][j]*(vArray[i][j]).t()*Bi); // dA/da
        Tgrad4_A(De*count+k)      = dot(-2.0*A1*Bigrad_A.slice(De*count+k)*v, 0.5*lambda*(s2-2.0*A2*u));
        Tgrad5_A(De*count+k)      = dot(s1-2.0*A1*u, 0.5*lambda * (-vArray[i][j]*(vArray[i][j]).t()*Bi*v -A2*Bigrad_A.slice(De*count+k)*v));
      }
      count++;
    }
  }
  vec Mgrad_A = 0.25*Mgrad2_A*overlap -1.5/De/detB *overlap*detBgrad_A;
  vec Mgrad_s = 0.5*Bi*v*overlap;
  vec Tgrad_A = (6.0/De*trace(A1*0.5*lambda*A2*Bi) + dot(s1-2.0*A1*u,0.5*lambda*(s2-2.0*A2*u)))*Mgrad_A
            + 6.0/De*(Tgrad2_A+Tgrad3_A)*overlap + (Tgrad4_A+Tgrad5_A)*overlap;
  vec Tgrad_s = (6.0/De*trace(A1*0.5*lambda*A2*Bi) + dot(s1-2.0*A1*u,0.5*lambda*(s2-2.0*A2*u)))*Mgrad_s
            + (-A1*Bi*0.5*lambda*(s2-2.0*A2*u) + (0.5*lambda - 0.5*lambda*A2*Bi)*(s1-2.0*A1*u) )*overlap;

  vec Vgrad_A(De*n*(n+1)/2);
  vec Vgrad_s(De*n);

  double T = overlap*(6.0/De*trace(A1*0.5*lambda*A2*Bi) + dot(s1-2.0*A1*u,0.5*lambda*(s2-2.0*A2*u)));
  double V = Vstrat.calculateExpectedPotential(A1, A2, s1, s2, Bi, detB, Vgrad_A, Vgrad_s, Bigrad_A, detBgrad_A);

  Hij         = T+V;
  Bij         = overlap;
  vec Hgrad_A = Tgrad_A + Vgrad_A;
  vec Hgrad_s = Tgrad_s + Vgrad_s;

  Hgrad       = join_vert(Hgrad_A, Hgrad_s);
  Mgrad       = join_vert(Mgrad_A, Mgrad_s);
}
