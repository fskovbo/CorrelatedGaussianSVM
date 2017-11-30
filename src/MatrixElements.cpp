#include "MatrixElements.h"

MatrixElements::MatrixElements(System& sys, PotentialStrategy& Vstrat)
: Vstrat(Vstrat), n(sys.n), De(sys.De), lambda(sys.lambdamat), vArrayList(sys.vArrayList) {

}

void MatrixElements::calculateH(mat& A1, mat& A2, vec& s1, vec& s2, double& Hij, double& Bij){
  mat B = A1 + A2;
  mat Bi = inv_sympd(B);
  double detB = det(B);
  vec v = 2.0*A1*s1 + 2.0*A2*s2;
  vec u = 0.5 * Bi*v;
  mat prod12 = A1*0.5*lambda*A2;

  double overlap = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 0.25*dot(v,Bi*v));
  double T = overlap*(6.0/De*trace(prod12*Bi) + 4.0*dot(u-s1,prod12*(u-s2)));
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
  size_t count = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray                    = vArrayList.at(k);
        Bigrad.slice(De*count+k)  = -Bi*vArray[i][j]*(vArray[i][j]).t()*Bi;
        detBgrad(De*count+k)      = detB * dot(vArray[i][j],Bi*vArray[i][j]);
        Tgrad2(De*count+k)        = trace(A1*0.5*lambda*A1*Bigrad.slice(De*count+k));
      }
      count++;
    }
  }
  Mgrad = -1.5/detB *overlap*detBgrad;
  vec Tgrad = 6.0/De*trace(prod12*Bi)*Mgrad - 6.0/De*Tgrad2*overlap; // HVORFOR MINUS???
  vec Vgrad(De*n*(n+1)/2);

  double T = overlap*(6.0/De*trace(prod12*Bi) );
  double V = Vstrat.calculateExpectedPotential_noShift(A1, A2, Bi, detB, Vgrad, Bigrad, detBgrad);

  Hij   = T+V;
  Bij   = overlap;
  Hgrad = Tgrad+Vgrad;



  // cout << Bigrad << endl;
  // cout << detBgrad << endl;
  // cout << Tgrad2 << endl;
  // cout << "Bgrad: " << Mgrad << endl;
  // cout << "Tgrad: " << Tgrad << endl;
  // cout << "Vgrad: " << Vgrad << endl;
  // while (1) {
  //   /* code */
  // }
}
