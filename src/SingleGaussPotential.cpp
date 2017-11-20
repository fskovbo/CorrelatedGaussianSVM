#include "SingleGaussPotential.h"

SingleGaussPotential::SingleGaussPotential(System& sys, double baseStr, double interactionRange)
  : vArrayList(sys.vArrayList), n(sys.n), lambdamat(sys.lambdamat) {
  alpha = 1.0/pow(interactionRange,2);
  interStr = calculateIntStr(sys.masses,baseStr,interactionRange);
  // buildInteractions(sys.Ui);
}

vec SingleGaussPotential::calculateIntStr(vec& masses, double baseStr, double intRange){
  interStr = zeros<vec>(n*(n+1)/2);
  size_t count = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      double mu = masses(i)*masses(j)/(masses(i)+masses(j));
      interStr(count) = baseStr/(2.0*mu*intRange*intRange);
      count++;
    }
  }
  return interStr;
}

void SingleGaussPotential::buildInteractions(mat& invTrans){
  interactions.reserve(n*(n+1)/2);

  for (size_t i = 0; i < n; i++) {
    size_t ibegin = 3*i;
    size_t iend = 3*i+2;
    for (size_t j = i+1; j < n+1; j++) {
      size_t jbegin = 3*j;
      size_t jend = 3*j+2;

      mat temp = zeros<mat>(3*(n+1),3*(n+1));
      temp(span(ibegin,iend),span(ibegin,iend)) = eye(3,3);
      temp(span(jbegin,jend),span(jbegin,jend)) = eye(3,3);
      temp(span(ibegin,iend),span(jbegin,jend)) = -eye(3,3);
      temp(span(jbegin,jend),span(ibegin,iend)) = -eye(3,3);
      temp *= alpha;
      temp = invTrans.t()*temp*invTrans;
      mat temp_red = temp(span(0,3*n-1),span(0,3*n-1));
      interactions.push_back(temp_red);
    }
  }
}

double SingleGaussPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  // mat B = A1 + A2;
  // vec v = 2.0*A1*s1 + 2.0*A2*s2;
  //
  // size_t p = interactions.n_pieces();
  // size_t n = v.n_rows/3;
  // vec kappavec(p);
  // mat kappa(3*n,3*n);
  // cube Ri(3*n,3*n,1);
  //
  // for (size_t i = 0; i < p; i++) {
  //   Ri = interactions.getPiece(i);
  //   kappa = B + Ri.slice(0);
  //   kappavec(i) = 1.0/(sqrt(det(kappa))) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 1/4.0 *dot(v,kappa.i()*v));
  // }
  //
  // return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);
  vec v = 2.0*A1*s1 + 2.0*A2*s2;

  vec kappavec(n*(n+1)/2);

  vec** vxArray = vArrayList.at(0);
  vec** vyArray = vArrayList.at(1);
  vec** vzArray = vArrayList.at(2);
  double detBp, wBw_x, wBw_y, wBw_z;
  mat Bpinv;
  size_t count = 0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      detBp = detB;
      Bpinv = Binv;
      wBw_x = 1+ alpha*dot((vxArray[i][j]),Binv*vxArray[i][j]);
      wBw_y = 1+ alpha*dot((vyArray[i][j]),Binv*vyArray[i][j]);
      wBw_z = 1+ alpha*dot((vzArray[i][j]),Binv*vzArray[i][j]);

      detBp *= wBw_x * wBw_y * wBw_z;
      Bpinv -= (Binv*vxArray[i][j]*(vxArray[i][j]).t()*Binv)/wBw_x;
      Bpinv -= (Binv*vyArray[i][j]*(vyArray[i][j]).t()*Binv)/wBw_y;
      Bpinv -= (Binv*vzArray[i][j]*(vzArray[i][j]).t()*Binv)/wBw_z;

      kappavec(count) = 1.0/sqrt(detBp) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 1/4.0 *dot(v,Bpinv*v));
      count++;
    }
  }

  return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);

}

double SingleGaussPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  vec kappavec(n*(n+1)/2);

  vec** vxArray = vArrayList.at(0);
  vec** vyArray = vArrayList.at(1);
  vec** vzArray = vArrayList.at(2);

  double detBp;
  size_t count = 0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      detBp = detB;
      detBp *= 1+ alpha*dot((vxArray[i][j]),Binv*vxArray[i][j]);
      detBp *= 1+ alpha*dot((vyArray[i][j]),Binv*vyArray[i][j]);
      detBp *= 1+ alpha*dot((vzArray[i][j]),Binv*vzArray[i][j]);
      kappavec(count) = 1.0/sqrt(detBp);
      count++;
    }
  }
  return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);

  // ------------------------- //

  // mat B = A1 + A2;
  // mat kappa(3*n,3*n);
  // size_t count = 0;
  // double result2 = 0;
  //
  // for (auto& M : interactions){
  //   kappa = B + M;
  //   result2 += interStr(count)/(sqrt(det(kappa)));
  //   count++;
  // }
  // return result2 * pow(datum::pi,3.0*n/2.0);
}
