#include "SingleGaussPotential.h"

SingleGaussPotential::SingleGaussPotential(System& sys, double baseStr, double interactionRange)
  : vArrayList(sys.vArrayList), n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {
  alpha = 1.0/pow(interactionRange,2);
  interStr = calculateIntStr(sys.masses,baseStr,interactionRange);

  buildInteractions(sys.Ui);
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

void SingleGaussPotential::buildInteractions(mat& Ui){
  for (size_t i = 0; i<n; i++){
    size_t ibegin = De*i;
    size_t iend = De*i+(De-1);

    for (size_t j = i+1; j<n+1; j++){
      size_t jbegin = De*j;
      size_t jend = De*j+(De-1);

      mat inter = zeros<mat>(De*(n+1),De*(n+1));
      inter(span(ibegin,iend),span(ibegin,iend)) = eye(De,De);
      inter(span(jbegin,jend),span(jbegin,jend)) = eye(De,De);
      inter(span(ibegin,iend),span(jbegin,jend)) = -eye(De,De);
      inter(span(jbegin,jend),span(ibegin,iend)) = -eye(De,De);
      inter *= alpha;
      inter = Ui.t()*inter*Ui;
      interactions.emplace_back(inter(span(0,De*n-1),span(0,De*n-1)));
    }
  }
}

double SingleGaussPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  vec v = 2.0*A1*s1 + 2.0*A2*s2;

  vec kappavec(n*(n+1)/2);
  vec** vArray;
  double detBp, wBw;
  mat Bpinv;
  size_t count = 0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      detBp = detB;
      Bpinv = Binv;
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        wBw = 1+ alpha*dot((vArray[i][j]),Binv*vArray[i][j]);
        detBp *= wBw;
        Bpinv -= (Binv*vArray[i][j]*(vArray[i][j]).t()*Binv)/wBw;
      }
      kappavec(count) = pow(detBp,-3.0/De/2.0) * exp(-dot(s1,A1*s1) - dot(s2,A2*s2) + 1/4.0 *dot(v,Bpinv*v));
      count++;
    }
  }
  return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);
}

double SingleGaussPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  vec kappavec(n*(n+1)/2);
  vec** vArray;
  double detBp;
  size_t count = 0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      detBp = detB;
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        detBp *= 1+ alpha*dot((vArray[i][j]),Binv*vArray[i][j]);
      }
      kappavec(count) = pow(detBp,-3.0/De/2.0);
      count++;
    }
  }
  return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);
}

// double SingleGaussPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad){
//   vec kappavec(n*(n+1)/2);
//   vec** vArray;
//   double detBp, Gtemp, vBv;
//   size_t count = 0;
//
//   for (size_t i = 0; i < n; i++) {
//     for (size_t j = i+1; j < n+1; j++) {
//       detBp = detB;
//       for (size_t k = 0; k < De; k++) {
//         vArray = vArrayList.at(k);
//         vBv = 1+ alpha*dot((vArray[i][j]),Binv*vArray[i][j]);
//         detBp *= vBv;
//       }
//       kappavec(count) = pow(detBp,-3.0/De/2.0);
//       count++;
//     }
//   }
//
//   //
//   //  Calculate gradient for De = 1
//   //
//   // vec** wArray = vArrayList.at(0);
//   // Vgrad = zeros<vec>(n*(n+1)/2);
//   // vec intergrad(n*(n+1)/2);
//   // size_t intcount = 0, paramcount = 0;
//   // for (size_t iparam = 0; iparam < n; iparam++) {
//   //   for (size_t jparam = iparam+1; jparam < n+1; jparam++) {
//   //     intcount = 0;
//   //
//   //     for (size_t iint = 0; iint < n; iint++) {
//   //       for (size_t jint = iint+1; jint < n+1; jint++) {
//   //         vec w = wArray[iint][jint];
//   //         detBp = detB*(1+alpha*dot(w,Binv*w));
//   //         cout << "Int nr: " << intcount+1 << " . Param nr: " << paramcount+1 << endl;
//   //         cout << dot(w,Binv*w)*detBgrad(paramcount) << endl;
//   //         cout << dot(w,Binvgrad.slice(paramcount)*w)*detB << endl;
//   //         double ddetBpdx = (1+alpha*dot(w,Binv*w))*detBgrad(paramcount)+alpha*dot(w,Binvgrad.slice(paramcount)*w)*detB;
//   //         intergrad(intcount) = -1.5/detBp*kappavec(intcount)*ddetBpdx;
//   //         intcount++;
//   //       }
//   //     }
//   //     Vgrad(paramcount) = pow(datum::pi,3.0*n/2.0)*dot(interStr,intergrad);
//   //     paramcount++;
//   //   }
//   // }
//
//
//
//   //
//   //  Calculate for general De
//   //
//   vec** wArray;
//   Vgrad = zeros<vec>(De*n*(n+1)/2);
//   vec intergrad(n*(n+1)/2);
//   size_t intcount = 0, paramcount = 0;
//   for (size_t iparam = 0; iparam < n; iparam++) {
//     for (size_t jparam = iparam+1; jparam < n+1; jparam++) {
//       for (size_t kparam = 0; kparam < De; kparam++) {
//         intcount = 0;
//
//         for (size_t iint = 0; iint < n; iint++) {
//           for (size_t jint = iint+1; jint < n+1; jint++) {
//             detBp = detB;
//             double ddetBpdx = 0;
//             for (size_t kint = 0; kint < De; kint++) {
//               wArray = vArrayList.at(kint);
//               vec w = wArray[iint][jint];
//               detBp *= 1+alpha*dot(w,Binv*w);
//               ddetBpdx += (1+alpha*dot(w,Binv*w))*detBgrad(paramcount)+alpha*dot(w,Binvgrad.slice(paramcount)*w)*detB;
//             }
//             intergrad(intcount) = -1.5/detBp*kappavec(intcount)*ddetBpdx;
//             intcount++;
//           }
//         }
//         Vgrad(paramcount) = pow(datum::pi,3.0*n/2.0)*dot(interStr,intergrad);
//         paramcount++;
//       }
//     }
//   }
//
//
//   return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);
// }

double SingleGaussPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad){
  size_t count = 0;
  mat Bp, Bpinv;
  double detBp;
  vec kappavec(interactions.size());
  vec** vArray;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      Bp = A1+A2+interactions.at(count);
      detBp = det(Bp);
      Bpinv = inv(Bp);
      kappavec(count) = pow(detBp,-3.0/De/2.0);
      count++;
    }
  }

  // //
  // //   Calculate gradient for De = 1
  // //
  // vec** wArray = vArrayList.at(0);
  // Vgrad = zeros<vec>(n*(n+1)/2);
  // vec intergrad(n*(n+1)/2);
  // size_t intcount = 0, paramcount = 0;
  // for (size_t iparam = 0; iparam < n; iparam++) {
  //   for (size_t jparam = iparam+1; jparam < n+1; jparam++) {
  //     intcount = 0;
  //     vec v = wArray[iparam][jparam];
  //
  //     for (size_t iint = 0; iint < n; iint++) {
  //       for (size_t jint = iint+1; jint < n+1; jint++) {
  //         Bp = A1+A2+interactions.at(intcount);
  //         detBp = det(Bp);
  //         Bpinv = inv(Bp);
  //         double Mp = pow(datum::pi,3.0*n/2.0)*pow(detBp,-3.0/De/2.0);
  //         double ddetBpdx = detBp*trace(Bpinv*v*v.t());
  //
  //         intergrad(intcount) = -1.5/detBp * ddetBpdx*Mp;
  //         intcount++;
  //       }
  //     }
  //     Vgrad(paramcount) = dot(interStr,intergrad);
  //     paramcount++;
  //   }
  // }


  //
  //  Calculate gradient for general De
  //
  Vgrad = zeros<vec>(De*n*(n+1)/2);
  vec intergrad(n*(n+1)/2);
  size_t intcount = 0, paramcount = 0;
  for (size_t iparam = 0; iparam < n+1; iparam++) {
    for (size_t jparam = iparam+1; jparam < n+1; jparam++) {
      for (size_t kparam = 0; kparam < De; kparam++) {
        intcount = 0;
        vec** wArray = vArrayList.at(kparam);
        vec v  = wArray[iparam][jparam];

        for (size_t iint = 0; iint < n; iint++) {
          for (size_t jint = iint+1; jint < n+1; jint++) {
            Bp = A1+A2+interactions.at(intcount);
            detBp = det(Bp);
            Bpinv = inv(Bp);
            double Mp = pow(datum::pi,3.0*n/2.0)*pow(detBp,-3.0/De/2.0);
            double ddetBpdx = detBp*dot(v,Bpinv*v);

            intergrad(intcount) = -1.5/De/detBp * ddetBpdx*Mp;
            intcount++;
          }
        }
        Vgrad(paramcount) = dot(interStr,intergrad);
        paramcount++;
      }
    }
  }


  return pow(datum::pi,3.0*n/2.0)*dot(interStr,kappavec);
}
