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
  vec v = s1 + s2;

  vec kappavec(n*(n+1)/2);
  vec** vArray;
  double detBp;
  mat Bp(size(A1)), x(size(A1)), L(size(A1)), temp = eye<mat>(size(A1));
  size_t count = 0;

  for (auto& inter : interactions){
    Bp  = A1+A2+inter;
    L   = chol(Bp,"lower");
    x   = solve(trimatl(L), temp,solve_opts::fast);
    Bp  = solve(trimatu(L.t()),x,solve_opts::fast); // inplace inverse

    detBp = 1;
    for (size_t i = 0; i < L.n_rows; i++) {
      detBp *= L(i,i);
    }
    detBp *= detBp;

    kappavec(count) = pow(detBp,-3.0/De/2.0) * exp(0.25*dot(v,Bp*v));
    count++;
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


double SingleGaussPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad){
  size_t count = 0;
  mat Bp(size(A1)), x(size(A1)), L(size(A1)), temp = eye<mat>(size(A1));
  double detBp, Mp, ddetBpdx;
  vec kappavec(interactions.size()), intergrad(interactions.size());
  Vgrad = zeros<vec>(De*n*(n+1)/2);
  size_t intcount = 0, paramcount = 0;

  for (size_t iparam = 0; iparam < n+1; iparam++) {
    for (size_t jparam = iparam+1; jparam < n+1; jparam++) {
      for (size_t kparam = 0; kparam < De; kparam++) {
        intcount      = 0;
        vec** wArray  = vArrayList.at(kparam);
        vec v         = wArray[iparam][jparam];

        for (size_t iint = 0; iint < n; iint++) {
          for (size_t jint = iint+1; jint < n+1; jint++) {
            Bp  = A1+A2+interactions.at(intcount);
            L   = chol(Bp,"lower");
            x   = solve(trimatl(L), temp,solve_opts::fast);
            Bp  = solve(trimatu(L.t()),x,solve_opts::fast); // inplace inverse

            detBp = 1;
            for (size_t i = 0; i < L.n_rows; i++) {
              detBp *= L(i,i);
            }
            detBp *= detBp;

            Mp            = pow(datum::pi,3.0*n/2.0)*pow(detBp,-3.0/De/2.0);
            ddetBpdx      = detBp*dot(v,Bp*v);

            kappavec(intcount)  = Mp;
            intergrad(intcount) = -1.5/De/detBp * ddetBpdx*Mp;
            intcount++;
          }
        }
        Vgrad(paramcount) = dot(interStr,intergrad);
        paramcount++;
      }
    }
  }
  return dot(interStr,kappavec);
}

double SingleGaussPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2,
                                                    mat& Binv, double detB, vec& Vgrad_A, vec& Vgrad_s,
                                                    cube& Binvgrad, vec& detBgrad)
{
  size_t count = 0;
  mat Bp(size(A1)), x(size(A1)), L(size(A1)), temp = eye<mat>(size(A1));
  double detBp, Mp, ddetBpdx;
  vec kappavec(interactions.size()), intergrad_A(interactions.size());
  vec v = s1+s2;
  Vgrad_A = zeros<vec>(De*n*(n+1)/2);
  size_t intcount = 0, paramcount = 0;

  for (size_t iparam = 0; iparam < n+1; iparam++) {
    for (size_t jparam = iparam+1; jparam < n+1; jparam++) {
      for (size_t kparam = 0; kparam < De; kparam++) {
        intcount      = 0;
        vec** wArray  = vArrayList.at(kparam);
        vec w         = wArray[iparam][jparam];
        Vgrad_s = zeros<vec>(De*n);

        for (size_t iint = 0; iint < n; iint++) {
          for (size_t jint = iint+1; jint < n+1; jint++) {
            Bp  = A1+A2+interactions.at(intcount);
            L   = chol(Bp,"lower");
            x   = solve(trimatl(L), temp,solve_opts::fast);
            Bp  = solve(trimatu(L.t()),x,solve_opts::fast); // inplace inverse

            detBp = 1;
            for (size_t i = 0; i < L.n_rows; i++) {
              detBp *= L(i,i);
            }
            detBp *= detBp;

            Mp            = pow(datum::pi,3.0*n/2.0)*pow(detBp,-3.0/De/2.0)*exp(0.25*dot(v,Bp*v));
            ddetBpdx      = detBp*dot(w,Bp*w);

            kappavec(intcount)    = Mp;
            intergrad_A(intcount) = (0.25*dot(v,Binvgrad.slice(paramcount)*v) -1.5/De/detBp * ddetBpdx)*Mp;
            Vgrad_s              += interStr(intcount)*0.5*Bp*v*Mp;
            intcount++;
          }
        }
        Vgrad_A(paramcount) = dot(interStr,intergrad_A);
        paramcount++;
      }
    }
  }
  return dot(interStr,kappavec);
}
