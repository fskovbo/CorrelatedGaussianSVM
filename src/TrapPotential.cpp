#include "TrapPotential.h"

TrapPotential::TrapPotential(System& sys)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {

}

TrapPotential::TrapPotential(System& sys, vec trapLength)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat) {

  updateTrap(trapLength);
}

void TrapPotential::updateTrap(vec trapLength){
  mat trapmat = diagmat(repmat( pow(trapLength,-4) ,n,1));
  Omega       = 0.5*diagmat(pow(lambdamat.diag(),-1))%trapmat;
  gsEnergy    = 1.5/De*trace(diagmat(repmat( pow(trapLength,-2) ,n,1)));
}

double TrapPotential::gsExpectedVal(){
  return gsEnergy;
  // return 1.5*trace(lambdamat)/De/De/trapLength_cur/trapLength_cur;
}

double TrapPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  vec v = s1 + s2;
  vec u = 0.5*Binv*v;
  double overlap = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(0.25*dot(v,Binv*v));
  return overlap*(1.5/De*trace(Omega*Binv) + dot(u,Omega*u));
}

double TrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  return pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0) * 1.5/De*trace(Omega*Binv);
}

double TrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad){
  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0);
  vec Mgrad = -1.5/De/detB *overlap*detBgrad;
  vec Vgrad2(Vgrad);

  size_t count = 0;
  Binvgrad.each_slice(
    [&](mat& X)
    {
      Vgrad(count++) = 1.5/De*overlap*trace(Omega*X);
    }
  );
  Vgrad += 1.5/De*trace(Omega*Binv)*Mgrad;
  return overlap * 1.5/De*trace(Omega*Binv);
}

double TrapPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2,
                                                 mat& Binv, double detB, vec& Vgrad_A, vec& Vgrad_s,
                                                 cube& Binvgrad, vec& detBgrad)
{
  vec v = s1+s2;
  vec u = 0.5*Binv*v;
  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)*exp(0.25*dot(v,Binv*v));
  double trapval = 1.5/De*trace(Omega*Binv) + dot(u,Omega*u);

  vec Mgrad_A(size(Vgrad_A));
  size_t count = 0;
  Binvgrad.each_slice(
    [&](mat& X)
    {
      auto temp         = Omega*X;
      Vgrad_A(count)    = overlap*(1.5/De*trace(temp) + dot(u,temp*v));
      Mgrad_A(count++)  = 0.25*overlap*dot(v,X*v);
    }
  );

  Mgrad_A -= 1.5/De/detB *overlap*detBgrad;
  Vgrad_A += trapval*Mgrad_A;
  Vgrad_s = trapval*0.5*Binv*v*overlap + (Omega*Binv*u)*overlap;
  return overlap*trapval;
}
