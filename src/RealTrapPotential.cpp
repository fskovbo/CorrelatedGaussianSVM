#include "RealTrapPotential.h"

RealTrapPotential::RealTrapPotential(System& sys, double trapLengthXY)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat), trapLengthXY(trapLengthXY) {

  XYmat = zeros<mat>(De*n,De*n);
  Zmat = zeros<mat>(De*n,De*n);

  for (size_t i = 0; i < n; i++) {
    XYmat(De*i,De*i)      = 1;
    XYmat(De*i+1,De*i+1)  = 1;
    Zmat(De*i+2,De*i+2)   = 1;
  }
}

RealTrapPotential::RealTrapPotential(System& sys, double trapLengthXY, double trapLengthZ)
  : n(sys.n), De(sys.De), lambdamat(sys.lambdamat), trapLengthXY(trapLengthXY) {

  XYmat = zeros<mat>(De*n,De*n);
  Zmat = zeros<mat>(De*n,De*n);

  for (size_t i = 0; i < n; i++) {
    XYmat(De*i,De*i)      = 1;
    XYmat(De*i+1,De*i+1)  = 1;
    Zmat(De*i+2,De*i+2)   = 1;
  }
  OmegaXY = lambdamat%XYmat * 0.5 * pow(trapLengthXY,-4);

  updateTrap(trapLengthZ);
}

void RealTrapPotential::updateTrap(double trapLengthZ){
  this->trapLengthZ = trapLengthZ;

  OmegaZ = lambdamat%Zmat * 0.5 * pow(trapLengthZ,-4);
}

double RealTrapPotential::gsExpectedVal(){
  double expZ   = 1.5*trace(lambdamat)/De/De/trapLengthZ/trapLengthZ;
  double expXY  = trace(lambdamat)/De/trapLengthXY/trapLengthXY;
  return expZ+expXY;
}

double RealTrapPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2, mat& Binv, double detB){
  vec v = s1 + s2;
  vec u = 0.5*Binv*v;
  double overlap = (pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)) * exp(0.25*dot(v,Binv*v));
  return overlap*(1.5/De*trace((OmegaXY+OmegaZ)*Binv) + dot(u,(OmegaXY+OmegaZ)*u));
}

double RealTrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB){
  return pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0) * 1.5/De*trace((OmegaXY+OmegaZ)*Binv);
}

double RealTrapPotential::calculateExpectedPotential_noShift(mat& A1, mat& A2, mat& Binv, double detB, vec& Vgrad, cube& Binvgrad, vec& detBgrad){
  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0);
  vec Mgrad = -1.5/De/detB *overlap*detBgrad;
  vec Vgrad2(Vgrad);

  size_t count = 0;
  Binvgrad.each_slice(
    [&](mat& X)
    {
      Vgrad(count++) = 1.5/De*overlap*trace((OmegaXY+OmegaZ)*X);
    }
  );
  Vgrad += 1.5/De*trace((OmegaXY+OmegaZ)*Binv)*Mgrad;
  return overlap * 1.5/De*trace((OmegaXY+OmegaZ)*Binv);
}

double RealTrapPotential::calculateExpectedPotential(mat& A1, mat& A2, vec& s1, vec& s2,
                                                 mat& Binv, double detB, vec& Vgrad_A, vec& Vgrad_s,
                                                 cube& Binvgrad, vec& detBgrad)
{
  vec v = s1+s2;
  vec u = 0.5*Binv*v;
  double overlap = pow(datum::pi,3.0*n/2.0)*pow(detB,-3.0/De/2.0)*exp(0.25*dot(v,Binv*v));
  double trapval = 1.5/De*trace((OmegaXY+OmegaZ)*Binv) + dot(u,(OmegaXY+OmegaZ)*u);

  vec Mgrad_A(size(Vgrad_A));
  size_t count = 0;
  Binvgrad.each_slice(
    [&](mat& X)
    {
      auto temp         = (OmegaXY+OmegaZ)*X;
      Vgrad_A(count)    = overlap*(1.5/De*trace(temp) + dot(u,temp*v));
      Mgrad_A(count++)  = 0.25*overlap*dot(v,X*v);
    }
  );

  Mgrad_A -= 1.5/De/detB *overlap*detBgrad;
  Vgrad_A += trapval*Mgrad_A;
  Vgrad_s = trapval*0.5*Binv*v*overlap + ((OmegaXY+OmegaZ)*Binv*u)*overlap;
  return overlap*trapval;
}
