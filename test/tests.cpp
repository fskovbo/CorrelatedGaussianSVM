#include "tests.h"

bool tests::testU(){
  vec mass1(2);
  vec mass2(3);
  mass1 << 1 << 2000;
  mass2 << 1 << 1 << 10000;

  mat U1(6,6);
  U1(span(0,2),span(0,2)) = eye<mat>(3,3);
  U1(span(0,2),span(3,5)) = -eye<mat>(3,3);
  U1(span(3,5),span(0,2)) = mass1(0)/sum(mass1) * eye<mat>(3,3);
  U1(span(3,5),span(3,5)) = mass1(1)/sum(mass1) * eye<mat>(3,3);
  mat U1test = JacobiCoordinates::buildTransformationMatrix(mass1);

  mat diff1 = U1-U1test;
  cout << "premade U" << endl << U1 << endl << "generated U" << endl << U1test << endl;
  cout << "difference" << endl << diff1 << endl;

  mat U2 = zeros<mat>(9,9);
  U2(span(0,2),span(0,2)) = eye<mat>(3,3);
  U2(span(0,2),span(3,5)) = -eye<mat>(3,3);
  U2(span(3,5),span(0,2)) = mass2(0)/(mass2(0)+mass2(1)) * eye<mat>(3,3);
  U2(span(3,5),span(3,5)) = mass2(1)/(mass2(0)+mass2(1)) * eye<mat>(3,3);
  U2(span(3,5),span(6,8)) = -eye<mat>(3,3);
  U2(span(6,8),span(0,2)) = mass2(0)/sum(mass2) * eye<mat>(3,3);
  U2(span(6,8),span(3,5)) = mass2(1)/sum(mass2) * eye<mat>(3,3);
  U2(span(6,8),span(6,8)) = mass2(2)/sum(mass2) * eye<mat>(3,3);
  mat U2test = JacobiCoordinates::buildTransformationMatrix(mass2);

  mat diff2 = U2-U2test;
  cout << "premade U" << endl << U2 << endl << "generated U" << endl << U2test << endl;
  cout << "difference" << endl << diff2 << endl;

  return true;
}

bool tests::testCoulomb(){
  size_t Kexpan = 8;
  vec masses1(2);
  vec charges1(2);
  vec masses2(3);
  vec charges2(3);
  masses1 << 1 << 1836;
  charges1 << -1 << 1;
  masses2 << 1 << 7296 << 1;
  charges2 << -1 << 2 << -1;

  mat U1 = JacobiCoordinates::buildTransformationMatrix(masses1);
  mat U2 = JacobiCoordinates::buildTransformationMatrix(masses2);

  vec b(Kexpan), c(Kexpan);
  Utils::invExpansion(b,c,0.001,12.0,1e4);

  mat Ui1 = U1.i();
  mat Ui2 = U2.i();
  vec alpha = 1.0/pow(b,2);
  fdcube inter1 = Utils::buildInteraction(masses1.n_rows,alpha,Ui1);
  fdcube inter2 = Utils::buildInteraction(masses2.n_rows,alpha,Ui2);
  vec Qinter1 = Utils::buildQinter(charges1);
  vec Qinter2 = Utils::buildQinter(charges2);
  Qinter2(1) = 0; //kun til test

  inter1.printall();
  inter2.printall();
  cout << U2 << endl << Qinter2 << endl;

  CoulombPotential Vstrat1(Qinter1,c,inter1);
  CoulombPotential Vstrat2(Qinter2,c,inter2);

  mat A1 = eye<mat>(3,3);
  vec s1 = zeros<vec>(3);

  mat A2 = eye<mat>(6,6);
  vec s2 = zeros<vec>(6);

  double pot1 = Vstrat1.calculateExpectedPotential(A1,A1,s1,s1);
  double pot2 = Vstrat2.calculateExpectedPotential(A2,A2,s2,s2);

  cout << "<V1> = " << pot1 << endl;
  cout << "<V2> = " << pot2 << endl;


  cout << "simple test ... " << endl;
  vec beta(2);
  vec cb(2);
  beta << 10 << 1;
  cb << 10 << 1;
  fdcube inter3 = Utils::buildInteraction(masses2.n_rows,beta,Ui2);
  inter3.printall();
  CoulombPotential Vstrat3(Qinter2,cb,inter3);
  double pot3 = Vstrat3.calculateExpectedPotential(A2,A2,s2,s2);
  cout << "<V3> = " << pot3 << endl;


  return true;
}
