#include "Presets.h"


// mat Presets::ThreeParticleSqueezeTwoDimSimul(size_t maxeval){
//   // IMPORTANT: USES OMEGA INSTEAD OF B_OSC
//   //vec trapwidth = logspace<vec>(-1,1.5,15);
//   vec omegatrap = logspace<vec>(-2,-2,1);
//   omegatrap = pow(omegatrap,-1);
//
//   size_t K = 25;
//   size_t states = 1;
//
//   // setup
//   arma_rng::set_seed_random();
//   vec masses(3);
//   masses << 130 << 130 << 6;
//   //masses << 130 << 130 << 130;
//   size_t n = masses.n_rows -1;
//
//   double intStr = -2.684;
//   double intWid = 0.1;
//
//   mat U = JacobiCoordinates::buildTransformationMatrix(masses);
//   mat lambdamat = JacobiCoordinates::buildReducedMassMatrix(masses);
//   mat Ui = U.i();
//
//   vec interstr = Utils::buildInterStr(masses,intStr,intWid);
//
//   vec interwidth = 1.0/pow(intWid,2)*ones<vec>(3);
//   fdcube inter = Utils::buildInteraction(masses.n_rows,interwidth,Ui);
//   cube Q = Utils::buildTrap(masses.n_rows,Ui);
//   SingleGaussPotential Gauss(interstr,inter);
//
//   mat results(omegatrap.n_rows,2);
//
//   for (size_t i = 0; i < omegatrap.n_rows; i++) {
//     double omegacur = omegatrap(i);
//     vec omegaSQ = Utils::buildOmegaSQ(masses, omegacur);
//     DoubleTrapPotential Trap(omegaSQ,omegaSQ,Q);
//
//     PotentialList Vstrat;
//     Vstrat.addPotential(&Gauss);
//     Vstrat.addPotential(&Trap);
//
//     MatrixElements elem(n,lambdamat,Vstrat);
//     Hamilton Hamil(K,n,elem);
//
//     vec Swidth = 0*ones<vec>(states);
//     //mat Amean = 0.46*ones<mat>(3*n,states); //for intWid = 1
//     mat Amean = 1*ones<mat>(3*n,states);
//     Amean(1,0) = 1.0/(2.5*omegacur);
//     Amean(2,0) = 1.0/(2.5*omegacur);
//     Amean(4,0) = 1.0/(0.75*omegacur);
//     Amean(5,0) = 1.0/(0.75*omegacur);
//
//     vec Kvec = K*ones<vec>(states);
//
//     try{
//       vec eigspec = Hamil.simpleOptimize(Amean,Swidth,Kvec,maxeval);
//       //double omegaz = sqrt(65.0/36.0 /(130.0/2.0) / pow(oscwidth,4)) + sqrt(65.0/399.0 /(2.0*130.0*6.0/(2.0*130.0 + 6.0))/pow(oscwidth,4));
//       results(i,0) = eigspec(0);
//       results(i,1) = eigspec(0) - 2.0*omegacur; //2.0*130.0*6.0/(2.0*130.0 +6.0) * omegacur;// -(65.0+6.0/(1+3.0/130.0))* omegacur;
//       cout << "Trapwidth " << omegacur << " --> " << results(i) << endl;
//     }
//     catch(...){
//       cout << "FAIL at " << omegacur << endl;
//       results(i) = 99;
//     }
//   }
//
//   mat data = join_horiz(omegatrap,results);
//   data.save("../data/DoubleSqueezeSimulAAB15",raw_ascii);
//   cout << "Three particle data: " << endl << data << endl;
//   return results;
// }

mat Presets::HeliumNew(size_t basisSize, size_t maxeval){
  // arma_rng::set_seed_random();
  //
  // vec masses(3);
  // vec charges(3);
  // masses << 7296 << 1 << 1;
  // charges << -1 << 2 << -1;
  // size_t K = basisSize;
  // size_t n = 2;
  // size_t states = 1;
  // size_t Kexpan = 10;
  //
  // mat U = JacobiCoordinates::buildTransformationMatrix(masses);
  // mat Ui = U.i();
  // mat lambdamat = JacobiCoordinates::buildReducedMassMatrix(masses);
  //
  //
  // vec b(Kexpan), c(Kexpan);
  // Utils::invExpansion(b,c,0.001,12.0,1e4);
  //
  // vec alpha = 1.0/pow(b,2);
  // fdcube inter = Utils::buildInteraction(masses.n_rows,alpha,Ui);
  // vec Qinter = Utils::buildQinter(charges);

  // CoulombPotential Vstrat(Qinter,c,inter);
  // MatrixElements elem(n,lambdamat,Vstrat);
  // vec **array;
  // array = new vec*[n+1];
  // for (size_t i = 0; i < n+1; i++) {
  //   array[i] = new vec[n+1];
  // }
  // for (size_t i = 0; i < n+1; i++) {
  //   for (size_t j = 0; j < n+1; j++) {
  //     array[i][j] = ones<vec>(n);
  //   }
  // }
  //Variational VariationalAnsatz1(n,elem,Ui);
  //
  // vec S_guess = 0*ones<vec>(3*n);
  // mat A_guess = eye<mat>(3*n,3*n);



  //VariationalAnsatz1.initializeBasis(1.0,K);
  //VariationalAnsatz1.printBasis();
  return eye<mat>(3,3);
}
