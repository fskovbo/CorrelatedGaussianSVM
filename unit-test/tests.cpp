#include <gtest/gtest.h>
#include <iostream>
#include <armadillo>
#include <time.h>
#include <vector>
#include <tuple>

#include "System.h"
#include "SingleGaussPotential.h"
#include "MatrixElements.h"
#include "Variational.h"
#include "TrapPotential.h"
#include "PotentialList.h"
#include "PotentialStrategy.h"
#include "test_utils.hpp"

using namespace arma;
using namespace std;

std::vector<mat> Alist;
std::vector<vec> mlist;
std::vector<size_t> Dlist;
std::vector<int> potlist;
std::vector<size_t> Klist;


class gradientTestFixture : public ::testing::TestWithParam<std::tuple<size_t, vec, int, size_t, mat>> {
public:

  System* sys;
  PotentialList* Vstrat;
  MatrixElements* matElem;
  TrapPotential* Trap;
  SingleGaussPotential* Gauss;

   gradientTestFixture( ) {
     auto De      = std::get<0>(GetParam());
     auto masses  = std::get<1>(GetParam());
     auto VstratNR= std::get<2>(GetParam());

     vec charges  = zeros<vec>(masses.n_rows);
     sys          = new System(masses,charges,De);
     Vstrat       = new PotentialList();
     Trap         = new TrapPotential(*sys);
     Gauss        = new SingleGaussPotential(*sys);
     Trap->updateTrap(1e-1);

     if (VstratNR  == 1) { Vstrat->addPotential(Trap); }
     if (VstratNR  == 2) { Vstrat->addPotential(Gauss); }
     if (VstratNR  == 3) {
       Vstrat->addPotential(Trap);
       Vstrat->addPotential(Gauss);
     }

     matElem      = new MatrixElements(*sys,*Vstrat);
   }

   void SetUp( ) {

   }

   ~gradientTestFixture( )  {
       delete sys;
       delete Vstrat;
       delete matElem;
       delete Gauss;
       delete Trap;
   }
};

class dimensionTestFixture : public ::testing::TestWithParam<std::tuple<vec, int, mat>> {
public:

  System* sys1;
  PotentialList* Vstrat1;
  MatrixElements* matElem1;
  SingleGaussPotential* Gauss1;
  System* sys3;
  PotentialList* Vstrat3;
  MatrixElements* matElem3;
  SingleGaussPotential* Gauss3;

   dimensionTestFixture( ) {
     auto masses  = std::get<0>(GetParam());
     auto VstratNR= std::get<1>(GetParam());

     vec charges  = zeros<vec>(masses.n_rows);
     sys1         = new System(masses,charges,1);
     Vstrat1      = new PotentialList();
     Gauss1       = new SingleGaussPotential(*sys1);

     sys3         = new System(masses,charges,3);
     Vstrat3      = new PotentialList();
     Gauss3       = new SingleGaussPotential(*sys3);

     if (VstratNR  == 2) {
       Vstrat1->addPotential(Gauss1);
       Vstrat3->addPotential(Gauss3);
     }

     matElem1      = new MatrixElements(*sys1,*Vstrat1);
     matElem3      = new MatrixElements(*sys3,*Vstrat3);
   }

   void SetUp( ) {

   }

   ~dimensionTestFixture( )  {
       delete sys1;
       delete Vstrat1;
       delete matElem1;
       delete Gauss1;
       delete sys3;
       delete Vstrat3;
       delete matElem3;
       delete Gauss3;
   }
};

TEST_P(gradientTestFixture, MatchNumericalAnalyticalGradientK1){
  //  Compare numeric and analytical gradient using a basis of one Gaussian (A).

  double Hij, Bij;
  vec Hg, Bg;

  auto A = buildA(sys);
  matElem->calculateH_noShift(A, A, Hij, Bij, Hg, Bg);

  vec analytical  = (2.0*Hg -Hij/Bij*2.0*Bg)/Bij;
  vec numeric     = NumGradient(*sys, *matElem, A);

  ASSERT_EQ(analytical.n_rows, numeric.n_rows);
  for (size_t i = 0; i < analytical.n_rows; i++) {
    EXPECT_NEAR(analytical(i),numeric(i),5*1e-5);
  }
}

TEST_P(gradientTestFixture, MatchNumericalAnalyticalGradientShiftK1){
  //  Compare numeric and analytical gradient using a basis of one Gaussian (A).

  double Hij, Bij;
  vec Hg, Bg;

  auto A = buildA(sys);
  auto s = builds(sys);
  matElem->calculateH(A, A, s, s, Hij, Bij, Hg, Bg);

  vec analytical  = (2.0*Hg -Hij/Bij*2.0*Bg)/Bij;
  vec numeric     = NumGradient(*sys, *matElem, A, s);

  ASSERT_EQ(analytical.n_rows, numeric.n_rows);
  for (size_t i = 0; i < analytical.n_rows; i++) {
    EXPECT_NEAR(analytical(i),numeric(i),5*1e-5);
  }
}

TEST_P(gradientTestFixture, MatchNumericalAnalyticalGradientDiff){
  //  Compare numeric and analytical gradient for two different Gaussians.

  double Hij, Bij;
  vec Hg, Bg;

  auto Afixed = buildA(sys);
  auto Avar   = buildA(sys);
  matElem->calculateH_noShift(Afixed, Avar, Hij, Bij, Hg, Bg);

  vec analytical  = (Hg -Hij/Bij*Bg)/Bij;
  vec numeric     = NumGradient(*sys, *matElem, Afixed, Avar);

  ASSERT_EQ(analytical.n_rows, numeric.n_rows);
  for (size_t i = 0; i < analytical.n_rows; i++) {
    EXPECT_NEAR(analytical(i),numeric(i),5*1e-5);
  }
}

TEST_P(gradientTestFixture, MatchNumericalAnalyticalGradientShiftDiff){
  //  Compare numeric and analytical gradient for two different Gaussians.

  double Hij, Bij;
  vec Hg, Bg;

  auto Afixed = buildA(sys);
  auto Avar   = buildA(sys);
  auto sfixed = builds(sys);
  auto svar   = builds(sys);
  matElem->calculateH(Afixed, Avar, sfixed, svar, Hij, Bij, Hg, Bg);

  vec analytical  = (Hg -Hij/Bij*Bg)/Bij;
  vec numeric     = NumGradient(*sys, *matElem, Afixed, Avar, sfixed, svar);

  ASSERT_EQ(analytical.n_rows, numeric.n_rows);
  for (size_t i = 0; i < analytical.n_rows; i++) {
    EXPECT_NEAR(analytical(i),numeric(i),5*1e-5);
  }
}

TEST_P(gradientTestFixture, MatchOneNumericalAnalyticalGradientFullBasis){
  //  Compare numeric and analytical gradient for basisfnct nr index using a basis of K Gaussians.
  size_t K = std::get<3>(GetParam());
  if ( sys->n == 1 && K > 4) {
    K = 3;
  }
  for (size_t index = 0; index < K; index++) {
    size_t Npar = sys->De*sys->n*(sys->n+1)/2;
    double Hij, Bij;
    vec Hg, Bg;
    mat Ai, Aj;
    mat H(K,K), B(K,K);
    std::vector<vec> HG(Npar);
    std::vector<vec> BG(Npar);
    for (size_t i = 0; i < Npar; i++) {
      HG.at(i) = zeros<vec>(K);
      BG.at(i) = zeros<vec>(K);
    }

    cube basis(sys->De*sys->n,sys->De*sys->n,K);
    for (size_t i = 0; i < K; i++) {
      basis.slice(i) = buildA(sys);
    }


    for (size_t i = 0; i < K; i++) {
      Ai = basis.slice(i);
      for (size_t j = i; j < K; j++) {
        Aj = basis.slice(j);
        matElem->calculateH_noShift(Ai, Aj, Hij, Bij);
        H(i,j) = Hij;
        H(j,i) = Hij;
        B(i,j) = Bij;
        B(j,i) = Bij;
      }
    }

    for (size_t i = 0; i < K; i++) {
      Ai = basis.slice(i);
      Aj = basis.slice(index);
      matElem->calculateH_noShift(Ai, Aj, Hij, Bij, Hg, Bg);

      for (size_t j = 0; j < Npar; j++) {
        (HG[j])(i) = Hg(j);
        (BG[j])(i) = Bg(j);
      }
    }

    mat L(K,K);
    vec analytical(Npar);
    vec eigval;
    mat eigvec;
    bool status = chol(L,B,"lower");
    eig_sym(eigval,eigvec, L.i()*H*(L.t()).i() );

    vec c = (L.t()).i() * eigvec.col(0);
    double E = eigval(0);

    for (size_t i = 0; i < Npar; i++) {
      analytical(i) = 2.0*c(index)*dot(c,(HG[i]-E*BG[i]));
    }
    analytical /= dot(c , B*c);
    vec numeric = NumGradient(*sys, *matElem, basis, index);

    ASSERT_EQ(analytical.n_rows, numeric.n_rows);
    for (size_t i = 0; i < analytical.n_rows; i++) {
      EXPECT_NEAR(analytical(i),numeric(i),1e-4);
    }
  }
}

TEST_P(gradientTestFixture, MatchOneNumericalAnalyticalGradientShiftFullBasis){
  //  Compare numeric and analytical gradient for basisfnct nr index using a basis of K Gaussians.
  size_t K = std::get<3>(GetParam());
  if ( sys->n == 1 && K > 2) {
    K = 2;
  }
  for (size_t index = 0; index < K; index++) {
    size_t Npar = sys->De*sys->n*(sys->n+1)/2 +sys->De*sys->n ;
    double Hij, Bij;
    vec Hg, Bg, si, sj;
    mat Ai, Aj;
    mat H(K,K), B(K,K);
    std::vector<vec> HG(Npar);
    std::vector<vec> BG(Npar);
    for (size_t i = 0; i < Npar; i++) {
      HG.at(i) = zeros<vec>(K);
      BG.at(i) = zeros<vec>(K);
    }

    cube basis(sys->De*sys->n,sys->De*sys->n,K);
    mat shift(sys->De*sys->n,K);
    for (size_t i = 0; i < K; i++) {
      basis.slice(i) = buildA(sys);
      shift.col(i) = builds(sys);
    }


    for (size_t i = 0; i < K; i++) {
      Ai = basis.slice(i);
      si = shift.col(i);
      for (size_t j = i; j < K; j++) {
        Aj = basis.slice(j);
        sj = shift.col(j);
        matElem->calculateH(Ai, Aj, si, sj, Hij, Bij);
        H(i,j) = Hij;
        H(j,i) = Hij;
        B(i,j) = Bij;
        B(j,i) = Bij;
      }
    }

    for (size_t i = 0; i < K; i++) {
      Ai = basis.slice(i);
      Aj = basis.slice(index);
      si = shift.col(i);
      sj = shift.col(index);
      matElem->calculateH(Ai, Aj, si, sj, Hij, Bij, Hg, Bg);

      for (size_t j = 0; j < Npar; j++) {
        (HG[j])(i) = Hg(j);
        (BG[j])(i) = Bg(j);
      }
    }

    mat L(K,K);
    vec analytical(Npar);
    vec eigval;
    mat eigvec;
    bool status = chol(L,B,"lower");
    eig_sym(eigval,eigvec, L.i()*H*(L.t()).i() );

    vec c = (L.t()).i() * eigvec.col(0);
    double E = eigval(0);

    for (size_t i = 0; i < Npar; i++) {
      analytical(i) = 2.0*c(index)*dot(c,(HG[i]-E*BG[i]));
    }
    analytical /= dot(c , B*c);
    vec numeric = NumGradient(*sys, *matElem, basis, shift, index);

    ASSERT_EQ(analytical.n_rows, numeric.n_rows);
    for (size_t i = 0; i < analytical.n_rows; i++) {
      EXPECT_NEAR(analytical(i),numeric(i),1e-4);
    }
  }
}

TEST_P(dimensionTestFixture, CompareGradientsParametrisationK1){
  //  Compare analytical gradients for De = 1 and De = 3 using a basis of one Gaussian (A).

  double Hij, Bij;
  vec Hg, Bg;
  mat A1, A3;

  buildA(sys1,sys3,A1,A3);
  matElem1->calculateH_noShift(A1, A1, Hij, Bij, Hg, Bg);
  vec analytical1 = (2.0*Hg -Hij/Bij*2.0*Bg)/Bij;
  matElem3->calculateH_noShift(A3, A3, Hij, Bij, Hg, Bg);
  vec analytical3 = (2.0*Hg -Hij/Bij*2.0*Bg)/Bij;

  ASSERT_EQ(3*analytical1.n_rows, analytical3.n_rows);
  for (size_t i = 0; i < analytical1.n_rows; i++) {
    EXPECT_NEAR(analytical1(i),3*analytical3(3*i),1e-8);
    EXPECT_NEAR(analytical1(i),3*analytical3(3*i+1),1e-8);
    EXPECT_NEAR(analytical1(i),3*analytical3(3*i+2),1e-8);
  }
}

INSTANTIATE_TEST_CASE_P(gradtest,gradientTestFixture,
        ::testing::Combine(::testing::ValuesIn(Dlist),
                           ::testing::ValuesIn(mlist),
                           ::testing::ValuesIn(potlist),
                           ::testing::ValuesIn(Klist),
                           ::testing::ValuesIn(Alist)
                           ));

INSTANTIATE_TEST_CASE_P(dimtest,dimensionTestFixture,
       ::testing::Combine(::testing::ValuesIn(mlist),
                          ::testing::ValuesIn(potlist),
                          ::testing::ValuesIn(Alist)
                          ));


int main(int argc, char **argv) {
  size_t trials = 1;
  arma_rng::set_seed_random();

  Alist.reserve(trials);
  for (size_t i = 0; i < trials; i++) {
    Alist.emplace_back(eye<mat>(1,1));
  }

  vec m11 = {1 , 1};
  vec m12 = {15 , 1};
  vec m13 = {2752 , 1};
  vec m14 = {1 , 535};
  vec m15 = {7 , 141};
  vec m20 = {1 , 1 , 1};
  vec m21 = {343 , 1 , 1};
  vec m22 = {1 , 123 , 1};
  vec m23 = {1 , 1 , 575};
  vec m24 = {227 , 1 , 246};
  vec m25 = {13 , 6 , 735};

  mlist.push_back(m11);
  mlist.push_back(m12);
  mlist.push_back(m13);
  mlist.push_back(m14);
  mlist.push_back(m15);
  mlist.push_back(m20);
  mlist.push_back(m21);
  mlist.push_back(m22);
  mlist.push_back(m23);
  mlist.push_back(m24);
  mlist.push_back(m25);

  Dlist.push_back(1);
  Dlist.push_back(3);

  potlist.push_back(0);
  potlist.push_back(1);
  potlist.push_back(2);
  potlist.push_back(3);

  Klist.push_back(2);
  Klist.push_back(3);
  Klist.push_back(5);
  Klist.push_back(7);
  Klist.push_back(10);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
