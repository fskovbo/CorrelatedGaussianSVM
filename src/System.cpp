#include "System.h"

System::System(vec& masses, vec& charges)
 : masses(masses), charges(charges) {

   N = masses.n_rows;
   n = N-1;

   setupCoordinates();
   Ui = U.i();

   //
   // build lambda-matrix
   //
   lambdamat = zeros<mat>(3*n,3*n);
   for (size_t i = 0; i<n; i++){
       int ibegin = 3*i;
       int iend = 3*i+2;

       for (size_t j = 0; j<n; j++){
           int jbegin = 3*j;
           int jend = 3*j+2;

           for (size_t k = 0; k<N; k++){
               int kbegin = 3*k;
               int kend = 3*k+2;
               mat Uik = U(span(ibegin,iend),span(kbegin,kend));
               mat Ujk = U(span(jbegin,jend),span(kbegin,kend));

               lambdamat(span(ibegin,iend),span(jbegin,jend)) += Uik*Ujk/masses(k);
           }
       }
   }

   //
   // allocate vArrays
   //
   vec **vxArray = new vec*[N];
   vec **vyArray = new vec*[N];
   vec **vzArray = new vec*[N];
   for (size_t i = 0; i < n+1; i++) {
     vxArray[i] = new vec[N];
     vyArray[i] = new vec[N];
     vzArray[i] = new vec[N];
   }

   //
   // build w vectors - need coordinates seperately
   //
   vec wxArray[N][N];
   vec wyArray[N][N];
   vec wzArray[N][N];
   for (size_t i = 0; i < N; i++) {
     for (size_t j = i; j < N; j++) {
       wxArray[i][j] = vec(3*N,fill::zeros);
       wxArray[j][i] = vec(3*N,fill::zeros);
       wyArray[i][j] = vec(3*N,fill::zeros);
       wyArray[j][i] = vec(3*N,fill::zeros);
       wzArray[i][j] = vec(3*N,fill::zeros);
       wzArray[j][i] = vec(3*N,fill::zeros);
     }
   }

   for (size_t i = 0; i < N; i++) {
     (wxArray[i][i])(3*i) = 1;
     (wyArray[i][i])(3*i+1) = 1;
     (wzArray[i][i])(3*i+2) = 1;
     for (size_t j = i+1; j < N; j++) {
       (wxArray[i][j])(3*i) = 1;
       (wyArray[i][j])(3*i+1) = 1;
       (wzArray[i][j])(3*i+2) = 1;
       (wxArray[i][j])(3*j) = -1;
       (wyArray[i][j])(3*j+1) = -1;
       (wzArray[i][j])(3*j+2) = -1;

       wxArray[j][i] = wxArray[i][j];
       wyArray[j][i] = wyArray[i][j];
       wzArray[j][i] = wzArray[i][j];
     }
   }

   //
   // transform w vector such that v = (U^-1)^t * w
   //
   vec v;
   for (size_t i = 0; i < N; i++) {
     v = Ui.t() * wxArray[i][i];
     vxArray[i][i] = v.rows(0,3*(N-1)-1);
     v = Ui.t() * wyArray[i][i];
     vyArray[i][i] = v.rows(0,3*(N-1)-1);
     v = Ui.t() * wzArray[i][i];
     vzArray[i][i] = v.rows(0,3*(N-1)-1);

     for (size_t j = i+1; j < N; j++) {
       v = Ui.t() * wxArray[i][j];
       vxArray[i][j] = v.rows(0,3*(N-1)-1);
       vxArray[j][i] = vxArray[i][j];

       v = Ui.t() * wyArray[i][j];
       vyArray[i][j] = v.rows(0,3*(N-1)-1);
       vyArray[j][i] = vxArray[i][j];

       v = Ui.t() * wzArray[i][j];
       vzArray[i][j] = v.rows(0,3*(N-1)-1);
       vzArray[j][i] = vzArray[i][j];
     }
   }


   //
   // add vArrays to list
   //
   vArrayList = {vxArray, vyArray, vzArray};
}

void System::setupCoordinates(){
  //
  // build transformationmatrix
  //
  int D = 3;
  U = zeros<mat>(D*N,D*N);
  for (size_t i = 0; i<N; i++){
      int ibegin = D*i;
      int iend = D*i+(D-1);

      for (size_t j = 0; j<N; j++){
          int jbegin = D*j;
          int jend = D*j+(D-1);

          if (j > i+1){
              U(span(ibegin,iend),span(jbegin,jend)) = zeros<mat>(D,D);
          }
          else if (j == i+1){
              U(span(ibegin,iend),span(jbegin,jend)) = -1*eye(D,D);
          }
          else {
              U(span(ibegin,iend),span(jbegin,jend)) = masses(j) /(sum(masses.rows(0,i))) *eye(D,D);
          }
      }
  }
}

void System::setupCoordinates2(){
  int D = 3;
  U = zeros<mat>(D*N,D*N);
  for (size_t i = 0; i<N; i++){
      int ibegin = D*i;
      int iend = D*i+(D-1);
      double mu_i = 1;
      if (i != n) {
        mu_i = masses(i+1)*sum(masses.rows(0,i))/sum(masses.rows(0,i+1));
      }

      for (size_t j = 0; j<N; j++){
          int jbegin = D*j;
          int jend = D*j+(D-1);

          if (j > i+1){
              U(span(ibegin,iend),span(jbegin,jend)) = zeros<mat>(D,D);
          }
          else if (j == i+1){
              U(span(ibegin,iend),span(jbegin,jend)) = -sqrt(mu_i)*eye(D,D);
          }
          else {
              U(span(ibegin,iend),span(jbegin,jend)) = sqrt(mu_i) * masses(j) /(sum(masses.rows(0,i))) *eye(D,D);
          }
      }
  }
}
