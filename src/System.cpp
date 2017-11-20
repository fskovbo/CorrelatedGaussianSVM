#include "System.h"

System::System(vec& masses, vec& charges, size_t De)
 : masses(masses), charges(charges), De(De) {

   N = masses.n_rows;
   n = N-1;

   setupCoordinates();
   setupLambdaMatrix();
   setupvArray2();
}

void System::setupCoordinates(){
  //
  // build transformationmatrix
  //
  U = zeros<mat>(De*N,De*N);
  for (size_t i = 0; i<N; i++){
      int ibegin = De*i;
      int iend = De*i+(De-1);

      for (size_t j = 0; j<N; j++){
          int jbegin = De*j;
          int jend = De*j+(De-1);

          if (j > i+1){
              U(span(ibegin,iend),span(jbegin,jend)) = zeros<mat>(De,De);
          }
          else if (j == i+1){
              U(span(ibegin,iend),span(jbegin,jend)) = -1*eye(De,De);
          }
          else {
              U(span(ibegin,iend),span(jbegin,jend)) = masses(j) /(sum(masses.rows(0,i))) *eye(De,De);
          }
      }
  }
  Ui = U.i();
}

void System::setupCoordinates2(){
  U = zeros<mat>(De*N,De*N);
  for (size_t i = 0; i<N; i++){
      int ibegin = De*i;
      int iend = De*i+(De-1);
      double mu_i = 1;
      if (i != n) {
        mu_i = masses(i+1)*sum(masses.rows(0,i))/sum(masses.rows(0,i+1));
      }

      for (size_t j = 0; j<N; j++){
          int jbegin = De*j;
          int jend = De*j+(De-1);

          if (j > i+1){
              U(span(ibegin,iend),span(jbegin,jend)) = zeros<mat>(De,De);
          }
          else if (j == i+1){
              U(span(ibegin,iend),span(jbegin,jend)) = -sqrt(mu_i)*eye(De,De);
          }
          else {
              U(span(ibegin,iend),span(jbegin,jend)) = sqrt(mu_i) * masses(j) /(sum(masses.rows(0,i))) *eye(De,De);
          }
      }
  }
  Ui = U.i();
}

void System::setupLambdaMatrix(){
  lambdamat = zeros<mat>(De*n,De*n);
  for (size_t i = 0; i<n; i++){
      int ibegin = De*i;
      int iend = De*i+(De-1);

      for (size_t j = 0; j<n; j++){
          int jbegin = De*j;
          int jend = De*j+(De-1);

          for (size_t k = 0; k<N; k++){
              int kbegin = De*k;
              int kend = De*k+(De-1);
              mat Uik = U(span(ibegin,iend),span(kbegin,kend));
              mat Ujk = U(span(jbegin,jend),span(kbegin,kend));

              lambdamat(span(ibegin,iend),span(jbegin,jend)) += Uik*Ujk/masses(k);
          }
      }
  }
}

void System::setupvArray(){
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

void System::setupvArray2(){
  for (size_t k = 0; k < De; k++) {

    vec **vArray = new vec*[N];
    for (size_t i = 0; i < n+1; i++) {
      vArray[i] = new vec[N];
    }

    vec wArray[N][N];
    for (size_t i = 0; i < N; i++) {
      for (size_t j = i; j < N; j++) {
        wArray[i][j] = vec(De*N,fill::zeros);
        wArray[j][i] = vec(De*N,fill::zeros);
      }
    }

    for (size_t i = 0; i < N; i++) {
      (wArray[i][i])(De*i+k) = 1;
      for (size_t j = i+1; j < N; j++) {
        (wArray[i][j])(De*i+k) = 1;
        (wArray[i][j])(De*j+k) = -1;

        wArray[j][i] = wArray[i][j];
      }
    }

    vec v;
    for (size_t i = 0; i < N; i++) {
      v = Ui.t() * wArray[i][i];
      vArray[i][i] = v.rows(0,De*(N-1)-1);

      for (size_t j = i+1; j < N; j++) {
        v = Ui.t() * wArray[i][j];
        vArray[i][j] = v.rows(0,De*(N-1)-1);
        vArray[j][i] = vArray[i][j];
      }
    }

    vArrayList.emplace_back(vArray);
  }
}
