#include "System.h"

System::System(vec& masses, vec& charges, size_t De)
 : masses(masses), charges(charges), De(De) {

   N = masses.n_rows;
   n = N-1;

   setupCoordinates();
   setupLambdaMatrix();
   setupvArray();
   setupvList();
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

void System::setupvList(){
  vec** vArray;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = i+1; j < N; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        vList.push_back(vArray[i][j]);
        vprodList.emplace_back(vArray[i][j]*(vArray[i][j]).t());
      }
    }
  }
}
