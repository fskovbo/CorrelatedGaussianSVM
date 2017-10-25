#include "JacobiCoordinates.h"

mat JacobiCoordinates::buildTransformationMatrix(vec& masses) {
    unsigned int bodies = masses.n_rows;
    mat U = zeros<mat>(bodies*3,bodies*3);

    for (size_t i = 0; i<bodies; i++){
        int ibegin = 3*i;
        int iend = 3*i+2;

        for (size_t j = 0; j<bodies; j++){
            int jbegin = 3*j;
            int jend = 3*j+2;

            if (j > i+1){
                U(span(ibegin,iend),span(jbegin,jend)) = zeros<mat>(3,3);
            }
            else if (j == i+1){
                U(span(ibegin,iend),span(jbegin,jend)) = -1*eye(3,3);
            }
            else {
                U(span(ibegin,iend),span(jbegin,jend)) = masses(j) /(sum(masses.rows(0,i))) *eye(3,3);
            }
        }
    }
    return U;
}

mat JacobiCoordinates::buildReducedMassMatrix(vec& masses){
  unsigned int bodies = masses.n_rows;
  mat lambda = zeros<mat>((bodies-1)*3,(bodies-1)*3);
  mat U = buildTransformationMatrix(masses);

  for (size_t i = 0; i<(bodies-1); i++){
      int ibegin = 3*i;
      int iend = 3*i+2;

      for (size_t j = 0; j<(bodies-1); j++){
          int jbegin = 3*j;
          int jend = 3*j+2;

          for (size_t k = 0; k<bodies; k++){
              int kbegin = 3*k;
              int kend = 3*k+2;
              mat Uik = U(span(ibegin,iend),span(kbegin,kend));
              mat Ujk = U(span(jbegin,jend),span(kbegin,kend));

              lambda(span(ibegin,iend),span(jbegin,jend)) += Uik*Ujk/masses(k);
          }
      }
  }
  return lambda;
}
