#include "Variational.h"

Variational::Variational(System& sys, MatrixElements& matElem)
: n(sys.n), De(sys.De), matElem(matElem), vArrayList(sys.vArrayList) {
  K = 0;
}


double Variational::groundStateEnergy(){
  mat L(K,K);
  bool status = chol(L,B,"lower");
  if (status) {
    vec eigs = eig_sym( L.i()*H*(L.t()).i() );
    return eigs(0);
  }
  else{
    return 9999*1e10;
  }
}

mat Variational::generateRandomGaussian(vec& Ameanval, vec& coeffs){
  size_t count = 0;
  mat A = zeros<mat>(De*n,De*n);
  vec alpha;
  vec** vArray;

  for (size_t i = 0; i < n+1; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        alpha = 1.0/pow(-Ameanval(k)*log(randu<vec>(n*(n+1)/2)),2);
        A += alpha(count) * (vArray[i][j] * (vArray[i][j]).t());
        coeffs(De*count+k) = alpha(count);
      }
      count++;
    }
  }
  return A;
}

double Variational::initializeBasis(size_t basisSize){
  K = basisSize;
  H.resize(K,K);
  B.resize(K,K);
  basis.set_size(De*n,De*n,K);
  basisCoefficients.set_size(De*n*(n+1)/2,K);

  vec coeffs(De*n*(n+1)/2);
  vec startingGuess = 10*ones<vec>(De);

  for (size_t i = 0; i < K; i++) {
    basis.slice(i) = generateRandomGaussian(startingGuess,coeffs);
    basisCoefficients.col(i) = coeffs;
  }

  mat A1(De*n,De*n), A2(De*n,De*n);
  double Hij, Bij;
  for (size_t i = 0; i < K; i++) {
    A1 = basis.slice(i);
    for (size_t j = i; j < K; j++) {
      A2 = basis.slice(j);

      matElem.calculateH_noShift(A1,A2,Hij,Bij);
      H(i,j) = Hij;
      H(j,i) = Hij;
      B(i,j) = Bij;
      B(j,i) = Bij;
    }
  }
  return groundStateEnergy();
}

vec Variational::sweepStochastic(size_t sweeps, size_t trials, vec& Ameanval){
  mat Atrial, Acurrent;
  double Hij, Bij, Etrial;
  double Ebest = groundStateEnergy();
  mat Hbest = H, Bbest = B;
  vec trialCoeffs(De*n*(n+1)/2);
  vec results = Ebest*ones<vec>(sweeps+1);

  for (size_t l = 0; l < sweeps; l++) {
    for (size_t j = 0; j < K; j++) {
      for (size_t k = 0; k < trials; k++) {
        Atrial = generateRandomGaussian(Ameanval,trialCoeffs);

        for (size_t i = 0; i < K; i++) {

          if (j == i) {
            matElem.calculateH_noShift(Atrial,Atrial,Hij,Bij);
            H(i,i) = Hij;
            B(i,i) = Bij;
          }
          else{
            Acurrent = basis.slice(i);
            matElem.calculateH_noShift(Acurrent,Atrial,Hij,Bij);
            H(i,j) = Hij;
            H(j,i) = Hij;
            B(i,j) = Bij;
            B(j,i) = Bij;
          }

        }

        Etrial = groundStateEnergy();
        if (Etrial < Ebest) { //if trial is better: update basis
          Ebest = Etrial;
          Hbest = H;
          Bbest = B;
          basis.slice(j) = Atrial;
          basisCoefficients.col(j) = trialCoeffs;
        }
        else{ //otherwise revert to old H,B
          H = Hbest;
          B = Bbest;
        }

      }
    }
    cout << "Energy after stochastic sweep " << l+1 << ": " << Ebest << "\n";
    results(l+1) = Ebest;
  }
  return results;
}

typedef struct {
    size_t index, n, K, De;
    vector<vec**>& vArrayList;
    mat H, B;
    cube& basis;
    MatrixElements& matElem;
} my_function_data;


double Variational::myvfunc(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_function_data *d = reinterpret_cast<my_function_data*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De;
  vector<vec**>& vArrayList = d->vArrayList;
  mat H = d->H, B = d->B;
  cube& basis = d->basis;
  MatrixElements& matElem = d->matElem;

  double Hij, Bij;
  mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
  size_t count = 0;
  vec** vArray;

  for (size_t i = 0; i < n+1; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        Atrial += x[De*count+k] * (vArray[i][j] * (vArray[i][j]).t());
      }
      count++;
    }
  }

  for (size_t j = 0; j < K; j++) {
    if (j == index) {
      matElem.calculateH_noShift(Atrial,Atrial,Hij,Bij);
      H(index,index) = Hij;
      B(index,index) = Bij;
    } else {
      Acurrent = basis.slice(j);

      matElem.calculateH_noShift(Acurrent,Atrial,Hij,Bij);
      H(j,index) = Hij;
      H(index,j) = Hij;
      B(j,index) = Bij;
      B(index,j) = Bij;
    }
  }

  mat L(K,K);
  bool status = chol(L,B,"lower");
  if (status) {
    vec eigs = eig_sym( L.i()*H*(L.t()).i() );
    return eigs(0);
  }
  else{
    return 9999*1e10;
  }
}

vec Variational::sweepDeterministic(size_t sweeps){
  size_t Npar = De*n*(n+1)/2;
  vec results(sweeps), xstart(Npar);
  vec** vArray;

  //
  //  NLOpt setup
  //
  std::vector<double> lb(Npar);
  std::vector<double> xs(Npar);
  for (size_t i = 0; i < Npar; i++) {
    lb[i] = 1e-6;
  }
  nlopt::opt opt(nlopt::LN_NELDERMEAD, Npar);
  opt.set_lower_bounds(lb);
  opt.set_xtol_abs(1e-10); // tolerance on parametres
  double minf;

  for (size_t l = 0; l < sweeps; l++) {
    for (size_t index = 0; index < K; index++) {
      //
      // optimize basis function index using its current values as starting guess
      //
      xstart = basisCoefficients.col(index);
      for (size_t i = 0; i < Npar; i++) {
        xs[i] = xstart(i);
      }
      my_function_data data = { index,n,K,De,vArrayList,H,B,basis,matElem };
      opt.set_min_objective(myvfunc, &data);

      bool status = false;
      size_t attempts = 0;
      while (!status && attempts < 5) {
        try{
          nlopt::result optresult = opt.optimize(xs, minf);
          status = true;
        }
        catch (const std::exception& e) {
          attempts++;
        }
      }

      for (size_t i = 0; i < Npar; i++) {
        xstart(i) = xs[i];
      }

      // ----------------------------------------- //
      double Hij, Bij;
      mat Acurrent(De*n,De*n), Anew = zeros<mat>(De*n,De*n);
      size_t count = 0;
      for (size_t i = 0; i < n+1; i++) {
        for (size_t j = i+1; j < n+1; j++) {
          for (size_t k = 0; k < De; k++) {
            vArray = vArrayList.at(k);
            Anew += xs[De*count+k] * (vArray[i][j] * (vArray[i][j]).t());
          }
          count++;
        }
      }

      for (size_t j = 0; j < K; j++) {
        if (j == index) {
          matElem.calculateH_noShift(Anew,Anew,Hij,Bij);
          H(index,index) = Hij;
          B(index,index) = Bij;
        } else {
          Acurrent = basis.slice(j);

          matElem.calculateH_noShift(Acurrent,Anew,Hij,Bij);
          H(j,index) = Hij;
          H(index,j) = Hij;
          B(j,index) = Bij;
          B(index,j) = Bij;
        }
      }
      // ----------------------------------------- //
      //
      // add optimized basis function to basis
      //
      basis.slice(index) = Anew;
      basisCoefficients.col(index) = xstart;
    }
    cout << "Energy after deterministic sweep " << l+1 << ": " << minf << "\n";
    results(l) = minf;
  }
  return results;
}


vec Variational::sweepDeterministicCMAES(size_t sweeps, size_t maxeval){
  vec results(sweeps), xstart(De*n*(n+1)/2);
  mat Anew(3*n,3*n);
  size_t index, vcount;
  double Ebest;
  vec** vArray;

  function<double(vec&)> fitness = [&](vec& alpha){
    double Hij, Bij;
    mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
    alpha = abs(alpha); // ensure only positive values
    size_t count = 0;

    for (size_t i = 0; i < n+1; i++) {
      for (size_t j = i+1; j < n+1; j++) {
        for (size_t k = 0; k < De; k++) {
          vArray = vArrayList.at(k);
          Atrial += alpha(De*count+k) * (vArray[i][j] * (vArray[i][j]).t());
        }
        count++;
      }
    }

    for (size_t j = 0; j < K; j++) {
      if (j == index) {
        matElem.calculateH_noShift(Atrial,Atrial,Hij,Bij);
        H(index,index) = Hij;
        B(index,index) = Bij;
      } else {
        Acurrent = basis.slice(j);

        matElem.calculateH_noShift(Acurrent,Atrial,Hij,Bij);
        H(j,index) = Hij;
        H(index,j) = Hij;
        B(j,index) = Bij;
        B(index,j) = Bij;
      }
    }

    return groundStateEnergy();
  };

  for (size_t l = 0; l < sweeps; l++) {
    for (index = 0; index < K; index++) {
      //
      // optimize basis function index using its current values as starting guess
      //
      xstart = basisCoefficients.col(index);
      CMAES::optimize(fitness, xstart, 200, 50, 1e-2, maxeval);
      Ebest = fitness(xstart); // needed to set H,B corresponding to optimized parameters

      //
      // add optimized basis function to basis
      //
      vcount = 0;
      Anew.zeros();
      for (size_t i = 0; i < n+1; i++) {
        for (size_t j = i+1; j < n+1; j++) {
          for (size_t k = 0; k < De; k++) {
            vArray = vArrayList.at(k);
            Anew += xstart(De*vcount+k) * (vArray[i][j] * (vArray[i][j]).t());
          }
          vcount++;
        }
      }
      basis.slice(index) = Anew;
      basisCoefficients.col(index) = xstart;
    }
    cout << "Energy after deterministic sweep " << l+1 << ": " << Ebest << "\n";
    results(l) = Ebest;
  }
  return results;
}

void Variational::printBasis(){
  cout << "Current basis:" << endl << basis << endl;
}

void Variational::printShift(){
  cout << "Current shift:" << endl << shift << endl;
}
