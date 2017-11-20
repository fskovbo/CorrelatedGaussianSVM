#include "Variational.h"

Variational::Variational(System& sys, MatrixElements& matElem)
: matElem(matElem) {
  K = 0;
  n = sys.n;
  vArrayList = sys.vArrayList;
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
  vec alphax = 1.0/pow(-Ameanval(0)*log(randu<vec>(n*(n+1)/2)),2);
  vec alphay = 1.0/pow(-Ameanval(1)*log(randu<vec>(n*(n+1)/2)),2);
  vec alphaz = 1.0/pow(-Ameanval(2)*log(randu<vec>(n*(n+1)/2)),2);

  size_t count = 0;
  mat A = zeros<mat>(3*n,3*n);
  vec** vxArray = vArrayList.at(0);
  vec** vyArray = vArrayList.at(1);
  vec** vzArray = vArrayList.at(2);

  for (size_t i = 0; i < n+1; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      A += alphax(count) * (vxArray[i][j] * (vxArray[i][j]).t());
      A += alphay(count) * (vyArray[i][j] * (vyArray[i][j]).t());
      A += alphaz(count) * (vzArray[i][j] * (vzArray[i][j]).t());
      coeffs(3*count) = alphax(count);
      coeffs(3*count+1) = alphay(count);
      coeffs(3*count+2) = alphaz(count);
      count++;
    }
  }
  return A;
}

double Variational::initializeBasis(size_t basisSize){
  K = basisSize;
  H.resize(K,K);
  B.resize(K,K);
  basis.set_size(3*n,3*n,K);
  basisCoefficients.set_size(3*n*(n+1)/2,K);

  vec coeffs(3*n*(n+1)/2);
  vec startingGuess = 10*ones<vec>(3);
  for (size_t i = 0; i < K; i++) {
    basis.slice(i) = generateRandomGaussian(startingGuess,coeffs);
    basisCoefficients.col(i) = coeffs;
  }

  mat A1(3*n,3*n), A2(3*n,3*n);
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
  vec trialCoeffs(3*n*(n+1)/2);
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
    size_t index, n, K;
    vector<vec**>& vArrayList;
    mat H, B;
    cube& basis;
    MatrixElements& matElem;
} my_function_data;


double Variational::myvfunc(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_function_data *d = reinterpret_cast<my_function_data*>(data);
  size_t index = d->index, n = d->n, K = d->K;
  vector<vec**>& vArrayList = d->vArrayList;
  mat H = d->H, B = d->B;
  cube& basis = d->basis;
  MatrixElements& matElem = d->matElem;

  double Hij, Bij;
  mat Acurrent(3*n,3*n), Atrial = zeros<mat>(3*n,3*n);
  size_t count = 0;
  vec** vxArray = vArrayList.at(0);
  vec** vyArray = vArrayList.at(1);
  vec** vzArray = vArrayList.at(2);

  for (size_t i = 0; i < n+1; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      Atrial += x[count] * (vxArray[i][j] * (vxArray[i][j]).t());
      count++;
      Atrial += x[count] * (vyArray[i][j] * (vyArray[i][j]).t());
      count++;
      Atrial += x[count] * (vzArray[i][j] * (vzArray[i][j]).t());
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
  vec results(sweeps), xstart(3*n*(n+1)/2);
  mat Anew(3*n,3*n);
  size_t index, vcount;
  double Ebest;
  vec** vxArray = vArrayList.at(0);
  vec** vyArray = vArrayList.at(1);
  vec** vzArray = vArrayList.at(2);

  //
  //  NLOpt setup
  //
  std::vector<double> lb(3*n*(n+1)/2);
  std::vector<double> xs(3*n*(n+1)/2);
  for (size_t i = 0; i < 3*n*(n+1)/2; i++) {
    lb[i] = 1e-6;
  }
  nlopt::opt opt(nlopt::LN_NELDERMEAD, 3*n*(n+1)/2);
  opt.set_lower_bounds(lb);
  opt.set_xtol_abs(1e-10); // tolerance on parametres
  double minf;

  for (size_t l = 0; l < sweeps; l++) {
    for (index = 0; index < K; index++) {
      //
      // optimize basis function index using its current values as starting guess
      //
      xstart = basisCoefficients.col(index);
      for (size_t i = 0; i < 3*n*(n+1)/2; i++) {
        xs[i] = xstart(i);
      }
      my_function_data data = { index,n,K,vArrayList,H,B,basis,matElem };
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

      for (size_t i = 0; i < 3*n*(n+1)/2; i++) {
        xstart(i) = xs[i];
      }
      Ebest = minf;

      // ----------------------------------------- //
      double Hij, Bij;
      mat Acurrent(3*n,3*n), Atrial = zeros<mat>(3*n,3*n);
      size_t count = 0;
      for (size_t i = 0; i < n+1; i++) {
        for (size_t j = i+1; j < n+1; j++) {
          Atrial += xs[count] * (vxArray[i][j] * (vxArray[i][j]).t());
          count++;
          Atrial += xs[count] * (vyArray[i][j] * (vyArray[i][j]).t());
          count++;
          Atrial += xs[count] * (vzArray[i][j] * (vzArray[i][j]).t());
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

      // ----------------------------------------- //

      //
      // add optimized basis function to basis
      //
      vcount = 0;
      Anew.zeros();
      for (size_t i = 0; i < n+1; i++) {
        for (size_t j = i+1; j < n+1; j++) {
          Anew += xstart(vcount) * (vxArray[i][j] * (vxArray[i][j]).t());
          vcount++;
          Anew += xstart(vcount) * (vyArray[i][j] * (vyArray[i][j]).t());
          vcount++;
          Anew += xstart(vcount) * (vzArray[i][j] * (vzArray[i][j]).t());
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


vec Variational::sweepDeterministicCMAES(size_t sweeps, size_t maxeval){
  vec results(sweeps), xstart(3*n*(n+1)/2);
  mat Anew(3*n,3*n);
  size_t index, vcount;
  double Ebest;
  vec** vxArray = vArrayList.at(0);
  vec** vyArray = vArrayList.at(1);
  vec** vzArray = vArrayList.at(2);

  function<double(vec&)> fitness = [&](vec& alpha){
    double Hij, Bij;
    mat Acurrent(3*n,3*n), Atrial = zeros<mat>(3*n,3*n);
    alpha = abs(alpha); // ensure only positive values
    size_t count = 0;

    for (size_t i = 0; i < n+1; i++) {
      for (size_t j = i+1; j < n+1; j++) {
        Atrial += alpha(count) * (vxArray[i][j] * (vxArray[i][j]).t());
        count++;
        Atrial += alpha(count) * (vyArray[i][j] * (vyArray[i][j]).t());
        count++;
        Atrial += alpha(count) * (vzArray[i][j] * (vzArray[i][j]).t());
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
          Anew += xstart(vcount) * (vxArray[i][j] * (vxArray[i][j]).t());
          vcount++;
          Anew += xstart(vcount) * (vyArray[i][j] * (vyArray[i][j]).t());
          vcount++;
          Anew += xstart(vcount) * (vzArray[i][j] * (vzArray[i][j]).t());
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


double Variational::addBasisFunctionCMAES(mat A_guess, vec S_guess, size_t state, size_t maxeval){
  /* Initialization */
  K++;
  H.resize(K,K);
  B.resize(K,K);

  function<double(vec&)> eval = [&](vec& x){
    assert(x.n_rows == 3*n*(3*n+1)/2 +3*n);

    double Hij, Bij;
    mat Acurrent(3*n,3*n);
    vec Scurrent(3*n);

    mat Atrial(3*n,3*n);
    vec xtemp = abs(x.rows(0,3*n*(3*n+1)/2-1));
    Utils::vec2symmetricMat(xtemp, Atrial);
    vec Strial = x.rows(3*n*(3*n+1)/2,3*n*(3*n+1)/2+3*n-1);

    for (size_t i = 0; i < K-1; i++) {
      Acurrent = basis.slice(i);
      Scurrent = shift.col(i);

      matElem.calculateH(Acurrent,Atrial,Scurrent,Strial,Hij,Bij);
      H(i,K-1) = Hij;
      H(K-1,i) = Hij;
      B(i,K-1) = Bij;
      B(K-1,i) = Bij;
    }
    matElem.calculateH(Atrial,Atrial,Strial,Strial,Hij,Bij);
    H(K-1,K-1) = Hij;
    B(K-1,K-1) = Bij;

    return groundStateEnergy();
  };

  vec x_guessA(3*n*(3*n+1)/2);
  Utils::symmetricMat2vec(x_guessA,A_guess);
  vec x_guess = join_vert(x_guessA,S_guess);
  x_guess = abs(x_guess%randn<vec>(3*n*(3*n+1)/2 +3*n)); // note: eig fails if basis functions too similar

  /* Setup CMAES parameters and run */
  size_t population = 200;
  size_t offsprings = 50;
  double distWidth = 1e-2;

  CMAES::optimize(eval, x_guess, population, offsprings, distWidth, maxeval);

  /* add result to basis */
  basis.resize(3*n,3*n,K);
  mat Atemp(3*n,3*n);
  vec xtemp = abs(x_guess.rows(0,3*n*(3*n+1)/2-1));
  Utils::vec2symmetricMat(xtemp,Atemp);
  basis.slice(K-1) = Atemp;

  shift.resize(3*n,K);
  shift.col(K-1) = x_guess.rows(3*n*(3*n+1)/2,3*n*(3*n+1)/2+3*n-1);

  return eval(x_guess);
}


void Variational::printBasis(){
  cout << "Current basis:" << endl << basis << endl;
}

void Variational::printShift(){
  cout << "Current shift:" << endl << shift << endl;
}
