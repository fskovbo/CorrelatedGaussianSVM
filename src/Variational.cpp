#include "Variational.h"

Variational::Variational(System& sys, MatrixElements& matElem)
: n(sys.n), De(sys.De), matElem(matElem), vArrayList(sys.vArrayList) {
  K = 0;
}


double Variational::eigenEnergy(size_t state){
  mat L(K,K);
  bool status = chol(L,B,"lower");
  if (status) {
    vec eigs = eig_sym( L.i()*H*(L.t()).i() );
    return eigs(state);
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
  shift = zeros<mat>(3*n,K);

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
  return eigenEnergy(0);
}

vec Variational::sweepStochastic(size_t state, size_t sweeps, size_t trials, vec Ameanval){
  mat Atrial, Acurrent;
  double Hij, Bij, Etrial;
  double Ebest = eigenEnergy(state);
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

        Etrial = eigenEnergy(state);
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

vec Variational::sweepStochasticShift(size_t state, size_t sweeps, size_t trials, vec Ameanval, vec maxShift){
  mat Atrial, Acurrent;
  vec strial, scurrent;
  double Hij, Bij, Etrial;
  double Ebest = eigenEnergy(state);
  mat Hbest = H, Bbest = B;
  vec trialCoeffs(De*n*(n+1)/2);
  vec results = Ebest*ones<vec>(sweeps+1);

  for (size_t l = 0; l < sweeps; l++) {
    for (size_t j = 0; j < K; j++) {
      for (size_t k = 0; k < trials; k++) {
        Atrial = generateRandomGaussian(Ameanval,trialCoeffs);
        strial = randn<vec>(3*n);
        vec stemp =  repmat(maxShift,n,1);
        strial = diagmat(stemp)*strial;

        for (size_t i = 0; i < K; i++) {

          if (j == i) {
            matElem.calculateH(Atrial,Atrial,strial,strial,Hij,Bij);
            H(i,i) = Hij;
            B(i,i) = Bij;
          }
          else{
            Acurrent = basis.slice(i);
            scurrent = shift.col(i);

            matElem.calculateH(Acurrent,Atrial,scurrent,strial,Hij,Bij);
            H(i,j) = Hij;
            H(j,i) = Hij;
            B(i,j) = Bij;
            B(j,i) = Bij;
          }

        }

        Etrial = eigenEnergy(state);
        if (Etrial < Ebest) { //if trial is better: update basis
          Ebest = Etrial;
          Hbest = H;
          Bbest = B;
          basis.slice(j) = Atrial;
          basisCoefficients.col(j) = trialCoeffs;
          shift.col(j) = strial;
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

vec Variational::sweepDeterministic(size_t state, size_t sweeps, size_t Nunique, vec uniquePar){
  size_t Npar = Nunique*n*(n+1)/2;
  vec results(sweeps), xstart(Npar);
  vec** vArray;

  //
  //  NLOpt setup
  //
  std::vector<double> lb(Npar);
  std::vector<double> xs(Npar);
  std::vector<mat> dummy;
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
      for (size_t i = 0; i < n*(n+1)/2; i++) {
        for (size_t k = 0; k < De; k++) {
          xs[Nunique*i+uniquePar(k)] = xstart(De*i+k);
        }
      }
      my_function_data data = { index,n,K,De,Nunique,state,uniquePar,vArrayList,H,B,dummy,dummy,basis,matElem };
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

      for (size_t i = 0; i < n*(n+1)/2; i++) {
        for (size_t k = 0; k < De; k++) {
          xstart(De*i+k) = xs[Nunique*i+uniquePar(k)];
        }
      }

      // ----------------------------------------- //
      double Hij, Bij;
      mat Acurrent(De*n,De*n), Anew = zeros<mat>(De*n,De*n);
      size_t count = 0;
      for (size_t i = 0; i < n+1; i++) {
        for (size_t j = i+1; j < n+1; j++) {
          for (size_t k = 0; k < De; k++) {
            vArray = vArrayList.at(k);
            Anew += xs[Nunique*count+uniquePar(k)] * (vArray[i][j] * (vArray[i][j]).t());
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

vec Variational::sweepDeterministicShift(size_t state, size_t sweeps, vec maxShift, size_t Nunique, vec uniquePar){
  size_t NparA = Nunique*n*(n+1)/2;
  size_t NparS = 3*n;
  size_t Npar  = NparA + NparS;
  vec results(sweeps), xstart(NparA);
  vec** vArray;

  //
  //  NLOpt setup
  //
  std::vector<double> lb(Npar);
  std::vector<double> ub(Npar);
  std::vector<double> xs(Npar);
  for (size_t i = 0; i < NparA; i++) {
    lb[i] = 1e-6;
    ub[i] = HUGE_VAL;
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < 3; k++) {
      lb[NparA + 3*i+k] = -3.0*maxShift(k);
      ub[NparA + 3*i+k] = 3.0*maxShift(k);
    }
  }
  nlopt::opt opt(nlopt::LN_NELDERMEAD, Npar);
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_xtol_abs(1e-10); // tolerance on parametres
  double minf;

  for (size_t l = 0; l < sweeps; l++) {
    for (size_t index = 0; index < K; index++) {
      //
      // optimize basis function index using its current values as starting guess
      //

      xstart = basisCoefficients.col(index);
      for (size_t i = 0; i < n*(n+1)/2; i++) {
        for (size_t k = 0; k < De; k++) {
          xs[Nunique*i+uniquePar(k)] = xstart(De*i+k);
        }
      }
      for (size_t i = 0; i < NparS; i++) {
        xs[i+NparA] = shift(i,index);
      }

      my_function_data_shift data = { index,n,K,De,Nunique,state,uniquePar,vArrayList,H,B,basis,shift,matElem };
      opt.set_min_objective(myvfunc_shift, &data);

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

      // ----------------------------------------- //
      double Hij, Bij;
      mat Acurrent(De*n,De*n), Anew = zeros<mat>(De*n,De*n);
      vec scurrent(3*n), snew(3*n);
      size_t count = 0;
      for (size_t i = 0; i < n+1; i++) {
        for (size_t j = i+1; j < n+1; j++) {
          for (size_t k = 0; k < De; k++) {
            vArray = vArrayList.at(k);
            Anew += xs[Nunique*count+uniquePar(k)] * (vArray[i][j] * (vArray[i][j]).t());
          }
          count++;
        }
      }
      for (size_t i = 0; i < NparS; i++) {
        snew(i) = xs[NparA+i];
      }

      for (size_t j = 0; j < K; j++) {
        if (j == index) {
          matElem.calculateH(Anew,Anew,snew,snew,Hij,Bij);
          H(index,index) = Hij;
          B(index,index) = Bij;
        } else {
          Acurrent = basis.slice(j);
          scurrent = shift.col(j);

          matElem.calculateH(Acurrent,Anew,scurrent,snew,Hij,Bij);
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
      vec Acoeffs(De*n*(n+1)/2);
      for (size_t i = 0; i < n*(n+1)/2; i++) {
        for (size_t k = 0; k < De; k++) {
          Acoeffs(De*i+k) = xs[Nunique*i+uniquePar(k)];
        }
      }

      basis.slice(index) = Anew;
      basisCoefficients.col(index) = Acoeffs;
      shift.col(index) = snew;
    }
    cout << "Energy after deterministic sweep " << l+1 << ": " << minf << "\n";
    results(l) = minf;
  }
  return results;
}

vec Variational::sweepDeterministicCMAES(size_t state, size_t sweeps, size_t maxeval){
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

    return eigenEnergy(state);
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

vec Variational::addBasisFunction(size_t state, size_t tries, vec startGuess, vec maxShift, size_t Nunique, vec uniquePar){
  size_t NparA = Nunique*n*(n+1)/2;
  size_t NparS = 3*n;
  size_t Npar  = NparA + NparS;
  vec xstart(De*n*(n+1)/2), results(state+1);
  vec** vArray;

  K++;
  basis.resize(De*n,De*n,K);
  basisCoefficients.resize(De*n*(n+1)/2,K);
  shift.resize(3*n,K);
  H.resize(K,K);
  B.resize(K,K);
  //
  //  NLOpt setup
  //
  std::vector<double> lb(Npar);
  std::vector<double> ub(Npar);
  std::vector<double> xs(Npar);
  for (size_t i = 0; i < NparA; i++) {
    lb[i] = 1e-6;
    ub[i] = HUGE_VAL;
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < 3; k++) {
      lb[NparA + 3*i+k] = -3.0*maxShift(k);
      ub[NparA + 3*i+k] = 3.0*maxShift(k);
    }
  }
  nlopt::opt opt(nlopt::LN_NELDERMEAD, Npar);
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_xtol_abs(1e-10); // tolerance on parametres
  double minf;

  for (size_t l = 0; l < tries; l++) {
    generateRandomGaussian(startGuess,xstart);
    for (size_t i = 0; i < n*(n+1)/2; i++) {
      for (size_t k = 0; k < De; k++) {
        xs[Nunique*i+uniquePar(k)] = xstart(De*i+k);
      }
    }
    for (size_t i = 0; i < NparS; i++) {
      xs[i+NparA] = shift(i,K-2);
    }

    my_function_data_shift data = { K-1,n,K,De,Nunique,state,uniquePar,vArrayList,H,B,basis,shift,matElem };
    opt.set_min_objective(myvfunc_shift, &data);

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

    // ----------------------------------------- //
    double Hij, Bij;
    mat Acurrent(De*n,De*n), Anew = zeros<mat>(De*n,De*n);
    vec scurrent(3*n), snew(3*n);
    size_t count = 0;
    for (size_t i = 0; i < n+1; i++) {
      for (size_t j = i+1; j < n+1; j++) {
        for (size_t k = 0; k < De; k++) {
          vArray = vArrayList.at(k);
          Anew += xs[Nunique*count+uniquePar(k)] * (vArray[i][j] * (vArray[i][j]).t());
        }
        count++;
      }
    }
    for (size_t i = 0; i < NparS; i++) {
      snew(i) = xs[NparA+i];
    }

    for (size_t j = 0; j < K-1; j++) {
      Acurrent = basis.slice(j);
      scurrent = shift.col(j);

      matElem.calculateH(Acurrent,Anew,scurrent,snew,Hij,Bij);
      H(j,K-1) = Hij;
      H(K-1,j) = Hij;
      B(j,K-1) = Bij;
      B(K-1,j) = Bij;
    }
    matElem.calculateH(Anew,Anew,snew,snew,Hij,Bij);
    H(K-1,K-1) = Hij;
    B(K-1,K-1) = Bij;

    // ----------------------------------------- //
    //
    // add optimized basis function to basis
    //
    vec Acoeffs(De*n*(n+1)/2);
    for (size_t i = 0; i < n*(n+1)/2; i++) {
      for (size_t k = 0; k < De; k++) {
        Acoeffs(De*i+k) = xs[Nunique*i+uniquePar(k)];
      }
    }

    basis.slice(K-1) = Anew;
    basisCoefficients.col(K-1) = Acoeffs;
    shift.col(K-1) = snew;
  }
  for (size_t i = 0; i <= state; i++) {
    results(i) = eigenEnergy(i);
  }
  cout << "Energy after adding basis function " << K << ":\n" << results << "\n";
  return results;
}


void Variational::printBasis(){
  cout << "Current basis:" << endl << basis << endl;
}

void Variational::printShift(){
  cout << "Current shift:" << endl << shift << endl;
}




// -------------------------------------------------------------------------------------------------- //
//
//
//      Gradient based methods below. Does not appear to work on this problem.
//
//
// -------------------------------------------------------------------------------------------------- //


double Variational::myvfunc_grad(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_function_data *d = reinterpret_cast<my_function_data*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De, Nunique = d->Nunique;
  vec& uniquePar = d->uniquePar;
  vector<vec**>& vArrayList = d->vArrayList;
  mat H = d->H, B = d->B;
  vector<mat> HG = d->HG, BG = d->BG;
  cube& basis = d->basis;
  MatrixElements& matElem = d->matElem;

  double Hij, Bij;
  vec Hgradij, Bgradij;
  mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
  size_t count = 0;
  vec** vArray;

  for (size_t i = 0; i < n+1; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        Atrial += x[Nunique*count+uniquePar(k)] * (vArray[i][j] * (vArray[i][j]).t());
      }
      count++;
    }
  }

  for (size_t j = 0; j < K; j++) {
    if (j == index) {
      matElem.calculateH_noShift(Atrial,Atrial,Hij,Bij,Hgradij,Bgradij);
      count = 0;
      for (auto iH = begin(HG), iB = begin(BG), e = end(HG); iH != e; ++iH, ++iB){
        (*iH)(index,index) = 2*Hgradij(count);
        (*iB)(index,index) = 2*Bgradij(count);
        count++;
      }
      H(index,index) = Hij;
      B(index,index) = Bij;
    } else {
      Acurrent = basis.slice(j);
      cout << "WALLAH" << endl;

      matElem.calculateH_noShift(Acurrent,Atrial,Hij,Bij,Hgradij,Bgradij);
      count = 0;
      for (auto iH = begin(HG), iB = begin(BG), e = end(HG); iH != e; ++iH, ++iB){
        (*iH)(j,index) = Hgradij(count);
        (*iH)(index,j) = Hgradij(count);
        (*iB)(j,index) = Bgradij(count);
        (*iB)(index,j) = Bgradij(count);
        count++;
      }
      H(j,index) = Hij;
      H(index,j) = Hij;
      B(j,index) = Bij;
      B(index,j) = Bij;
    }
  }

  mat L(K,K);
  vec eigval;
  mat eigvec;
  bool status = chol(L,B,"lower");
  if (status) {
    eig_sym(eigval,eigvec, L.i()*H*(L.t()).i() );
  }
  else{
    eigval = 9999*1e10*ones<vec>(K);
  }

  count = 0;
  for (auto iH = begin(HG), iB = begin(BG), e = end(HG); iH != e; ++iH, ++iB){
    grad.at(count) = dot(eigvec.col(0), ((*iH)-eigval(0)*(*iB)) * eigvec.col(0));
    count++;
  }
  return eigval(0);
}

double Variational::myvfunc_grad_test(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  my_function_data *d = reinterpret_cast<my_function_data*>(data);
  size_t index = d->index, n = d->n, K = d->K, De = d->De, Nunique = d->Nunique;
  vec& uniquePar = d->uniquePar;
  vector<vec**>& vArrayList = d->vArrayList;
  mat H = d->H, B = d->B;
  vector<mat> HG = d->HG, BG = d->BG;
  cube& basis = d->basis;
  MatrixElements& matElem = d->matElem;

  double Hii, Bii;
  vec Hgradii, Bgradii;
  mat A = zeros<mat>(De*n,De*n);
  size_t count = 0;
  vec** vArray;

  for (size_t i = 0; i < n+1; i++) {
    for (size_t j = i+1; j < n+1; j++) {
      for (size_t k = 0; k < De; k++) {
        vArray = vArrayList.at(k);
        A += x[count+k] * (vArray[i][j] * (vArray[i][j]).t());
      }
      count++;
    }
  }

  matElem.calculateH_noShift(A,A,Hii,Bii,Hgradii,Bgradii);
  for (size_t i = 0; i < Hgradii.n_rows; i++) {
    (HG[i])(0,0) = 2*Hgradii(i);
    (BG[i])(0,0) = 2*Bgradii(i);
  }
  H(0,0) = Hii;
  B(0,0) = Bii;

  mat L(K,K);
  vec eigval;
  mat eigvec;
  bool status = chol(L,B,"lower");
  if (status) {
    eig_sym(eigval,eigvec, L.i()*H*(L.t()).i() );
  }
  else{
    eigval = 9999*1e10*ones<vec>(K);
  }

  if (!grad.empty()){
    for (size_t i = 0; i < HG.size(); i++) {
      grad[i] = dot(eigvec.col(0), ((HG[i])-eigval(0)*(BG[i])) * eigvec.col(0))/dot(eigvec.col(0), B*eigvec.col(0));
    }
  }
  // for (size_t i = 0; i < HG.size(); i++) {
  //   cout << HG[i] << endl;
  //   grad[i] = dot(eigvec.col(0), ((HG[i])-eigval(0)*(BG[i])) * eigvec.col(0));
  // }
  cout << eigval(0) << endl;
  return eigval(0);
}

vec Variational::sweepDeterministic_grad_test(){
  size_t Npar = n*(n+1)/2;
  vec xstart(Npar);

  //
  //  Set initial matrix gradients
  //
  std::vector<mat> HG(Npar);
  std::vector<mat> BG(Npar);
  for (size_t i = 0; i < Npar; i++) {
    HG.at(i) = (mat(K,K));
    BG.at(i) = (mat(K,K));
  }

  //
  //  NLOpt setup
  //
  std::vector<double> lb(Npar);
  std::vector<double> xs(Npar);
  for (size_t i = 0; i < Npar; i++) {
    lb[i] = 1e-6;
  }
  nlopt::opt opt(nlopt::LD_SLSQP, Npar);
  opt.set_lower_bounds(lb);
  opt.set_ftol_abs(1e-8); // tolerance on parametres
  double minf;

  xstart = basisCoefficients.col(0);
  for (size_t i = 0; i < Npar; i++) {
    xs[i] = xstart(i);
  }

  size_t Nunique = 1000 , index = 1000, state = 0; vec uniquePar = {6, 6, 6, 6};
  my_function_data data = { index,n,K,De,Nunique,state,uniquePar,vArrayList,H,B,HG,BG,basis,matElem };
  opt.set_min_objective(myvfunc_grad_test, &data);
  nlopt::result optresult = opt.optimize(xs, minf);

  basis.slice(0) = {xs[0]};

  vec result = {minf};

  return result;
}

vec Variational::sweepDeterministic_grad(size_t sweeps, size_t Nunique, vec uniquePar){
  size_t Npar = Nunique*n*(n+1)/2;
  vec results(sweeps), xstart(Npar);
  vec** vArray;

  //
  //  Set initial matrix gradients
  //
  std::vector<mat> HG(Npar);
  std::vector<mat> BG(Npar);
  for (size_t i = 0; i < Npar; i++) {
    HG.at(i) = (mat(K,K));
    BG.at(i) = (mat(K,K));
  }
  for (size_t i = 0; i < K; i++) {
    mat Ai = basis.slice(i);
    for (size_t j = i; j < K; j++) {
      mat Aj = basis.slice(j);
      double Hij, Bij;
      vec Hgradij, Bgradij;

      matElem.calculateH_noShift(Ai,Aj,Hij,Bij,Hgradij,Bgradij);
      if (j == i) {
        Hgradij *= 2;
        Bgradij *= 2;
      }
      size_t count = 0;
      for (auto iH = begin(HG), iB = begin(BG), e = end(HG); iH != e; ++iH, ++iB){
        (*iH)(i,j) = Hgradij(count);
        (*iH)(j,i) = Hgradij(count);
        (*iB)(i,j) = Bgradij(count);
        (*iB)(j,i) = Bgradij(count);
        count++;
      }
    }
  }

  //
  //  NLOpt setup
  //
  std::vector<double> lb(Npar);
  std::vector<double> xs(Npar);
  for (size_t i = 0; i < Npar; i++) {
    lb[i] = 1e-6;
  }
  nlopt::opt opt(nlopt::LD_SLSQP, Npar);
  opt.set_lower_bounds(lb);
  opt.set_xtol_abs(1e-10); // tolerance on parametres
  double minf;

  for (size_t l = 0; l < sweeps; l++) {
    for (size_t index = 0; index < K; index++) {
      cout << "index " << index << endl;
      //
      // optimize basis function index using its current values as starting guess
      //
      xstart = basisCoefficients.col(index);
      for (size_t i = 0; i < n*(n+1)/2; i++) {
        for (size_t k = 0; k < De; k++) {
          xs[Nunique*i+uniquePar(k)] = xstart(De*i+k);
        }
      }
      size_t state = 0;
      my_function_data data = { index,n,K,De,Nunique,state,uniquePar,vArrayList,H,B,HG,BG,basis,matElem };
      opt.set_min_objective(myvfunc_grad, &data);

      for (auto& lol : HG){
        cout << lol << endl;
      }
      for (auto& lol : xs){
        cout << lol << endl;
      }
      cout << basis << endl;

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

      for (auto& lol : HG){
        cout << lol << endl;
      }
      for (auto& lol : xs){
        cout << lol << endl;
      }
      auto TEMP = HG;

      for (size_t i = 0; i < n*(n+1)/2; i++) {
        for (size_t k = 0; k < De; k++) {
          xstart(De*i+k) = xs[Nunique*i+uniquePar(k)];
        }
      }

      // ----------------------------------------- //
      double Hij, Bij;
      mat Acurrent(De*n,De*n), Anew = zeros<mat>(De*n,De*n);
      size_t count = 0;
      for (size_t i = 0; i < n+1; i++) {
        for (size_t j = i+1; j < n+1; j++) {
          for (size_t k = 0; k < De; k++) {
            vArray = vArrayList.at(k);
            Anew += xs[Nunique*count+uniquePar(k)] * (vArray[i][j] * (vArray[i][j]).t());
          }
          count++;
        }
      }

      cout << Anew << endl;

      vec Hgradij, Bgradij;
      for (size_t j = 0; j < K; j++) {
        if (j == index) {
          matElem.calculateH_noShift(Anew,Anew,Hij,Bij,Hgradij,Bgradij);
          count = 0;
          for (auto iH = begin(HG), iB = begin(BG), e = end(HG); iH != e; ++iH, ++iB){
            (*iH)(index,index) = 2*Hgradij(count);
            (*iB)(index,index) = 2*Bgradij(count);
            count++;
          }
          H(index,index) = Hij;
          B(index,index) = Bij;
        } else {
          Acurrent = basis.slice(j);
          cout << "WALLAH" << endl;

          matElem.calculateH_noShift(Acurrent,Anew,Hij,Bij,Hgradij,Bgradij);
          count = 0;
          for (auto iH = begin(HG), iB = begin(BG), e = end(HG); iH != e; ++iH, ++iB){
            (*iH)(index,j) = Hgradij(count);
            (*iH)(j,index) = Hgradij(count);
            (*iB)(index,j) = Bgradij(count);
            (*iB)(j,index) = Bgradij(count);
            count++;
          }
          H(j,index) = Hij;
          H(index,j) = Hij;
          B(j,index) = Bij;
          B(index,j) = Bij;
        }
      }
      size_t tempcount = 0;
      for (auto& lol : HG){
        cout << lol << endl;
        cout << lol-TEMP[tempcount] << endl;
        tempcount++;
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
