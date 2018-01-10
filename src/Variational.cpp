#include "Variational.h"

Variational::Variational(System& sys, MatrixElements& matElem)
: n(sys.n), De(sys.De), matElem(matElem), vArrayList(sys.vArrayList), vList(sys.vList) {
  K = 0;

  Nunique   = De;
  uniquePar = linspace<vec>(0,De-1,De);
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

void Variational::setUniqueCoordinates(size_t Nunique_, vec uniquePar_){
  Nunique   = Nunique_;
  uniquePar = uniquePar_;
}

mat Variational::generateRandomGaussian(vec& Ameanval, vector<double>& coeffs){
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
        coeffs[De*count+k] = alpha(count);
      }
      count++;
    }
  }
  return A;
}

mat Variational::updateMatrices(vector<double> x, size_t index, bool shifted, vec& snew){
  size_t count = 0, NparS = 3*n;
  double Hij, Bij;
  mat Acurrent(De*n,De*n), Anew = zeros<mat>(De*n,De*n);
  vec scurrent(NparS);
  snew = zeros<vec>(NparS);

  for (auto& w : vList){
    Anew += x[count] *w*w.t();
    count++;
  }
  if (shifted) {
    for (size_t i = 0; i < NparS; i++) {
      snew(i) = x[count+i];
    }
  }

  for (size_t j = 0; j < K; j++) {
    if (j == index) {
      if (shifted) { matElem.calculateH(Anew,Anew,snew,snew,Hij,Bij); }
      else         { matElem.calculateH_noShift(Anew,Anew,Hij,Bij);   }
      H(index,index) = Hij;
      B(index,index) = Bij;

    } else {
      Acurrent = basis.slice(j);
      scurrent = shift.col(j);

      if (shifted) { matElem.calculateH(Acurrent,Anew,scurrent,snew,Hij,Bij); }
      else         { matElem.calculateH_noShift(Acurrent,Anew,Hij,Bij);       }
      H(j,index) = Hij;
      H(index,j) = Hij;
      B(j,index) = Bij;
      B(index,j) = Bij;
    }
  }
  return Anew;
}

double Variational::initializeBasis(size_t basisSize){
  K = basisSize;
  H.resize(K,K);
  B.resize(K,K);
  basis.set_size(De*n,De*n,K);
  for (size_t i = 0; i < K; i++) {
    basisCoefficients.emplace_back(vector<double>(De*n*(n+1)/2));
  }
  shift = zeros<mat>(3*n,K);

  vector<double> coeffs(De*n*(n+1)/2);
  vec startingGuess = 10*ones<vec>(De);

  for (size_t i = 0; i < K; i++) {
    basis.slice(i) = generateRandomGaussian(startingGuess,coeffs);
    basisCoefficients[i] = coeffs;
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
  vector<double> trialCoeffs(De*n*(n+1)/2);
  vec results = Ebest*ones<vec>(sweeps+1);

  vector<double> runtimedata;
  runtimedata.push_back(Ebest);

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
          basisCoefficients[j] = trialCoeffs;
        }
        else{ //otherwise revert to old H,B
          H = Hbest;
          B = Bbest;
        }
        runtimedata.push_back(Ebest);
      }
    }
    cout << "Energy after stochastic sweep " << l+1 << ": " << Ebest << "\n";
    results(l+1) = Ebest;
  }
  vec datavec(runtimedata.size());
  size_t lol = 0;
  for (auto& val: runtimedata){
    datavec(lol++) = val;
  }
  return datavec;
}

vec Variational::sweepStochasticShift(size_t state, size_t sweeps, size_t trials, vec Ameanval, vec maxShift){
  mat Atrial, Acurrent;
  vec strial, scurrent;
  double Hij, Bij, Etrial;
  double Ebest = eigenEnergy(state);
  mat Hbest = H, Bbest = B;
  vector<double> trialCoeffs(De*n*(n+1)/2);
  vec results = Ebest*ones<vec>(sweeps+1);

  for (size_t l = 0; l < sweeps; l++) {
    for (size_t j = 0; j < K; j++) {
      for (size_t k = 0; k < trials; k++) {
        Atrial = generateRandomGaussian(Ameanval,trialCoeffs);
        strial = randu<vec>(3*n);
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
          basisCoefficients[j] = trialCoeffs;
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

vec Variational::sweepDeterministicCMAES(size_t state, size_t sweeps, size_t maxeval){
  vec results(sweeps);
  vector<double> xstart(De*n*(n+1)/2);
  mat Anew(3*n,3*n);
  size_t index, vcount;
  double Ebest = eigenEnergy(state);
  vec** vArray;

  vector<double> runtimedata;
  runtimedata.push_back(Ebest);

  function<double(vec&)> fitness = [&](vec& alpha){
    double Hij, Bij;
    mat Acurrent(De*n,De*n), Atrial = zeros<mat>(De*n,De*n);
    size_t count = 0;
    alpha = abs(alpha);

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
    double Eval = eigenEnergy(state);
    if (Eval < runtimedata.back()) {
      runtimedata.push_back(Eval);
    }
    else{
      runtimedata.push_back(runtimedata.back());
    }

    return Eval;
  };

  for (size_t l = 0; l < sweeps; l++) {
    for (index = 0; index < K; index++) {
      //
      // optimize basis function index using its current values as starting guess
      //
      xstart = basisCoefficients[index];
      vec xs(xstart.size());
      for (size_t i = 0; i < xstart.size(); i++) {
        xs(i) = xstart[i];
      }
      CMAES::optimize(fitness, xs, 100, 50, 1e-1, maxeval);
      Ebest = fitness(xs); // needed to set H,B corresponding to optimized parameters
      for (size_t i = 0; i < xstart.size(); i++) {
        xstart[i] = xs(i);
      }

      //
      // add optimized basis function to basis
      //
      vcount = 0;
      Anew.zeros();
      for (size_t i = 0; i < n+1; i++) {
        for (size_t j = i+1; j < n+1; j++) {
          for (size_t k = 0; k < De; k++) {
            vArray = vArrayList.at(k);
            Anew += xstart[De*vcount+k] * (vArray[i][j] * (vArray[i][j]).t());
          }
          vcount++;
        }
      }
      basis.slice(index) = Anew;
      basisCoefficients[index] = xstart;
    }
    cout << "Energy after deterministic sweep " << l+1 << ": " << Ebest << "\n";
    results(l) = Ebest;
  }
  vec datavec(runtimedata.size());
  size_t lol = 0;
  for (auto& val: runtimedata){
    datavec(lol++) = val;
  }
  return datavec;
}

vec Variational::sweepDeterministic_grad(size_t state, size_t sweeps){
  size_t Npar = De*n*(n+1)/2;
  vec results(sweeps);
  vec** vArray;
  std::vector<double> lb(Npar,1e-6);
  std::vector<double> xs(Npar);
  nlopt::opt opt(nlopt::LD_MMA, Npar); //<--- sets optimization algorithm
  opt.set_lower_bounds(lb);
  opt.set_xtol_abs(1e-6); // tolerance on parametres
  double minf;

  double Ebest = eigenEnergy(state);
  vector<double> runtimedata;
  runtimedata.push_back(Ebest);

  for (size_t l = 0; l < sweeps; l++) {
    for (size_t index = 0; index < K; index++) {
      xs = basisCoefficients[index];
      my_function_data data = { index,n,K,De,De,state,results,vArrayList,vList,H,B,basis,matElem,runtimedata};
      opt.set_min_objective(myvfunc_grad, &data);

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
      vec dummy;
      basis.slice(index) = updateMatrices(xs,index,false,dummy);
      basisCoefficients[index] = xs;
    }
    cout << "Energy after deterministic sweep " << l+1 << ": " << minf << "\n";
    results(l) = minf;
  }
  vec datavec(runtimedata.size());
  size_t lol = 0;
  for (auto& val: runtimedata){
    datavec(lol++) = val;
  }
  return datavec;
}

vec Variational::fullBasisSearch(size_t state){
  size_t Npar = K*De*n*(n+1)/2;

  double Ebest = eigenEnergy(state);
  vector<double> runtimedata;
  runtimedata.push_back(Ebest);

  //
  //  NLOpt setup
  //
  std::vector<double> lb(Npar,1e-6);
  std::vector<double> xs;
  nlopt::opt opt(nlopt::LN_NELDERMEAD, Npar);
  opt.set_lower_bounds(lb);
  opt.set_xtol_abs(1e-10); // tolerance on parametres
  double minf;

  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < De*n*(n+1)/2; j++) {
      xs.push_back(basisCoefficients[i][j]);
    }
  }


  global_data data = { n,K,De,Npar,state,vList,matElem,runtimedata };
  opt.set_min_objective(globalvfunc, &data);
  bool status = false;
  size_t attempts = 0;
  while (!status && attempts < 5) {
    try{
      nlopt::result optresult = opt.optimize(xs, minf);
      status = true;
    }
    catch (const std::exception& e) {
      cout << "Optimization failed at attempt " << attempts << endl;
      attempts++;
    }
  }
  vec datavec(runtimedata.size());
  size_t lol = 0;
  for (auto& val: runtimedata){
    datavec(lol++) = val;
  }
  return datavec;
}

vec Variational::sweepDeterministic(size_t state, size_t sweeps, vec shiftBounds){
  size_t NparA = Nunique*n*(n+1)/2;
  size_t NparS = 3*n;
  size_t Npar  = NparA + NparS;
  vec results(sweeps), snew;
  bool shifted = !all(shiftBounds == 0);

  //
  //  NLOpt setup
  //
  std::vector<double> lb(Npar);
  std::vector<double> ub(Npar);
  std::vector<double> xs(Npar);
  std::vector<double> Acoeff(De*n*(n+1)/2);
  std::vector<double> fullcoeff(De*n*(n+1)/2 + NparS);

  for (size_t i = 0; i < NparA; i++) {
    lb[i] = 1e-6;
    ub[i] = HUGE_VAL;
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < 3; k++) {
      lb[NparA + 3*i+k] = -shiftBounds(k);
      ub[NparA + 3*i+k] = shiftBounds(k);
    }
  }
  nlopt::opt opt(nlopt::LN_SBPLX, Npar);
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_xtol_abs(1e-10); // tolerance on parametres
  double minf;

  for (size_t l = 0; l < sweeps; l++) {
    for (size_t index = 0; index < K; index++) {

      Acoeff = basisCoefficients[index];
      for (size_t i = 0; i < n*(n+1)/2; i++) {
        for (size_t k = 0; k < De; k++) {
          xs[Nunique*i+uniquePar(k)] = Acoeff[De*i+k];
        }
      }
      for (size_t i = 0; i < NparS; i++) {
        xs[i+NparA] = shift(i,index);
      }

      function_data data = { index,n,K,De,Nunique,state,uniquePar,vList,H,B,basis,shift,matElem,shifted };
      opt.set_min_objective(fitness, &data);

      try{ nlopt::result optresult = opt.optimize(xs, minf); }
      catch (const std::exception& e) { }

      //
      // add optimized basis function to basis
      //
      for (size_t i = 0; i < n*(n+1)/2; i++) {
        for (size_t k = 0; k < De; k++) {
          Acoeff[De*i+k]      = xs[Nunique*i+uniquePar(k)];
          fullcoeff[De*i+k]   = xs[Nunique*i+uniquePar(k)];
        }
      }
      for (size_t i = 0; i < NparS; i++) {
        fullcoeff[De*n*(n+1)/2 + i] = xs[NparA + i];
      }

      basis.slice(index)        = updateMatrices(fullcoeff,index,shifted,snew);
      basisCoefficients[index]  = Acoeff;
      shift.col(index)          = snew;
    }
    cout << "Energy after deterministic sweep " << l+1 << ": " << minf << "\n";
    results(l) = minf;
  }
  return results;
}


void Variational::printBasis(){
  cout << "Current basis:" << endl << basis << endl;
}

void Variational::printShift(){
  cout << "Current shift:" << endl << shift << endl;
}
