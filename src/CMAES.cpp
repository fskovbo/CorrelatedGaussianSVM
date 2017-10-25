#include "CMAES.h"

void CMAES::optimize(std::function<double(vec&)> fitness, vec& xmean, size_t lambda, size_t mu, double sigma, size_t maxeval){
  size_t N = xmean.n_rows;

  vec weights   = log(mu+0.5)-log(linspace<vec>(1, mu, mu));
  weights      /= sum(weights);
  vec tmp       = pow(weights,2);
  double mueff  = pow(sum(weights),2)/sum(tmp);

  double cc     = (4+mueff/N)/(N+4+2.0*mueff/N);
  double cs     = (mueff+2)/(N+mueff+5);
  double c1     = 2.0/(pow(N+1.3,2) +mueff);
  double cmu    = min(1-c1, 2.0*(mueff-2 +1.0/mueff)/(pow(N+2,2)+mueff));
  double damps  = 1 + 2.0*std::max(0.0,sqrt((mueff-1)/(N+1)) -1) +cs;

  vec pc  = zeros<vec>(N);
  vec ps  = zeros<vec>(N);
  mat B   = eye<mat>(N,N);
  vec D   = ones<vec>(N);
  mat C   = B*diagmat(pow(D,2))*B.t();
  mat invsqrtC = B*diagmat(pow(D,-1))*B.t();
  size_t eigeneval = 0;
  double chiN = pow(N,0.5) * (1-1.0/(4*N)+1.0/(21*pow(N,2)));

  size_t counteval = 0;
  mat arx(N,lambda);
  vec arfitness(lambda), xtrial(N), xold(N);
  uvec arindex(lambda);

  vec xbest = xmean;
  double fbest = fitness(xmean);//double fbest = 9999;

  while (counteval < maxeval) {

    for (size_t k = 0; k < lambda; k++) {
      xtrial = xmean + sigma*B*(D%randn<vec>(N));
      //arfitness(k) = fitness(xtrial);
      try {
        arfitness(k) = fitness(xtrial);
      }
      catch (...){
        xmean = xbest;
        return;
      }
      arx.col(k) = xtrial;
      counteval++;
    }

    arindex = sort_index(arfitness);
    xold = xmean;
    xmean = arx.cols(arindex.rows(0,mu-1))*weights;

    if (arfitness(arindex(0)) < fbest) {
      fbest = arfitness(arindex(0));
      xbest = arx.col(arindex(0));
    }

    ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold)/sigma;
    int hsig = norm(ps)/sqrt(1-pow((1-cs),2.0*counteval/lambda))/chiN < 1.4 + 2.0/(N+1);
    pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold)/sigma;

    mat artmp = (1.0/sigma) * (arx.cols(arindex.rows(0,mu-1))-repmat(xold,1,mu));
    C = (1-c1-cmu)*C + c1 * (dot(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu*artmp*diagmat(weights)*artmp.t();

    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));

    if (counteval-eigeneval > lambda/(c1+cmu)/N/10.0) {
      eigeneval = counteval;
      mat utri = trimatu(C);
      utri.diag().zeros();
      C = trimatu(C) + utri.t();
      eig_sym(D,B,C);
      D = sqrt(abs(D));

      invsqrtC = B * diagmat(pow(D,-1)) * B.t();
    }

    if (max(D) > 1e7 * min(D)) {
      //reset
      pc.zeros();
      ps.zeros();
      B.eye();
      D.ones();
      C = B*diagmat(pow(D,2))*B.t();
      invsqrtC = B*diagmat(pow(D,-1))*B.t();
      sigma = 0.01;
    }
  }

  if (fbest < fitness(xmean)) {
    xmean = xbest;
  }
}
