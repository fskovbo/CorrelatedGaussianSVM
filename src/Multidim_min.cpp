#include "Multidim_min.h"

void Multidim_min::reflect(vec& highest, vec& centroid, vec& reflected){
  reflected = 2*centroid - highest;
}

void Multidim_min::expand(vec& highest, vec& centroid, vec& expanded){
  expanded = 3*centroid - 2*highest;
}

void Multidim_min::contract(vec& highest, vec& centroid, vec& contracted){
  contracted = 0.5*centroid + 0.5*highest;
}

void Multidim_min::reduce(mat& simplex, size_t low){
  for (size_t i = 0; i < simplex.n_rows; i++) {
    if (i != low) {
      for (size_t j = 0; j < simplex.n_cols; j++) {
        simplex(i,j) = simplex(i,j)/2.0 + simplex(low,j)/2.0;
      }
    }
  }
}

void Multidim_min::SimplexUpdate(mat& simplex, vec& fval, size_t& high, size_t& low, vec& centroid){
  high = 0;
  low = 0;
  double highest = fval(0);
  double lowest = fval(0);

  for (size_t i = 0; i < simplex.n_rows; i++) {
    double next = fval(i);

    if (next > highest) {
      highest = next;
      high = i;
    }
    if (next < lowest) {
      lowest = next;
      low = i;
    }
  }

  for (size_t i = 0; i < simplex.n_cols; i++) {
    double s = 0;
    for (size_t j = 0; j < simplex.n_rows; j++) {
      if ( j != high) {
        s += simplex(j,i);
      }
    }
    centroid(i) = s/simplex.n_cols;
  }
}

void Multidim_min::SimplexInit(std::function<double(vec&)> f, mat& simplex, vec& fval, size_t& high, size_t& low, vec& centroid){
  for (size_t i = 0; i < simplex.n_rows; i++) {
    vec x = simplex.row(i).t();
    fval(i) = f(x);
  }
  SimplexUpdate(simplex,fval,high,low,centroid);
}

size_t Multidim_min::DownhillSimplex(std::function<double(vec&)> fitness, mat& simplex, double goalSize, size_t maxeval, vec& result){
  size_t high, low, steps = 0, n = simplex.n_cols;
  vec centroid(n), fval(n+1), p1(n), p2(n), highvec(n);

  SimplexInit(fitness,simplex,fval,high,low,centroid);

  while (dimSize(simplex,n) > goalSize && steps < maxeval) {
    SimplexUpdate(simplex,fval,high,low,centroid);
    highvec = simplex.row(high).t();
    reflect(highvec,centroid,p1);
    double f_re = fitness(p1);
    if (f_re < fval(low)) {
      highvec = simplex.row(high).t();
      expand(highvec,centroid,p2);
      double f_ex = fitness(p2);
      if (f_ex < f_re) {
        for (size_t i = 0; i < n; i++) {
          simplex(high,i) = p2(i);
          fval(high) = f_ex;
        }
      }
      else{
        for (size_t i = 0; i < n; i++) {
          simplex(high,i) = p1(i);
          fval(high) = f_re;
        }
      }
    }
    else{
      if (f_re < fval(high)) {
        for (size_t i = 0; i < n; i++) {
          simplex(high,i) = p1(i);
          fval(high) = f_re;
        }
      }
      else{
        highvec = simplex.row(high).t();
        contract(highvec,centroid,p1);
        double f_co = fitness(p1);
        if (f_co < fval(high)) {
          for (size_t i = 0; i < n; i++) {
            simplex(high,i) = p1(i);
            fval(high) = f_co;
          }
        }
        else{
          reduce(simplex,low);
          SimplexInit(fitness,simplex,fval,high,low,centroid);
        }
      }
    }
    steps++;
  }
  result = simplex.row(low).t();
  return steps;
}

double Multidim_min::dimSize(mat& simplex, size_t dim){
  double s = 0;
  for (size_t i = 1; i < dim+1; i++) {
    double d = 0;
    for (size_t k = 0; k < dim; k++) {
      d += pow(simplex(i,k) - simplex(0,k),2);
    }
    d = sqrt(d);
    if (d>s) s = d;
  }
  return s;
}

size_t Multidim_min::QuasiNewtonMin(std::function<double(vec&)> fitness, vec& xstart, double dx, double epsilon, size_t maxeval){
  size_t n = xstart.n_rows;
  vec z(n), Dx(n), s(n), xinit = xstart;
  double fz, fx = fitness(xstart), finit = fx;
  mat H1 = eye<mat>(n,n);
  vec ddx = NumGradient(fitness,xstart,dx);

  size_t iterations = 0;
  do {
    iterations++;
    Dx = -H1*ddx;
    s = 2*Dx;

    do {
      s *= 0.5;
      z = xstart + s;
      fz = fitness(z);
      //if( abs(fz) < abs(fx)+0.1*dot(s,ddx) ) break;
      if( fz < fx+0.1*dot(s,ddx) ) break;
      if( norm(s,2) < dx ) {H1.eye(); break;}
    } while(1);
    vec ddz = NumGradient(fitness,z,dx);
    vec y = ddz-ddx;
    H1 += ((s-H1*y)*s.t()*H1)/dot(y,H1*s);
    xstart = z; fx = fz; ddx = ddz;
  } while(norm(Dx,2) > dx && norm(ddx) > epsilon && iterations < maxeval);
  if (fx > finit) {
    xstart = xinit;
  }

  return iterations;
}

vec Multidim_min::NumGradient(std::function<double(vec&)> f, vec& x, double dx){
  size_t n = x.n_rows;
  vec ddx(n);
  double fx = f(x);
  for (size_t i = 0; i < n; i++) {
    x(i) += dx;
    ddx(i) = (f(x) - fx)/dx;
    x(i) -= dx;
  }
  return ddx;
}

void Multidim_min::Trial(std::function<double(vec&)> fitness, vec& xstart, size_t lambda){
  size_t N = xstart.n_rows;

  vec xbest = xstart;
  double fbest = fitness(xbest);

  vec testvec(N);
  double f;

  for (size_t i = 0; i < lambda; i++) {
    testvec = xbest + randn<vec>(N);
    f = fitness(testvec);

    if (f < fbest) {
      f = fbest;
      xbest = testvec;
    }
  }
  xstart = xbest;
}

// int GSLamoeba(double f(const gsl_vector* x, void* params), vec& xstart, double size_goal){
//   gsl_vector *xvec = gsl_vector_alloc(xstart.n_rows);
//   for (size_t i = 0; i < xstart.n_rows; i++) {
//     gsl_vector_set(xvec,i,xstart(i));
//   }
//
//   gsl_multimin_fminimizer *S = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2,x->size);
//   gsl_multimin_function F = {.f=f,.n=x->size,.params=NULL};
//   gsl_multimin_fminimizer_set(S, &F, x, ss);
//
//   int status,iter=0;
//   do{
//   	iter++;
//   	status = gsl_multimin_fminimizer_iterate(S);
//   	if (status) break;
//   	double size = gsl_multimin_fminimizer_size (S);
//   	//printf("iter=%i f() = %f simplex_size = %f\n", iter, S->fval, size);
//   	status = gsl_multimin_test_size (size, size_goal);
//   	if (status == GSL_SUCCESS) { printf ("converged to minimum at\n"); }
//   }while (status == GSL_CONTINUE && iter < 200);
//   return 0;
// }
