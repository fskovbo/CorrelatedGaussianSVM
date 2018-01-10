#ifndef MULTIDIM_MIN_H
#define MULTIDIM_MIN_H

#include <armadillo>
#include <functional>

using namespace arma;
using namespace std;

class Multidim_min {
private:
  static void SimplexInit(std::function<double(vec&)> f, mat& simplex, vec& fval, size_t& high, size_t& low, vec& centroid);
  static void SimplexUpdate(mat& simplex, vec& fval, size_t& high, size_t& low, vec& centroid);
  static void reflect(vec& highest, vec& centroid, vec& reflected);
  static void expand(vec& highest, vec& centroid, vec& expanded);
  static void contract(vec& highest, vec& centroid, vec& contracted);
  static void reduce(mat& simplex, size_t low);
  static double dimSize(mat& simplex, size_t dim);
  static vec NumGradient(std::function<double(vec&)> f, vec& x, double dx);

public:
  static size_t DownhillSimplex(std::function<double(vec&)> fitness, mat& simplex, double goalSize, size_t maxeval, vec& result);
  static size_t QuasiNewtonMin(std::function<double(vec&)> fitness, vec& xstart, double dx, double epsilon, size_t maxeval);
  static void Trial(std::function<double(vec&)> fitness, vec& xstart, size_t lambda);

  //static int GSLamoeba(double f(const gsl_vector* x, void* params), gsl_vector* x, gsl_vector* ss, double size_goal);
};

#endif
