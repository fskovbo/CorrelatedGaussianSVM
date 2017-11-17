#include <iostream>
#include "math.h"
#include "nlopt.hpp"

//
//  Function to be optimized
//
double myvfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    if (!grad.empty()) {
        grad[0] = 0.0;
        grad[1] = 0.5 / sqrt(x[1]);
    }
    return sqrt(x[1]);
}

//
//  Constraints function
//
typedef struct {
    double a, b;
} my_constraint_data;

double myvconstraint(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
    my_constraint_data *d = reinterpret_cast<my_constraint_data*>(data);
    double a = d->a, b = d->b;
    if (!grad.empty()) {
        grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
        grad[1] = -1.0;
    }
    return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
}

// static double wrap(const std::vector`<double>` &x, std::vector`<double>` &grad, void *data) {
//     return (*reinterpret_cast`<MyFunction*>`(data))(x, grad);
// }

int main() {

  //
  //  Specify lower bounds for variables
  //
  std::vector<double> lb(2);
  lb[0] = -HUGE_VAL; lb[1] = 0;

  //
  //  Set dimensionality for algorithm and set bounds
  //
  //nlopt::opt opt(nlopt::LD_MMA, 2); // LD = local & with derivative
  nlopt::opt opt(nlopt::LN_COBYLA, 2);
  opt.set_lower_bounds(lb);
  opt.set_min_objective(myvfunc, NULL);

  //
  //  Specify constraints
  //
  my_constraint_data data[2] = { {2,0}, {-1,1} }; // parametres of the contraint functions
  opt.add_inequality_constraint(myvconstraint, &data[0], 1e-8); //  1e-8 tolerance on constraints
  opt.add_inequality_constraint(myvconstraint, &data[1], 1e-8);
  opt.set_xtol_rel(1e-8); // tolerance on parametres

  //
  //  Set initial guess x
  //
  std::vector<double> x(2);
  x[0] = 1.234; x[1] = 5.678;
  double minf;

  //
  //  Perform optimization
  //
  nlopt::result result = opt.optimize(x, minf);

  for (auto& v : x){
    std::cout << v << std::endl;
  }
  std::cout << minf << std::endl;

  return 0;
}
