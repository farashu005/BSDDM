#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <mlpack.h>
#include <omp.h>

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo, RcppDist, RcppEnsmallen, mlpack)]]



using namespace arma;
using namespace Rcpp;


#define g(x, Ind_1, Ind_2 ) (x)( ((Ind_1)(Ind_2)) )
#define h(x, Ind) (x)((Ind))



/////////////////////////////////////////////////////////////////////////// General Functions ////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////// log density ////////////////////////////////////////////////////////////////////////////////////



inline double AccuLog_mod(const vec &x){
  return (x.n_elem) ? mlpack::AccuLog(x):0;
}



vec compute_log_add(const vec& X, const vec& Y) {
  vec A(X.n_elem);
  for (size_t i = 0; i < X.n_elem; ++i) {
    A(i) = mlpack::LogAdd(X(i), Y(i));
  }
  return A;
}


vec compute_log_add2(const double X, vec Y) {
  // vec A(Y.n_elem);
  // for (size_t i = 0; i < Y.n_elem; ++i) {
  //   A(i) = mlpack::LogAdd(X, Y(i));
  //
  //
  // }

  Y.transform( [&X](double val) { return (mlpack::LogAdd(X, val)); } );
  return Y;
}



inline double EXPITE(double x) {
   if (x >= 0.0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);   // case x >= 0
    } else {
        double z = std::exp(x);
        return z / (1.0 + z);     // case x < 0
    }
}



inline double EXPITE_sq(double x) {
  double s = EXPITE(x);
    return s * (1.0 - s);   // numerically stable derivative
}


// log density (Go Trial)


vec log_dens(const double sigma,const vec &tau, const vec &b, const vec &nu){

  // Rcpp::Rcout<<"A"<< endl;

vec result(tau.n_elem);
result.fill(trunc_log(0.0));  // if trunc_log(0.0) = -1e10 or similar

// Logical index for tau_s > 0
uvec valid_idx = find(tau > 0);


if (!valid_idx.is_empty()) {
  vec tau_valid = tau.elem(valid_idx);

  vec b_valid = b.elem(valid_idx);

  vec nu_valid = nu.elem(valid_idx);

  vec partial_result =(b_valid -  ((log(2 * datum::pi*sigma))/2)
                - ((3 * log(tau_valid)) / 2) - ((exp(2 * b_valid)) / (2 * sigma*tau_valid)) +
                  (((exp(nu_valid+b_valid)))/(sigma))-(((exp(2*nu_valid))%tau_valid)/(2*sigma)));

  result.elem(valid_idx) = partial_result;
  }

  // Rcpp::Rcout<<"result"<<result<<endl;

  if(!result.is_finite()){
    cout << " tau:" << tau.t() << std::endl;
    (result.t()).print("result_log_dens: ");
    cout << " b:" << b.t()<< std::endl;
    cout << " exp(2 * b):" <<   exp(2 * b).t()<< std::endl;
    cout << " nu:" <<   nu.t()<< std::endl;
    Rcpp::stop("Error in log-density:");
  }

  return  result;


}



// Exponent1 function

inline vec exponent_1(const double sigma,const vec &tau,const vec &m,const vec &b){

  return (exp(b)-m)/(sqrt(sigma*tau));
}


// Exponent2 function

inline vec exponent_2(const double sigma,const vec &tau,const vec &m,const vec &b){


  return (-exp(b)-m)/(sqrt(sigma*tau));
}


// log density for Failed Stop trial (u<ssd)

vec log_density(const double sigma, const vec &tau, const vec &b,const vec &m) {

  // Rcpp::Rcout<<"A"<< endl;

  vec x = exponent_1(sigma,tau,m,b);
  vec result = b-(log(sigma)/2)- ((3 * log(  tau)) / 2)
    +log_normpdf(x);


  return  result;

}





// phi_rat function

vec phi_rat(const vec q2,const bool lt = 1){

  vec y1 = q2;
  y1.transform([&lt](double E) { return R::pnorm(E, 0, 1, lt, 1); });

  vec norm_dens=log_normpdf(q2);

  return (exp(y1-norm_dens));



}


// SSD_Dif function

inline vec SSD_Dif(const double sigma,const vec &tau,const vec &SSD,const vec &nu1,const vec &nu2,const double DEL,const double DEL_s){

   return ((2*(SSD+DEL_s-DEL)%(exp(nu2)-exp(nu1)))/(sqrt(sigma*tau)));

  // return (((SSD+DEL_s-DEL)%(exp(nu2)-exp(nu1)))/(sqrt(sigma*tau)));



}


// log density (Failed Stop trial)


// vec log_dens_s(const double sigma,const vec &tau,const vec &tau_s,const vec &SSD,const vec &b, const vec &nu1,
//                const vec &nu2,const double DEL,const double DEL_s,const bool lt = 1){
//
//
//
//   vec m=(exp(nu1)%(SSD+DEL_s-DEL))+(exp(nu2)%tau_s);
//   vec log_dens=log_density(sigma,tau,b,m);
//
//   vec diff=SSD_Dif(sigma,tau,SSD,nu1,nu2,DEL,DEL_s);
//
//   vec q2 = exponent_2(sigma,tau,m,b);
//
//   vec phi_ratio=phi_rat(q2,lt);
//
//   vec val=phi_ratio%diff;
//
//   vec lp = log1p(val);
//
//   return (log_dens+lp);
//
// }


vec log_dens_s(const double sigma, const vec &tau, const vec &tau_s, const vec &SSD,
               const vec &b, const vec &nu1, const vec &nu2,
               const double DEL, const double DEL_s, const bool lt = 1) {

  vec m = (exp(nu1) % (SSD + DEL_s - DEL)) + (exp(nu2) % tau_s);


  vec log_dens = log_density(sigma, tau, b, m);

  vec diff = SSD_Dif(sigma, tau, SSD, nu1, nu2, DEL, DEL_s);

  vec q2 = exponent_2(sigma, tau, m, b);

  vec phi_ratio = phi_rat(q2, lt);

  vec val = phi_ratio % diff;

  val.clamp(-0.999,datum::inf);

  vec lp = log1p(val);



  vec result = log_dens + lp;


  if (!result.is_finite()) {
    Rcpp::Rcout << "NaN in final result (log_dens + lp)\n";
    cout << "DEL: " << DEL << ", DEL_s: " << DEL_s << ", lt: " << lt << "\n";
    Rcpp::Rcout << "result_log_dens_s: " << result.t();

    // Print all input parameters again
    cout << "--- Input Parameters ---\n";
    cout << "sigma: " << sigma << "\n";
    
    cout << "tau: " << tau.t();
    cout << "tau_s: " << tau_s.t();
    cout << "SSD: " << SSD.t();
    cout << "b: " << b.t();
    cout << "nu1: " << nu1.t();
    cout << "nu2: " << nu2.t();
  }





  return result;
}





// log density (Stop Trial)

vec log_dens_stop(const double sigma, const vec &tau_s, const double b_stop, const double nu_stop) {

  vec result(tau_s.n_elem);
  result.fill(trunc_log(0.0));  // if trunc_log(0.0) = -1e10 or similar

  // Logical index for tau_s > 0
  uvec valid_idx = find(tau_s > 0);

  if (!valid_idx.is_empty()) {
    vec tau_valid = tau_s.elem(valid_idx);

    vec DF=(exp(b_stop))-(exp(nu_stop)*tau_valid);


    vec partial_result = b_stop - ((log(2 * datum::pi * sigma)) / 2) -
      ((3 * log(tau_valid)) / 2) - ((square(DF)) / (2 * sigma * tau_valid));


    result.elem(valid_idx) = partial_result;
  }

  // Check for NaNs or infinite values (excluding zeros from tau_s <= 0)
  if (!result.is_finite()) {
    result.t().print("result_log_dens_stop: ");
    std::cout << "b_stop = " << b_stop << std::endl;
    std::cout << "exp(nu_stop) = " << exp(nu_stop) << std::endl;
    tau_s.t().print("tau_s: ");
    Rcpp::stop("Error in log-density_stop:");
  }

  return result;
}


/////////////////////////////////////////////////////////////////////////// log Survival ////////////////////////////////////////////////////////////////////////////////////


// log Survival (Go Trial)


vec log_survival(const double sigma, const vec &tau, const vec &b, const vec &nu,const bool lt = 1) {

  // Rcpp::Rcout << "D" << std::endl;

  vec A = -sqrt((exp(2*b)) / (sigma * tau));

  vec B = (exp(nu - b)) % tau;

  vec C = (2 * (exp(nu + b))) / sigma;

  vec q1 = A % (B - 1);

  vec q2 = A % (B+1);

  vec y1 = q1;
  y1.transform([&lt](double E) { return R::pnorm(E, 0, 1, lt, 1); });


  vec y2 = q2;
  y2.transform([&lt](double D) { return R::pnorm(D, 0, 1, lt, 1); });


  vec y3 = C + y2;

  vec val = -exp(y3 - y1);

  val.clamp(-0.999,datum::inf);
  
  vec result = y1 + log1p(val);
  

  if(any(val<-1)){
    (val.t()).print("val_log_survival: ");
    (result.t()).print("result_log_survival: ");
    (exp(b.t())).print("boundary");
    (exp(nu.t())).print("drift");

    Rcpp::stop("Error in log-survival:");
  }

  return result;
}


// log Survival (Failed Stop Trial)

vec log_survival_s(const double sigma, const vec &tau,const vec &tau_s,const vec &SSD, const vec &b,const vec &nu1,
                   const vec &nu2,const double DEL,const double DEL_s,const bool lt = 1) {

  // Rcpp::Rcout << "D" << std::endl;

  vec m=(exp(nu1)%(SSD+DEL_s-DEL))+(exp(nu2)%tau_s);

  vec q1 = exponent_1(sigma,tau,m,b);

  vec y1 = q1;
  y1.transform([&lt](double E) { return R::pnorm(E, 0, 1, lt, 1); });

  vec C=(2*exp(b)%m)/(sigma*tau);

  vec q2 = exponent_2(sigma,tau,m,b);

  vec y2 = q2;
  y2.transform([&lt](double F) { return R::pnorm(F, 0, 1, lt, 1); });


  vec y3 = C + y2;

  vec val = -exp(y3 - y1);


  val.clamp(-0.999,datum::inf);

  vec result= (y1 + log1p(val));


  if (any(val < -1)) {
    val.t().print("val_log_survival_s: ");
    m.t().print("m: ");
    result.t().print("result_log_survival_s: ");
    y1.t().print("y1: ");
    y2.t().print("y2: ");
    Rcpp::stop("Error in log-survival_s:");
  }

  return result;

}






// log Survival (Stop Trial)


vec log_survival_stop(const double sigma, const vec &tau_s, const double b_stop, const double nu_stop, const bool lt = 1) {

  vec result(tau_s.n_elem, fill::zeros);

  // Find valid indices where tau_s > 0
  uvec valid_idx = find(tau_s > 0);

  if (!valid_idx.is_empty()) {
    vec tau_valid = tau_s.elem(valid_idx);

    vec A = -sqrt((exp(2*b_stop)) / (sigma * tau_valid));

    vec B = (exp(nu_stop - b_stop)) * tau_valid;

    double C = (2 * (exp(nu_stop + b_stop))) / sigma;

    vec q1 = A % (B - 1);
    vec q2 = A % (B + 1);

    vec y1 = q1;
    y1.transform([&lt](double E) { return R::pnorm(E, 0, 1, lt, 1); });

    vec y2 = q2;
    y2.transform([&lt](double D) { return R::pnorm(D, 0, 1, lt, 1); });

    vec y3 = C + y2;
    vec val = -exp(y3 - y1);


    val.clamp(-0.999,datum::inf);


    vec partial_result = y1 + log1p(val);

    // Safety check for numerical stability
    if (any(val < -1)) {
      val.t().print("val_log_survival_stop: ");
      partial_result.t().print("partial_result_log_survival_stop: ");
      y1.t().print("y1: ");
      y2.t().print("y2: ");
      Rcpp::stop("Error in log-survival_stop:");
    }

    result.elem(valid_idx) = partial_result;
  }

  return result;
}


/////////////////////////////////////////////////////////////////////////// Penalty Function ////////////////////////////////////////////////////////////////////////////////////


// Penalty Function (Go Trial)


vec penalty_fun(const vec &nu_s, const vec &nu_s_squared,const vec &penal_param) {

  //Rcpp::Rcout<<"F"<< endl;


  vec X=(exp(2*penal_param(0)))*nu_s_squared;
  double Y=exp(2*penal_param(1));
  vec hpot=sqrt(X+Y);


  return 1/(-(exp(penal_param(0))*nu_s)+hpot);

  // return zeros(size(nu_s));
}



// Penalty Function (Stop Trial)

double penalty_fun_s(const double &nu_stop, const vec &penal_param) {

  //Rcpp::Rcout<<"G"<< endl;


  // double X=(exp(2*penal_param(0)))*(exp(2*nu_stop));
  double X=(exp(2*(penal_param(0)+nu_stop)));
  double Y=exp(2*penal_param(1));
  double hpot=sqrt(X+Y);


  // return 1/(-(exp(penal_param(0))*exp(nu_stop))+hpot);


  return 1/(-(exp(penal_param(0)+nu_stop))+hpot);

}


///////////////////////////////////////////////////////////////////////////////// Weight Function ////////////////////////////////////////////////////////////////////////////////////


// weight Function (Go Trial)


inline vec weight(const double sigma,const vec &tau, const vec &nu,const vec &b){

  /*Rcpp::Rcout<<"H"<< endl;*/

  return exp((2*log_dens(sigma,tau,b,nu))-log_survival(sigma,tau,b,nu));

}




// weight Function(Go Process-Stop Trial)


inline vec weight_s(const double sigma,const vec &tau,const vec &tau_s,const vec &SSD,
                    const vec &b,const vec &nu1,const vec &nu2,
                    const double DEL,const double DEL_s,const bool lt = 1){

  // Rcpp::Rcout<<"I"<< endl;

  return exp((2*log_dens_s(sigma,tau,tau_s,SSD,b,nu1,nu2,DEL,DEL_s,lt))-log_survival_s(sigma,tau,tau_s,SSD,b,nu1,nu2,DEL,DEL_s,lt));

}






// weight Function (Stop Trial)

inline vec weight_stop(const vec &tau_s, const double sigma,const vec stop_param){

  /*Rcpp::Rcout<<"J"<< endl;*/

  return exp((2*log_dens_stop(sigma,tau_s,stop_param(0),stop_param(1)))-log_survival_stop(sigma,tau_s,stop_param(0),stop_param(1)));

}








///////////////////////////////////////////////////// Incorporating Random effect Components /////////////////////////////////////////////////////////////////////////////

// log Spectral Density

inline arma::vec log_spec_dens( const double nu,const double log_l,const vec &lam, const double D=1) {
  // lam is the constant lambda_{j} elements from the paper
  return (D + nu) * M_LN2 - 2 * nu * log_l  + nu * log(nu)  + 0.5 * D * log(M_PI)
  - ((D / 2) + nu) * log((2 * nu) * exp(-2 * log_l) + 4 *  pow(M_PI,2) *lam)
  - lgamma(nu)  + lgamma(D / 2 + nu);
}




// prior on log(l); normalizing constants are not included****/
double log_prior_ARD(const double y,const double kappa) {
  return log(kappa) - kappa * exp(-y) - y;
}



//**prior on log(alpha); normalizing constants are not included

double log_prior_alph(const double y){
  return -mlpack::LogAdd(y/2,-y/2);
}


// // //**prior on log(alpha); normalizing constants are not included
//
// double log_prior_alph(const double y) {
//   return std::log(2 * std::cosh(y));
// }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// log likelihood (Go Process)


double log_lhood(const double Inhib_likelihood,
                 const double sigma, const vec &tau,const vec &tau_s,const vec &SSD,const vec &Ind_GF,
                 const vec &b_l, const vec &b_r,
                 const vec &nu_l, const vec &nu_r, const vec &nu_l_p, const vec &nu_r_p,
                 const vec &nu_l_s,const vec &nu_r_s,
                 const uvec &Ind_L,const uvec &Ind_R,
                 const uvec &Ind_LCR,const uvec &Ind_LIR,const uvec &Ind_RIR,const uvec &Ind_RCR,
                 const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,const uvec &Ind_S_RIR,const uvec &Ind_S_RCR,

                 const vec &lk_LCR_FS, const vec &lk_LIR_FS,const vec &lk_RCR_FS, const vec &lk_RIR_FS,
                 const vec &prob_param,const double T_Go,const double K){

  //Rcpp::Rcout<<"K"<< endl;

  double sum_dens=mlpack::LogAdd(prob_param(1),prob_param(2));

  double A=(K*prob_param(0))+((T_Go-K)*sum_dens);




  /// Density
  // Go Process

  vec B= log_dens(sigma,g(tau, Ind_L, Ind_LCR ),g(b_l, Ind_L, Ind_LCR ),g(nu_l, Ind_L, Ind_LCR ));
  vec C= log_dens(sigma,g(tau, Ind_L, Ind_LIR ),g(b_r, Ind_L, Ind_LIR ),h(nu_r_p,Ind_LIR ));
  vec D= log_dens(sigma,g(tau, Ind_R, Ind_RCR ),g(b_r, Ind_R, Ind_RCR ),g(nu_r, Ind_R, Ind_RCR ));
  vec E= log_dens(sigma,g(tau, Ind_R, Ind_RIR ),g(b_l, Ind_R, Ind_RIR ),h(nu_l_p, Ind_RIR ));



  /// Survival
  // Go Process
  vec F=log_survival(sigma,g(tau, Ind_L, Ind_LCR ),g(b_r, Ind_L, Ind_LCR ),h(nu_r_p, Ind_LCR ));
  vec G=log_survival(sigma,g(tau, Ind_L, Ind_LIR ),g(b_l, Ind_L, Ind_LIR ),g(nu_l, Ind_L, Ind_LIR ));
  vec H=log_survival(sigma,g(tau, Ind_R, Ind_RCR ),g(b_l, Ind_R, Ind_RCR ),h(nu_l_p, Ind_RCR ));
  vec I=log_survival(sigma,g(tau, Ind_R, Ind_RIR ),g(b_r, Ind_R, Ind_RIR ),g(nu_r, Ind_R, Ind_RIR));



  // Inhibit Trial

  double W=Inhib_likelihood;


  double result= A+sum(B)+sum(C)+sum(D)+sum(E)+sum(F)+sum(G)+sum(H)+sum(I)+
    sum(lk_LCR_FS)+sum(lk_LIR_FS)+sum(lk_RCR_FS)+sum(lk_RIR_FS)+W;



  // Rcpp::Rcout<<"result"<<result<<endl;


  if(!std::isfinite(result)){
    cout << "result:" << result << std::endl;
    cout << "A:" << A << std::endl;
    cout << "B:" << sum(B) << std::endl;
    cout << "C:" << sum(C) << std::endl;
    cout << "D:" << sum(D) << std::endl;
    cout << "E:" << sum(E) << std::endl;
    cout << "F:" << sum(F) << std::endl;
    cout << "G:" << sum(G) << std::endl;
    cout << "H:" << sum(H) << std::endl;
    cout << "I:" << sum(I) << std::endl;
    cout << "lk_LCR_FS:" << sum(lk_LCR_FS) << std::endl;
    cout << "lk_LCR_FS:" << sum(lk_LIR_FS) << std::endl;
    cout << "lk_RCR_FS:" << sum(lk_RCR_FS) << std::endl;
    cout << "lk_RIR_FS:" << sum(lk_RIR_FS) << std::endl;
    cout << "W:" << W << std::endl;

  Rcpp::stop("Error in log-likelihood:");
  }



  return result;
}














// MVN_prior

double mvn_prior(const mat &gama,const vec &mu_gl,const mat &sigma_gl_inv,
                 const vec &mu_gr,const mat &sigma_gr_inv,
                 double eta_b_l,double eta_b_r){

  unsigned P=gama.n_rows;

  vec mean_diff_L=gama.col(0)-mu_gl;

  mat Z_gl=sigma_gl_inv*mean_diff_L;
  double D_gl=eta_b_l*dot(mean_diff_L,Z_gl);


  vec mean_diff_R=gama.col(1)-mu_gr;
  mat Z_gr=sigma_gr_inv*mean_diff_R;
  double D_gr=eta_b_r*dot(mean_diff_R,Z_gr);


  double result=((P-1)*log(eta_b_l))/2+((P-1)*log(eta_b_r))/2-
    (D_gl/2)-(D_gr/2);

  return result;

}



// SS1

inline double SS1(const double alpha,const vec &g,  const vec &S){

  return (accu((square(g))%exp(-S-alpha)));

}





// MVN_prior2

double mvn_prior2(const double alpha_gama_l,const double alpha_gama_r,
                  const mat &gama,const vec &S_gama_l,const vec &S_gama_r){

  // Rcpp::Rcout<<"M"<< endl;

  unsigned m=gama.n_rows;

  double D_gl=SS1(alpha_gama_l,gama.col(0),S_gama_l);

  double D_gr=SS1(alpha_gama_r,gama.col(1),S_gama_r);



  double result=-((m*alpha_gama_l)/2)-(accu(S_gama_l)/2)-(D_gl/2)
                -((m*alpha_gama_r)/2)-(accu(S_gama_r)/2)-(D_gr/2);

  return result;

}


// log_normal_prior

inline double log_normal_prior(const vec &penal_param,const double mu_lambda_prime,const double sigma_lambda_prime,
                               const double mu_alpha_prime,const double sigma_alpha_prime){


  return (-((log(sigma_lambda_prime)) / 2) -((pow((penal_param(0) - mu_lambda_prime),2)) / (2 * sigma_lambda_prime))
            -((log(sigma_alpha_prime)) / 2) -((pow((penal_param(1) - mu_alpha_prime),2)) / (2 * sigma_alpha_prime)));

}


// log-gama_prior

double log_gama_prior(const vec &prob_param,double a_01,double a_00, double a_10){

  double result=-(lgamma(a_00))+(a_00*prob_param(1))-exp(prob_param(1))-
    (lgamma(a_10))+(a_10*prob_param(2))-exp(prob_param(2))-
    (lgamma(a_01))+(a_01*prob_param(0))-exp(prob_param(0));
  return result;

}





// Rand_prior

double rand_prior(const vec &rand_param, double kappa){

  double result=log_prior_alph(rand_param(0))+log_prior_ARD(rand_param(1),kappa);

  return result;
}




///////////////////////////////////////////////////////////////////////////////// Updating Necessary Vectors //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function-1

field <field<vec>> update_lk_FS(const field <vec> &tau,const field <vec> &tau_s,
                                const double sigma,const field <vec> &SSD,
                                const field <vec> &b_l,const field <vec> &b_r,
                                const field <vec> &nu_l,const field <vec> &nu_r,
                                const field <vec> &nu_r_p,const field <vec> &nu_l_p,
                                const field <vec> &nu_l_s,const field <vec> &nu_r_s,
                                const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                                const field <uvec> &Ind_LCR, const field <uvec> &Ind_LIR,
                                const field <uvec>&Ind_RCR,const field <uvec>&Ind_RIR,
                                const field <uvec> &Ind_S_LCR,const field <uvec> &Ind_S_LIR,
                                const field <uvec>&Ind_S_RCR,const field <uvec>&Ind_S_RIR,
                                const mat &stop_param,const mat &prob_param,
                                const vec &DEL, const vec &DEL_s,
                                const bool lt=1){


  unsigned N=tau.n_elem;

  field<vec>lk_LCR_FS(N);
  field<vec>lk_LIR_FS(N);
  field<vec>lk_RCR_FS(N);
  field<vec>lk_RIR_FS(N);

  for (unsigned i = 0; i < N; ++i) {

    /// Density
    // Go Process-Failed Stop Trial

    vec B= log_dens(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(b_l(i), Ind_L(i), Ind_S_LCR(i) ),g(nu_l(i), Ind_L(i), Ind_S_LCR(i) ));
    vec C= log_dens(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(b_r(i), Ind_L(i), Ind_S_LIR(i) ),h(nu_r_p(i),Ind_S_LIR(i) ));
    vec D= log_dens(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(b_r(i), Ind_R(i), Ind_S_RCR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RCR(i) ));
    vec E= log_dens(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(b_l(i), Ind_R(i), Ind_S_RIR(i) ),h(nu_l_p(i), Ind_S_RIR(i) ));



    /// Survival
    // Go Process-Failed Stop Trial


    vec F=log_survival(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(b_r(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p(i), Ind_S_LCR(i) ));
    vec G=log_survival(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(b_l(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l(i), Ind_L(i), Ind_S_LIR(i) ));
    vec H=log_survival(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(b_l(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p(i), Ind_S_RCR(i) ));
    vec I=log_survival(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(b_r(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RIR(i)));


    /// Density
    // Failed Stop Trial


    vec J=log_dens_s(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),g(SSD(i), Ind_L(i), Ind_S_LCR(i) ),g(b_l(i), Ind_L(i), Ind_S_LCR(i) ),
                     g(nu_l(i), Ind_L(i), Ind_S_LCR(i) ),g(nu_l_s(i), Ind_L(i), Ind_S_LCR(i) ),DEL(i),DEL_s(i),lt);


    vec K=log_dens_s(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),g(SSD(i), Ind_L(i), Ind_S_LIR(i) ),g(b_r(i), Ind_L(i), Ind_S_LIR(i) ),
                     h(nu_r_p(i), Ind_S_LIR(i) ),g(nu_r_s(i), Ind_L(i), Ind_S_LIR(i) ),DEL(i),DEL_s(i),lt);


    vec L=log_dens_s(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),g(SSD(i), Ind_R(i), Ind_S_RCR(i) ),g(b_r(i), Ind_R(i), Ind_S_RCR(i) ),
                     g(nu_r(i), Ind_R(i), Ind_S_RCR(i) ),g(nu_r_s(i), Ind_R(i), Ind_S_RCR(i) ),DEL(i),DEL_s(i),lt);

    vec M=log_dens_s(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),g(SSD(i), Ind_R(i), Ind_S_RIR(i) ),g(b_l(i), Ind_R(i), Ind_S_RIR(i) ),
                     h(nu_l_p(i), Ind_S_RIR(i) ),g(nu_l_s(i), Ind_R(i), Ind_S_RIR(i) ),DEL(i),DEL_s(i),lt);


    /// Survival
    // Failed Stop Trial


    vec N=log_survival_s(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),g(SSD(i), Ind_L(i), Ind_S_LCR(i) ),g(b_r(i), Ind_L(i), Ind_S_LCR(i) ),
                         h(nu_r_p(i), Ind_S_LCR(i) ),g(nu_r_s(i), Ind_L(i), Ind_S_LCR(i) ),DEL(i),DEL_s(i),lt);


    vec O=log_survival_s(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),g(SSD(i), Ind_L(i), Ind_S_LIR(i) ),g(b_l(i), Ind_L(i), Ind_S_LIR(i) ),
                         g(nu_l(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l_s(i), Ind_L(i), Ind_S_LIR(i) ),DEL(i),DEL_s(i),lt);

    vec P=log_survival_s(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),g(SSD(i), Ind_R(i), Ind_S_RCR(i) ),g(b_l(i), Ind_R(i), Ind_S_RCR(i) ),
                         h(nu_l_p(i), Ind_S_RCR(i) ),g(nu_l_s(i), Ind_R(i), Ind_S_RCR(i) ),DEL(i),DEL_s(i),lt);

    vec Q=log_survival_s(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),g(SSD(i), Ind_R(i), Ind_S_RIR(i) ),g(b_r(i), Ind_R(i), Ind_S_RIR(i) ),
                         g(nu_r(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r_s(i), Ind_R(i), Ind_S_RIR(i) ),DEL(i),DEL_s(i),lt);



    //survival

    // Stop Trial

    // vec stop_param_i=stop_param.col(i);

    vec R_LCR=log_survival_stop(sigma,g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),stop_param(0,i),stop_param(1,i));

    vec R_LIR=log_survival_stop(sigma,g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),stop_param(0,i),stop_param(1,i));


    vec R_RCR=log_survival_stop(sigma,g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),stop_param(0,i),stop_param(1,i));

    vec R_RIR=log_survival_stop(sigma,g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),stop_param(0,i),stop_param(1,i));





    // Failed Stop Trial Combined

    lk_LCR_FS(i)=compute_log_add((prob_param(2,i)+B+F),(prob_param(1,i)+R_LCR+J+N));

    lk_LIR_FS(i)=compute_log_add((prob_param(2,i)+C+G),(prob_param(1,i)+R_LIR+K+O));

    lk_RCR_FS(i)=compute_log_add((prob_param(2,i)+D+H),(prob_param(1,i)+R_RCR+L+P));

    lk_RIR_FS(i)=compute_log_add((prob_param(2,i)+E+I),(prob_param(1,i)+R_RIR+M+Q));

  }



  field<field<vec>> result(4);

  result(0)=lk_LCR_FS;
  result(1)=lk_LIR_FS;

  result(2)=lk_RCR_FS;
  result(3)=lk_RIR_FS;

  return result;

}



// Function-2

field <field<vec>> update_lk_FS2(const field <vec> &tau,const field <vec> &tau_s,
                                 const double sigma,const field <vec> &SSD,
                                 const field <vec> &b_l,const field <vec> &b_r,
                                 const field <vec> &nu_l,const field <vec> &nu_r,
                                 const field <vec> &nu_r_p,const field <vec> &nu_l_p,
                                 const field <vec> &nu_l_s,const field <vec> &nu_r_s,
                                 const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                                 const field <uvec> &Ind_LCR, const field <uvec> &Ind_LIR,
                                 const field <uvec>&Ind_RCR,const field <uvec>&Ind_RIR,
                                 const field <uvec> &Ind_S_LCR,const field <uvec> &Ind_S_LIR,
                                 const field <uvec>&Ind_S_RCR,const field <uvec>&Ind_S_RIR,
                                 const mat &stop_param,const mat &prob_param,
                                 const vec &DEL, const vec &DEL_s,
                                 const bool lt=1){


  // Rcpp::Rcout<<"lk_FS"<<endl;


  unsigned N=tau.n_elem;

  field<vec>lk_LCR_FS(N);
  field<vec>lk_LIR_FS(N);
  field<vec>lk_RCR_FS(N);
  field<vec>lk_RIR_FS(N);


  field<vec>diff_X_LCR(N);
  field<vec>diff_Y_LCR(N);

  field<vec>diff_X_LIR(N);
  field<vec>diff_Y_LIR(N);


  field<vec>diff_X_RCR(N);
  field<vec>diff_Y_RCR(N);

  field<vec>diff_X_RIR(N);
  field<vec>diff_Y_RIR(N);


  for (unsigned i = 0; i < N; ++i) {

    // Density
    // Go Process-Failed Stop Trial

    vec B= log_dens(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(b_l(i), Ind_L(i), Ind_S_LCR(i) ),g(nu_l(i), Ind_L(i), Ind_S_LCR(i) ));
    vec C= log_dens(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(b_r(i), Ind_L(i), Ind_S_LIR(i) ),h(nu_r_p(i),Ind_S_LIR(i) ));
    vec D= log_dens(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(b_r(i), Ind_R(i), Ind_S_RCR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RCR(i) ));
    vec E= log_dens(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(b_l(i), Ind_R(i), Ind_S_RIR(i) ),h(nu_l_p(i), Ind_S_RIR(i) ));



    /// Survival
    // Go Process-Failed Stop Trial


    vec F=log_survival(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(b_r(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p(i), Ind_S_LCR(i) ));
    vec G=log_survival(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(b_l(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l(i), Ind_L(i), Ind_S_LIR(i) ));
    vec H=log_survival(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(b_l(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p(i), Ind_S_RCR(i) ));
    vec I=log_survival(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(b_r(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RIR(i)));


    /// Density
    // Failed Stop Trial



    vec J=log_dens_s(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),g(SSD(i), Ind_L(i), Ind_S_LCR(i) ),g(b_l(i), Ind_L(i), Ind_S_LCR(i) ),
                     g(nu_l(i), Ind_L(i), Ind_S_LCR(i) ),g(nu_l_s(i), Ind_L(i), Ind_S_LCR(i) ),DEL(i),DEL_s(i),lt);


    vec K=log_dens_s(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),g(SSD(i), Ind_L(i), Ind_S_LIR(i) ),g(b_r(i), Ind_L(i), Ind_S_LIR(i) ),
                     h(nu_r_p(i), Ind_S_LIR(i) ),g(nu_r_s(i), Ind_L(i), Ind_S_LIR(i) ),DEL(i),DEL_s(i),lt);


    vec L=log_dens_s(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),g(SSD(i), Ind_R(i), Ind_S_RCR(i) ),g(b_r(i), Ind_R(i), Ind_S_RCR(i) ),
                     g(nu_r(i), Ind_R(i), Ind_S_RCR(i) ),g(nu_r_s(i), Ind_R(i), Ind_S_RCR(i) ),DEL(i),DEL_s(i),lt);

    vec M=log_dens_s(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),g(SSD(i), Ind_R(i), Ind_S_RIR(i) ),g(b_l(i), Ind_R(i), Ind_S_RIR(i) ),
                     h(nu_l_p(i), Ind_S_RIR(i) ),g(nu_l_s(i), Ind_R(i), Ind_S_RIR(i) ),DEL(i),DEL_s(i),lt);


    /// Survival
    // Failed Stop Trial


    vec N=log_survival_s(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),g(SSD(i), Ind_L(i), Ind_S_LCR(i) ),g(b_r(i), Ind_L(i), Ind_S_LCR(i) ),
                         h(nu_r_p(i), Ind_S_LCR(i) ),g(nu_r_s(i), Ind_L(i), Ind_S_LCR(i) ),DEL(i),DEL_s(i),lt);


    vec O=log_survival_s(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),g(SSD(i), Ind_L(i), Ind_S_LIR(i) ),g(b_l(i), Ind_L(i), Ind_S_LIR(i) ),
                         g(nu_l(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l_s(i), Ind_L(i), Ind_S_LIR(i) ),DEL(i),DEL_s(i),lt);

    vec P=log_survival_s(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),g(SSD(i), Ind_R(i), Ind_S_RCR(i) ),g(b_l(i), Ind_R(i), Ind_S_RCR(i) ),
                         h(nu_l_p(i), Ind_S_RCR(i) ),g(nu_l_s(i), Ind_R(i), Ind_S_RCR(i) ),DEL(i),DEL_s(i),lt);

    vec Q=log_survival_s(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),g(SSD(i), Ind_R(i), Ind_S_RIR(i) ),g(b_r(i), Ind_R(i), Ind_S_RIR(i) ),
                         g(nu_r(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r_s(i), Ind_R(i), Ind_S_RIR(i) ),DEL(i),DEL_s(i),lt);



    //survival

    // Stop Trial

    // vec stop_param_i=stop_param.col(i);

    vec R_LCR=log_survival_stop(sigma,g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),stop_param(0,i),stop_param(1,i));

    vec R_LIR=log_survival_stop(sigma,g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),stop_param(0,i),stop_param(1,i));


    vec R_RCR=log_survival_stop(sigma,g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),stop_param(0,i),stop_param(1,i));

    vec R_RIR=log_survival_stop(sigma,g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),stop_param(0,i),stop_param(1,i));



    // Failed Stop Trial Combined

    // vec prob_param_i=prob_param.col(i);

    vec X_LCR=(prob_param(2,i)+B+F);
    vec Y_LCR=(prob_param(1,i)+R_LCR+J+N);

    vec X_LIR=(prob_param(2,i)+C+G);
    vec Y_LIR=(prob_param(1,i)+R_LIR+K+O);

    vec X_RCR=(prob_param(2,i)+D+H);
    vec Y_RCR=(prob_param(1,i)+R_RCR+L+P);

    vec X_RIR=(prob_param(2,i)+E+I);
    vec Y_RIR=(prob_param(1,i)+R_RIR+M+Q);


    lk_LCR_FS(i)=compute_log_add(X_LCR,Y_LCR);

    lk_LIR_FS(i)=compute_log_add(X_LIR,Y_LIR);

    lk_RCR_FS(i)=compute_log_add(X_RCR,Y_RCR);

    lk_RIR_FS(i)=compute_log_add(X_RIR,Y_RIR);


    diff_X_LCR(i)=exp(X_LCR- lk_LCR_FS(i));
    diff_Y_LCR(i)=exp(Y_LCR- lk_LCR_FS(i));

    diff_X_LIR(i)=exp(X_LIR- lk_LIR_FS(i));
    diff_Y_LIR(i)=exp(Y_LIR- lk_LIR_FS(i));

    diff_X_RCR(i)=exp(X_RCR- lk_RCR_FS(i));
    diff_Y_RCR(i)=exp(Y_RCR- lk_RCR_FS(i));

    diff_X_RIR(i)=exp(X_RIR- lk_RIR_FS(i));
    diff_Y_RIR(i)=exp(Y_RIR- lk_RIR_FS(i));

  }



  field<field<vec>> result(12);

  result(0)=lk_LCR_FS;
  result(1)=lk_LIR_FS;

  result(2)=lk_RCR_FS;
  result(3)=lk_RIR_FS;


  result(4)=diff_X_LCR;
  result(5)=diff_Y_LCR;

  result(6)=diff_X_LIR;
  result(7)=diff_Y_LIR;


  result(8)=diff_X_RCR;
  result(9)=diff_Y_RCR;

  result(10)=diff_X_RIR;
  result(11)=diff_Y_RIR;

  return result;

}



// Function 3

field<vec> update_lk_FS3(const vec &tau,const vec &tau_s,
                         const double sigma,const vec &SSD,
                         const vec &b_l,const vec &b_r,
                         const vec &nu_l,const vec &nu_r,
                         const vec &nu_r_p,const vec &nu_l_p,
                         const vec &nu_l_s,const vec &nu_r_s,
                         const uvec &Ind_L,const uvec &Ind_R,
                         const uvec &Ind_LCR, const uvec &Ind_LIR,
                         const uvec &Ind_RCR,const uvec &Ind_RIR,
                         const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,
                         const uvec &Ind_S_RCR,const uvec &Ind_S_RIR,
                         const vec &stop_param,const vec &prob_param,
                         const double DEL, const double DEL_s,
                         const bool lt=1){



  /// Density
  // Go Process-Failed Stop Trial

  vec B= log_dens(sigma,g(tau, Ind_L, Ind_S_LCR ),g(b_l, Ind_L, Ind_S_LCR ),g(nu_l, Ind_L, Ind_S_LCR ));
  vec C= log_dens(sigma,g(tau, Ind_L, Ind_S_LIR ),g(b_r, Ind_L, Ind_S_LIR ),h(nu_r_p,Ind_S_LIR ));
  vec D= log_dens(sigma,g(tau, Ind_R, Ind_S_RCR ),g(b_r, Ind_R, Ind_S_RCR ),g(nu_r, Ind_R, Ind_S_RCR ));
  vec E= log_dens(sigma,g(tau, Ind_R, Ind_S_RIR ),g(b_l, Ind_R, Ind_S_RIR ),h(nu_l_p, Ind_S_RIR ));



  /// Survival
  // Go Process-Failed Stop Trial


  vec F=log_survival(sigma,g(tau, Ind_L, Ind_S_LCR ),g(b_r, Ind_L, Ind_S_LCR ),h(nu_r_p, Ind_S_LCR ));
  vec G=log_survival(sigma,g(tau, Ind_L, Ind_S_LIR ),g(b_l, Ind_L, Ind_S_LIR ),g(nu_l, Ind_L, Ind_S_LIR ));
  vec H=log_survival(sigma,g(tau, Ind_R, Ind_S_RCR ),g(b_l, Ind_R, Ind_S_RCR ),h(nu_l_p, Ind_S_RCR ));
  vec I=log_survival(sigma,g(tau, Ind_R, Ind_S_RIR ),g(b_r, Ind_R, Ind_S_RIR ),g(nu_r, Ind_R, Ind_S_RIR));


  /// Density
  // Failed Stop Trial



  vec J=log_dens_s(sigma,g(tau, Ind_L, Ind_S_LCR ),g(tau_s, Ind_L, Ind_S_LCR ),g(SSD, Ind_L, Ind_S_LCR ),g(b_l, Ind_L, Ind_S_LCR ),
                   g(nu_l, Ind_L, Ind_S_LCR ),g(nu_l_s, Ind_L, Ind_S_LCR ),DEL,DEL_s,lt);


  vec K=log_dens_s(sigma,g(tau, Ind_L, Ind_S_LIR ),g(tau_s, Ind_L, Ind_S_LIR ),g(SSD, Ind_L, Ind_S_LIR ),g(b_r, Ind_L, Ind_S_LIR ),
                   h(nu_r_p, Ind_S_LIR ),g(nu_r_s, Ind_L, Ind_S_LIR ),DEL,DEL_s,lt);


  vec L=log_dens_s(sigma,g(tau, Ind_R, Ind_S_RCR ),g(tau_s, Ind_R, Ind_S_RCR ),g(SSD, Ind_R, Ind_S_RCR ),g(b_r, Ind_R, Ind_S_RCR ),
                   g(nu_r, Ind_R, Ind_S_RCR ),g(nu_r_s, Ind_R, Ind_S_RCR ),DEL,DEL_s,lt);

  vec M=log_dens_s(sigma,g(tau, Ind_R, Ind_S_RIR ),g(tau_s, Ind_R, Ind_S_RIR ),g(SSD, Ind_R, Ind_S_RIR ),g(b_l, Ind_R, Ind_S_RIR ),
                   h(nu_l_p, Ind_S_RIR ),g(nu_l_s, Ind_R, Ind_S_RIR ),DEL,DEL_s,lt);


  /// Survival
  // Failed Stop Trial


  vec N=log_survival_s(sigma,g(tau, Ind_L, Ind_S_LCR ),g(tau_s, Ind_L, Ind_S_LCR ),g(SSD, Ind_L, Ind_S_LCR ),g(b_r, Ind_L, Ind_S_LCR ),
                       h(nu_r_p, Ind_S_LCR ),g(nu_r_s, Ind_L, Ind_S_LCR ),DEL,DEL_s,lt);


  vec O=log_survival_s(sigma,g(tau, Ind_L, Ind_S_LIR ),g(tau_s, Ind_L, Ind_S_LIR ),g(SSD, Ind_L, Ind_S_LIR ),g(b_l, Ind_L, Ind_S_LIR ),
                       g(nu_l, Ind_L, Ind_S_LIR ),g(nu_l_s, Ind_L, Ind_S_LIR ),DEL,DEL_s,lt);

  vec P=log_survival_s(sigma,g(tau, Ind_R, Ind_S_RCR ),g(tau_s, Ind_R, Ind_S_RCR ),g(SSD, Ind_R, Ind_S_RCR ),g(b_l, Ind_R, Ind_S_RCR ),
                       h(nu_l_p, Ind_S_RCR ),g(nu_l_s, Ind_R, Ind_S_RCR ),DEL,DEL_s,lt);

  vec Q=log_survival_s(sigma,g(tau, Ind_R, Ind_S_RIR ),g(tau_s, Ind_R, Ind_S_RIR ),g(SSD, Ind_R, Ind_S_RIR ),g(b_r, Ind_R, Ind_S_RIR ),
                       g(nu_r, Ind_R, Ind_S_RIR ),g(nu_r_s, Ind_R, Ind_S_RIR ),DEL,DEL_s,lt);



  //survival

  // Stop Trial

  // vec stop_param_i=stop_param.col(i);

  vec R_LCR=log_survival_stop(sigma,g(tau_s, Ind_L, Ind_S_LCR ),stop_param(0),stop_param(1));

  vec R_LIR=log_survival_stop(sigma,g(tau_s, Ind_L, Ind_S_LIR ),stop_param(0),stop_param(1));


  vec R_RCR=log_survival_stop(sigma,g(tau_s, Ind_R, Ind_S_RCR ),stop_param(0),stop_param(1));

  vec R_RIR=log_survival_stop(sigma,g(tau_s, Ind_R, Ind_S_RIR ),stop_param(0),stop_param(1));






  // Failed Stop Trial Combined


  vec lk_LCR_FS=compute_log_add((prob_param(2)+B+F),(prob_param(1)+R_LCR+J+N));

  vec lk_LIR_FS=compute_log_add((prob_param(2)+C+G),(prob_param(1)+R_LIR+K+O));

  vec lk_RCR_FS=compute_log_add((prob_param(2)+D+H),(prob_param(1)+R_RCR+L+P));

  vec lk_RIR_FS=compute_log_add((prob_param(2)+E+I),(prob_param(1)+R_RIR+M+Q));


  field<vec> result(4);

  result(0)=lk_LCR_FS;
  result(1)=lk_LIR_FS;

  result(2)=lk_RCR_FS;
  result(3)=lk_RIR_FS;

  return result;

}

// Function-4

field<vec> update_lk_FS4(const vec &tau,const vec &tau_s,
                         const double sigma,const vec &SSD,
                         const vec &b_l,const vec &b_r,
                         const vec &nu_l,const vec &nu_r,
                         const vec &nu_r_p,const vec &nu_l_p,
                         const vec &nu_l_s,const vec &nu_r_s,
                         const uvec &Ind_L,const uvec &Ind_R,
                         const uvec &Ind_LCR, const uvec &Ind_LIR,
                         const uvec &Ind_RCR,const uvec &Ind_RIR,
                         const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,
                         const uvec &Ind_S_RCR,const uvec &Ind_S_RIR,
                         const vec &stop_param,const vec &prob_param,
                         const double DEL, const double DEL_s,
                         const bool lt=1){



  /// Density
  // Go Process-Failed Stop Trial

  vec B= log_dens(sigma,g(tau, Ind_L, Ind_S_LCR ),g(b_l, Ind_L, Ind_S_LCR ),g(nu_l, Ind_L, Ind_S_LCR ));
  vec C= log_dens(sigma,g(tau, Ind_L, Ind_S_LIR ),g(b_r, Ind_L, Ind_S_LIR ),h(nu_r_p,Ind_S_LIR ));
  vec D= log_dens(sigma,g(tau, Ind_R, Ind_S_RCR ),g(b_r, Ind_R, Ind_S_RCR ),g(nu_r, Ind_R, Ind_S_RCR ));
  vec E= log_dens(sigma,g(tau, Ind_R, Ind_S_RIR ),g(b_l, Ind_R, Ind_S_RIR ),h(nu_l_p, Ind_S_RIR ));



  /// Survival
  // Go Process-Failed Stop Trial


  vec F=log_survival(sigma,g(tau, Ind_L, Ind_S_LCR ),g(b_r, Ind_L, Ind_S_LCR ),h(nu_r_p, Ind_S_LCR ));
  vec G=log_survival(sigma,g(tau, Ind_L, Ind_S_LIR ),g(b_l, Ind_L, Ind_S_LIR ),g(nu_l, Ind_L, Ind_S_LIR ));
  vec H=log_survival(sigma,g(tau, Ind_R, Ind_S_RCR ),g(b_l, Ind_R, Ind_S_RCR ),h(nu_l_p, Ind_S_RCR ));
  vec I=log_survival(sigma,g(tau, Ind_R, Ind_S_RIR ),g(b_r, Ind_R, Ind_S_RIR ),g(nu_r, Ind_R, Ind_S_RIR));


  /// Density
  // Failed Stop Trial



  vec J=log_dens_s(sigma,g(tau, Ind_L, Ind_S_LCR ),g(tau_s, Ind_L, Ind_S_LCR ),g(SSD, Ind_L, Ind_S_LCR ),g(b_l, Ind_L, Ind_S_LCR ),
                   g(nu_l, Ind_L, Ind_S_LCR ),g(nu_l_s, Ind_L, Ind_S_LCR ),DEL,DEL_s,lt);


  vec K=log_dens_s(sigma,g(tau, Ind_L, Ind_S_LIR ),g(tau_s, Ind_L, Ind_S_LIR ),g(SSD, Ind_L, Ind_S_LIR ),g(b_r, Ind_L, Ind_S_LIR ),
                   h(nu_r_p, Ind_S_LIR ),g(nu_r_s, Ind_L, Ind_S_LIR ),DEL,DEL_s,lt);


  vec L=log_dens_s(sigma,g(tau, Ind_R, Ind_S_RCR ),g(tau_s, Ind_R, Ind_S_RCR ),g(SSD, Ind_R, Ind_S_RCR ),g(b_r, Ind_R, Ind_S_RCR ),
                   g(nu_r, Ind_R, Ind_S_RCR ),g(nu_r_s, Ind_R, Ind_S_RCR ),DEL,DEL_s,lt);

  vec M=log_dens_s(sigma,g(tau, Ind_R, Ind_S_RIR ),g(tau_s, Ind_R, Ind_S_RIR ),g(SSD, Ind_R, Ind_S_RIR ),g(b_l, Ind_R, Ind_S_RIR ),
                   h(nu_l_p, Ind_S_RIR ),g(nu_l_s, Ind_R, Ind_S_RIR ),DEL,DEL_s,lt);


  /// Survival
  // Failed Stop Trial


  vec N=log_survival_s(sigma,g(tau, Ind_L, Ind_S_LCR ),g(tau_s, Ind_L, Ind_S_LCR ),g(SSD, Ind_L, Ind_S_LCR ),g(b_r, Ind_L, Ind_S_LCR ),
                       h(nu_r_p, Ind_S_LCR ),g(nu_r_s, Ind_L, Ind_S_LCR ),DEL,DEL_s,lt);


  vec O=log_survival_s(sigma,g(tau, Ind_L, Ind_S_LIR ),g(tau_s, Ind_L, Ind_S_LIR ),g(SSD, Ind_L, Ind_S_LIR ),g(b_l, Ind_L, Ind_S_LIR ),
                       g(nu_l, Ind_L, Ind_S_LIR ),g(nu_l_s, Ind_L, Ind_S_LIR ),DEL,DEL_s,lt);

  vec P=log_survival_s(sigma,g(tau, Ind_R, Ind_S_RCR ),g(tau_s, Ind_R, Ind_S_RCR ),g(SSD, Ind_R, Ind_S_RCR ),g(b_l, Ind_R, Ind_S_RCR ),
                       h(nu_l_p, Ind_S_RCR ),g(nu_l_s, Ind_R, Ind_S_RCR ),DEL,DEL_s,lt);

  vec Q=log_survival_s(sigma,g(tau, Ind_R, Ind_S_RIR ),g(tau_s, Ind_R, Ind_S_RIR ),g(SSD, Ind_R, Ind_S_RIR ),g(b_r, Ind_R, Ind_S_RIR ),
                       g(nu_r, Ind_R, Ind_S_RIR ),g(nu_r_s, Ind_R, Ind_S_RIR ),DEL,DEL_s,lt);



  //survival

  // Stop Trial

  // vec stop_param_i=stop_param.col(i);

  vec R_LCR=log_survival_stop(sigma,g(tau_s, Ind_L, Ind_S_LCR ),stop_param(0),stop_param(1));

  vec R_LIR=log_survival_stop(sigma,g(tau_s, Ind_L, Ind_S_LIR ),stop_param(0),stop_param(1));


  vec R_RCR=log_survival_stop(sigma,g(tau_s, Ind_R, Ind_S_RCR ),stop_param(0),stop_param(1));

  vec R_RIR=log_survival_stop(sigma,g(tau_s, Ind_R, Ind_S_RIR ),stop_param(0),stop_param(1));



  // Failed Stop Trial Combined


  vec X_LCR=(prob_param(2)+B+F);
  vec Y_LCR=(prob_param(1)+R_LCR+J+N);

  vec X_LIR=(prob_param(2)+C+G);
  vec Y_LIR=(prob_param(1)+R_LIR+K+O);

  vec X_RCR=(prob_param(2)+D+H);
  vec Y_RCR=(prob_param(1)+R_RCR+L+P);

  vec X_RIR=(prob_param(2)+E+I);
  vec Y_RIR=(prob_param(1)+R_RIR+M+Q);


  vec lk_LCR_FS=compute_log_add(X_LCR,Y_LCR);

  vec lk_LIR_FS=compute_log_add(X_LIR,Y_LIR);

  vec lk_RCR_FS=compute_log_add(X_RCR,Y_RCR);

  vec lk_RIR_FS=compute_log_add(X_RIR,Y_RIR);


  vec diff_X_LCR=exp(X_LCR- lk_LCR_FS);
  vec diff_Y_LCR=exp(Y_LCR- lk_LCR_FS);

  vec diff_X_LIR=exp(X_LIR- lk_LIR_FS);
  vec diff_Y_LIR=exp(Y_LIR- lk_LIR_FS);

  vec diff_X_RCR=exp(X_RCR- lk_RCR_FS);
  vec diff_Y_RCR=exp(Y_RCR- lk_RCR_FS);

  vec diff_X_RIR=exp(X_RIR- lk_RIR_FS);
  vec diff_Y_RIR=exp(Y_RIR- lk_RIR_FS);



  field<vec> result(12);

  result(0)=lk_LCR_FS;
  result(1)=lk_LIR_FS;

  result(2)=lk_RCR_FS;
  result(3)=lk_RIR_FS;

  result(4)=diff_X_LCR;
  result(5)=diff_Y_LCR;

  result(6)=diff_X_LIR;
  result(7)=diff_Y_LIR;

  result(8)=diff_X_RCR;
  result(9)=diff_Y_RCR;

  result(10)=diff_X_RIR;
  result(11)=diff_Y_RIR;


  return result;

}







///////////////////////////////////////////////////////////////////////////////// Gama Parameter related Vectors //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//Function-1


field <field<vec>> update_gama_param_ess1(const field <mat> &t,const field <vec> &Rand_gama_l,const field <vec> &Rand_gama_r,
                                          const mat &gama){

  // Rcpp::Rcout<<"P"<< endl;


  if (t(0).n_cols != gama.n_rows)
    Rcpp::stop("t(0).n_cols != gama.n_rows in --update_gama_param_ess1-- ");

  // vec gama_l=gama.col(0);
  // vec gama_r=gama.col(1);

  unsigned N=Rand_gama_l.n_elem;



  field <vec> b_l(N);
  field <vec> b_r(N);

  field <vec> b_l_main(N);
  field <vec> b_r_main(N);


  for (unsigned i = 0; i < N; ++i) {


    b_l_main(i) = (t(i) * gama.col(0));
    b_r_main(i) = (t(i) * gama.col(1));

    b_l(i) = b_l_main(i)+ Rand_gama_l(i);
    b_r(i) = b_r_main(i)+ Rand_gama_r(i);

  }


  field<field<vec>> result(4);

  result(0)=b_l;
  result(1)=b_r;

  result(2)=b_l_main;
  result(3)=b_r_main;

  return result;

}




// Function-2


field <field<vec>> update_gama_param_ess2(const field <vec> &tau, const double sigma, const field <vec> &b_l,const field <vec> &b_r,
                                          const field <vec> &nu_l,const field <vec> &nu_r,const field <vec> &nu_r_p,const field <vec> &nu_l_p,
                                          const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                                          const field <uvec> &Ind_LCR, const field <uvec> &Ind_LIR,const field <uvec>&Ind_RCR,const field <uvec>&Ind_RIR,
                                          const field <uvec> &Ind_S_LCR, const field <uvec> &Ind_S_LIR,const field <uvec>&Ind_S_RCR,const field <uvec>&Ind_S_RIR){


  // Rcpp::Rcout<<"Q"<< endl;

  unsigned N=tau.n_elem;

  field<vec>weight_LCR(N);
  field<vec>weight_LIR(N);
  field<vec>weight_RCR(N);
  field<vec>weight_RIR(N);


  field<vec>weight_LCR_G_FS(N);
  field<vec>weight_LIR_G_FS(N);
  field<vec>weight_RCR_G_FS(N);
  field<vec>weight_RIR_G_FS(N);

  for (unsigned i = 0; i < N; ++i) {


    weight_LCR(i)=weight(sigma,g(tau(i), Ind_L(i), Ind_LCR(i) ),h(nu_r_p(i), Ind_LCR(i) ),g(b_r(i), Ind_L(i), Ind_LCR(i) ));

    weight_LIR(i)=weight(sigma,g(tau(i), Ind_L(i), Ind_LIR(i) ),g(nu_l(i), Ind_L(i), Ind_LIR(i) ),
               g(b_l(i), Ind_L(i), Ind_LIR(i) ));

    weight_RCR(i)=weight(sigma,g(tau(i), Ind_R(i), Ind_RCR(i) ),h(nu_l_p(i), Ind_RCR(i) ),g(b_l(i), Ind_R(i), Ind_RCR(i) ));

    weight_RIR(i)=weight(sigma,g(tau(i), Ind_R(i), Ind_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_RIR(i) ),
               g(b_r(i), Ind_R(i), Ind_RIR(i) ));



    weight_LCR_G_FS(i)=weight(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p(i), Ind_S_LCR(i) ),g(b_r(i), Ind_L(i), Ind_S_LCR(i) ));

    weight_LIR_G_FS(i)=weight(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l(i), Ind_L(i), Ind_S_LIR(i) ),g(b_l(i), Ind_L(i), Ind_S_LIR(i) ));

    weight_RCR_G_FS(i)=weight(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p(i), Ind_S_RCR(i) ),g(b_l(i), Ind_R(i), Ind_S_RCR(i) ));

    weight_RIR_G_FS(i)=weight(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RIR(i) ),
                    g(b_r(i), Ind_R(i), Ind_S_RIR(i) ));


  }

  field<field<vec>> result(8);
  result(0)=weight_LCR;
  result(1)=weight_LIR;
  result(2)=weight_RCR;
  result(3)=weight_RIR;

  result(4)=weight_LCR_G_FS;
  result(5)=weight_LIR_G_FS;
  result(6)=weight_RCR_G_FS;
  result(7)=weight_RIR_G_FS;

  return result;


}



//Function-3


field <field<vec>> update_gama_param_ess3(const field <vec> &b_l_main,const field <vec> &b_r_main,
                                          const field <mat> &Phi,
                                          const field <mat> &gama_I){

  // Rcpp::Rcout<<"P"<< endl;

  unsigned N=b_l_main.n_elem;

  field <vec> b_l(N);
  field <vec> b_r(N);

  field <vec> Rand_gama_l(N);
  field <vec> Rand_gama_r(N);





  for (unsigned i = 0; i < N; ++i) {

    Rand_gama_l(i) = (Phi(i) * gama_I(i).col(0));
    Rand_gama_r(i) = (Phi(i) * gama_I(i).col(1));


    b_l(i) = b_l_main(i)+ Rand_gama_l(i);
    b_r(i) = b_r_main(i)+ Rand_gama_r(i);

  }

  field<field<vec>> result(4);
  result(0)=b_l;
  result(1)=b_r;

  result(2)=Rand_gama_l;
  result(3)=Rand_gama_r;

  return result;



}

//Function-4


field <field<vec>> update_gama_param_ess4(const field <mat> &t,const mat &gama){

  // Rcpp::Rcout<<"P"<< endl;

  unsigned N=t.n_elem;

  field <vec> b_l_main(N);
  field <vec> b_r_main(N);


  for (unsigned i = 0; i < N; ++i) {


    b_l_main(i) = (t(i) * gama.col(0));
    b_r_main(i) = (t(i) * gama.col(1));


  }


  field<field<vec>> result(2);

  result(0)=b_l_main;
  result(1)=b_r_main;


  return result;



}


//Function-5



field<mat> update_sq(const field <mat> &gama_I){

  // Rcpp::Rcout<<"P"<< endl;

  unsigned N=gama_I.n_elem;

  unsigned m=gama_I(0).n_rows;

  mat sq_l(m,N, arma::fill::zeros);
  mat sq_r(m,N, arma::fill::zeros);

  for (unsigned i = 0; i < N; ++i) {

    sq_l.col(i) = square (gama_I(i).col(0));
    sq_r.col(i) = square (gama_I(i).col(1));


  }

  field<mat> result(2);

  result(0)=sq_l;
  result(1)=sq_r;


  return result;


}




///////////////////////////////////////////////////////////////////////////////// Beta Parameter related Vectors /////////////////////////////////////////////////////////////////////////

// Function-1



field <field<vec>>  update_beta_param_ess1(const field <mat> &t,const field <vec> &Rand_beta_l,
                                           const field <vec> &Rand_beta_r,
                                           const mat &beta){

  // Rcpp::Rcout<<"R"<< endl;

  if (t(0).n_cols != beta.n_rows)
    Rcpp::stop("t(0).n_cols != beta.n_rows in --update_beta_param_ess1-- ");


  unsigned N=Rand_beta_l.n_elem;

  field <vec> nu_l(N);
  field <vec> nu_r(N);

  field <vec> nu_l_squared(N);
  field <vec> nu_r_squared(N);

  field <vec> nu_l_main(N);
  field <vec> nu_r_main(N);


  for (unsigned i = 0; i < N; ++i) {

    nu_l_main(i) = (t(i) * beta.col(0));
    nu_l(i) = nu_l_main(i)+ Rand_beta_l(i);
    nu_l_squared(i) = square(nu_l(i));

    nu_r_main(i) = (t(i) * beta.col(1));
    nu_r(i) = nu_r_main(i)+ Rand_beta_r(i);
    nu_r_squared(i) = square(nu_r(i));

  }



  field<field<vec>> result(6);
  result(0)=nu_l;
  result(1)=nu_r;
  result(2)=nu_l_squared;
  result(3)=nu_r_squared;
  result(4)=nu_l_main;
  result(5)=nu_r_main;


  return result;

}



// Function-2




field <field<vec>> update_beta_param_ess2(const field<vec> &tau,const double sigma, const field <vec> &b_l,const field <vec> &b_r,
                                          const field <vec> &nu_l,const field <vec> &nu_r,const field <vec> &nu_l_squared, const field <vec> &nu_r_squared,
                                          const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                                          const field <uvec> &Ind_LCR,const field <uvec> &Ind_LIR,const field <uvec>&Ind_RCR,const field <uvec>&Ind_RIR,
                                          const field <uvec> &Ind_S_LCR,const field <uvec> &Ind_S_LIR,const field <uvec>&Ind_S_RCR,const field <uvec>&Ind_S_RIR,

                                          const mat &penal_param){



  // Rcpp::Rcout<<"S"<< endl;

  unsigned N=tau.n_elem;

  field<vec>nu_l_p(N);
  field<vec>nu_r_p(N);
  field<vec>weight_LCR(N);
  field<vec>weight_LIR(N);
  field<vec>weight_RCR(N);
  field<vec>weight_RIR(N);

  field<vec>weight_LCR_G_FS(N);
  field<vec>weight_LIR_G_FS(N);
  field<vec>weight_RCR_G_FS(N);
  field<vec>weight_RIR_G_FS(N);


  for (unsigned i = 0; i < N; ++i) {

    nu_l_p(i) = h(nu_l(i), Ind_R(i) ) - penalty_fun(h(nu_r(i), Ind_R(i) ),h(nu_r_squared(i), Ind_R(i) ),penal_param.col(i));
    nu_r_p(i) = h(nu_r(i), Ind_L(i) ) - penalty_fun(h(nu_l(i), Ind_L(i) ),h(nu_l_squared(i), Ind_L(i) ),penal_param.col(i));


    weight_LCR(i)=weight(sigma,g(tau(i), Ind_L(i), Ind_LCR(i) ),h(nu_r_p(i),Ind_LCR(i) ),g(b_r(i), Ind_L(i), Ind_LCR(i) ));
    weight_LIR(i)=weight(sigma,g(tau(i), Ind_L(i), Ind_LIR(i) ),g(nu_l(i),Ind_L(i),Ind_LIR(i) ),g(b_l(i), Ind_L(i), Ind_LIR(i) ));
    weight_RCR(i)=weight(sigma,g(tau(i), Ind_R(i), Ind_RCR(i) ),h(nu_l_p(i), Ind_RCR(i) ),g(b_l(i), Ind_R(i), Ind_RCR(i) ));
    weight_RIR(i)=weight(sigma,g(tau(i), Ind_R(i), Ind_RIR(i) ),g(nu_r(i),Ind_R(i), Ind_RIR(i) ),g(b_r(i), Ind_R(i), Ind_RIR(i) ));


    weight_LCR_G_FS(i)=weight(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p(i), Ind_S_LCR(i) ),g(b_r(i), Ind_L(i), Ind_S_LCR(i) ));

    weight_LIR_G_FS(i)=weight(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l(i), Ind_L(i), Ind_S_LIR(i) ),g(b_l(i), Ind_L(i), Ind_S_LIR(i) ));

    weight_RCR_G_FS(i)=weight(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p(i), Ind_S_RCR(i) ),g(b_l(i), Ind_R(i), Ind_S_RCR(i) ));

    weight_RIR_G_FS(i)=weight(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RIR(i) ),
                    g(b_r(i), Ind_R(i), Ind_S_RIR(i) ));




  }

  field<field<vec>> result(10);
  result(0)=nu_l_p;
  result(1)=nu_r_p;
  result(2)=weight_LCR;
  result(3)=weight_LIR;
  result(4)=weight_RCR;
  result(5)=weight_RIR;

  result(6)=weight_LCR_G_FS;
  result(7)=weight_LIR_G_FS;
  result(8)=weight_RCR_G_FS;
  result(9)=weight_RIR_G_FS;




  return result;


}


// Function-3



field <field<vec>> update_beta_param_ess3(const field <vec> &nu_l,const field <vec> &nu_r,const field <vec> &nu_l_squared, const field <vec> &nu_r_squared,
                                          const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                                          const mat &penal_param){



  // Rcpp::Rcout<<"S"<< endl;

  unsigned N=nu_l.n_elem;

  field<vec>nu_l_p(N);
  field<vec>nu_r_p(N);

  for (unsigned i = 0; i < N; ++i) {

    nu_l_p(i) = h(nu_l(i), Ind_R(i) ) - penalty_fun(h(nu_r(i), Ind_R(i) ),h(nu_r_squared(i), Ind_R(i) ),penal_param.col(i));
    nu_r_p(i) = h(nu_r(i), Ind_L(i) ) - penalty_fun(h(nu_l(i), Ind_L(i) ),h(nu_l_squared(i), Ind_L(i) ),penal_param.col(i));


  }

  field<field<vec>> result(2);
  result(0)=nu_l_p;
  result(1)=nu_r_p;
  return result;


}




// Function-4



field <field<vec>>  update_beta_param_ess4(const field <vec> &nu_l_main,const field <vec> &nu_r_main,
                                           const field <mat> &Phi,
                                           const field <mat> &beta_I){

  // Rcpp::Rcout<<"R"<< endl;

  unsigned N=nu_l_main.n_elem;

  field <vec> nu_l(N);
  field <vec> nu_r(N);

  field <vec> nu_l_squared(N);
  field <vec> nu_r_squared(N);


  field <vec> Rand_beta_l(N);
  field <vec> Rand_beta_r(N);


  for (unsigned i = 0; i < N; ++i) {


    // vec beta_I_l=beta_I(i).col(0);
    // vec beta_I_r=beta_I(i).col(1);

    Rand_beta_l(i) = (Phi(i) * beta_I(i).col(0));
    nu_l(i) = nu_l_main(i)+ Rand_beta_l(i);
    nu_l_squared(i) = square(nu_l(i));

    Rand_beta_r(i) = (Phi(i) * beta_I(i).col(1));
    nu_r(i) = nu_r_main(i)+ Rand_beta_r(i);
    nu_r_squared(i) = square(nu_r(i));

  }



  field<field<vec>> result(6);
  result(0)=nu_l;
  result(1)=nu_r;
  result(2)=nu_l_squared;
  result(3)=nu_r_squared;
  result(4)=Rand_beta_l;
  result(5)=Rand_beta_r;


  return result;

}



// Function-5



field <field<vec>>  update_beta_param_ess5(const field <mat> &t,const mat &beta){

  // Rcpp::Rcout<<"R"<< endl;


  unsigned N=t.n_elem;


  field <vec> nu_l_main(N);
  field <vec> nu_r_main(N);


  for (unsigned i = 0; i < N; ++i) {


    nu_l_main(i) = (t(i) * beta.col(0));

    nu_r_main(i) = (t(i) * beta.col(1));

  }



  field<field<vec>> result(2);
  result(0)=nu_l_main;
  result(1)=nu_r_main;


  return result;

}

///////////////////////////////////////////////////////////////////////////////// Expit Function /////////////////////////////////////////////////////////////////////////




inline vec expit(vec x) {
  arma::vec out(x.n_elem);
    for (arma::uword i = 0; i < x.n_elem; ++i) {
        double v = x(i);
        if (v >= 0.0) {
            double z = std::exp(-v);
            out(i) = 1.0 / (1.0 + z);
        } else {
            double z = std::exp(v);
            out(i) = z / (1.0 + z);
        }
    }
    return out;
}


///////////////////////////////////////////////////////////////////////////////// Delta_prime Parameter related Vectors /////////////////////////////////////////////////////////////////////////


//Function-1

field <field<vec>> update_delta_prime_param_ess(const field<vec> &tau,const field<vec> &tau_stop,const field<vec> &lower_bound_init,
                                                const vec &DEL,const vec &DEL_s){



  // Rcpp::Rcout<<"T"<< endl;
  unsigned N=tau.n_elem;


  field<vec>tau_prime(N);

  field<vec>tau_s(N);

  field<vec>lower_bound(N);


  for (unsigned i = 0; i < N; ++i) {

    lower_bound(i) = lower_bound_init(i) + DEL_s(i);

    tau_s(i) = vec(tau_stop(i).n_elem, fill::zeros);
    tau_prime(i) = vec(tau(i).n_elem, fill::zeros);

    // Apply update only where tau_stop > 0
    uvec idx_stop_pos = find(tau_stop(i) > 0);
    tau_s(i).elem(idx_stop_pos) = tau_stop(i).elem(idx_stop_pos) - DEL_s(i);

    // Apply update only where tau > 0
    uvec idx_tau_pos = find(tau(i) > 0);
    tau_prime(i).elem(idx_tau_pos) = tau(i).elem(idx_tau_pos) - DEL(i);


// ===== Check tau_prime =====
const vec  &tp = tau_prime(i);
const vec  &t  = tau(i);

uvec bad_neg_p = find(tp < 0);
uvec bad_nf_p  = find_nonfinite(tp);
uvec bad_idx_p = unique(join_cols(bad_neg_p, bad_nf_p));

if (!bad_idx_p.is_empty()) {
  Rcpp::Rcout << "WARNING: Invalid tau_prime at i=" << i 
              << " (DEL=" << DEL(i) << ")\n";
  for (uword k = 0; k < bad_idx_p.n_elem; ++k) {
    uword j = bad_idx_p(k);
    Rcpp::Rcout << "  j=" << j
                << "  tau_prime=" << tp(j)   // bad value
                << "  tau=" << t(j)          // original tau
                << "  DEL=" << DEL(i)        // scalar used
                << "\n";
  }
}

  }


   


  field<field<vec>> result(3);
  result(0)=tau_prime;
  result(1)=tau_s;
  result(2)=lower_bound;
  return result;

}

///////////////////////////////////////////////////////////////////////////////// Stop Process related Vectors /////////////////////////////////////////////////////////////////////////

//Funciton-1

field <field<vec>> update_stop_param_ess(const field<vec> &tau,const field<vec> &tau_s,const double sigma,const field <vec> &SSD,
                                         const mat &stop_param,
                                         const field <vec> &b_l,const field <vec> &b_r,
                                         const field <vec> &nu_l,const field <vec> &nu_r,
                                         const field <vec> &nu_r_p, const field <vec> &nu_l_p,
                                         const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                                         const field <uvec> &Ind_S_LCR,const field <uvec> &Ind_S_LIR,
                                         const field <uvec>&Ind_S_RCR,const field <uvec>&Ind_S_RIR,
                                         const mat &penal_param,
                                         const vec &DEL, const vec &DEL_s,const bool lt = 1){



  // Rcpp::Rcout<<"U"<< endl;

  unsigned N=b_l.n_elem;

  field<vec>nu_l_s(N);
  field<vec>nu_r_s(N);


  field<vec>weight_LCR_S(N);
  field<vec>weight_LIR_S(N);
  field<vec>weight_RCR_S(N);
  field<vec>weight_RIR_S(N);


  for (unsigned i = 0; i < N; ++i) {


    // vec stop_param_i=stop_param.col(i);

    nu_l_s(i)=nu_l(i)-penalty_fun_s(stop_param(1,i),penal_param.col(i));
    nu_r_s(i)=nu_r(i)-penalty_fun_s(stop_param(1,i),penal_param.col(i));


    weight_LCR_S(i)=weight_s(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),g(SSD(i), Ind_L(i), Ind_S_LCR(i) ),
                 g(b_r(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p(i), Ind_S_LCR(i) ),
                 g(nu_r_s(i), Ind_L(i), Ind_S_LCR(i) ),DEL(i),DEL_s(i),lt);

    weight_LIR_S(i)=weight_s(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),g(SSD(i), Ind_L(i), Ind_S_LIR(i) ),
                 g(b_l(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l(i),Ind_L(i), Ind_S_LIR(i) ),
                 g(nu_l_s(i), Ind_L(i), Ind_S_LIR(i) ),DEL(i),DEL_s(i),lt);

    weight_RCR_S(i)=weight_s(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),g(SSD(i), Ind_R(i), Ind_S_RCR(i) ),
                 g(b_l(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p(i), Ind_S_RCR(i) ),
                 g(nu_l_s(i),Ind_R(i), Ind_S_RCR(i) ),DEL(i),DEL_s(i),lt);


    weight_RIR_S(i)=weight_s(sigma, g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),g(SSD(i), Ind_R(i), Ind_S_RIR(i) ),
                 g(b_r(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RIR(i) ),
                 g(nu_r_s(i),Ind_R(i), Ind_S_RIR(i) ),DEL(i),DEL_s(i),lt);




  }




  field<field<vec>> result(6);
  result(0)=nu_l_s;
  result(1)=nu_r_s;
  result(2)=weight_LCR_S;
  result(3)=weight_LIR_S;
  result(4)=weight_RCR_S;
  result(5)=weight_RIR_S;



  return result;


}




//Funciton-2

field <field<vec>> update_stop_param_ess2(const field <vec> &nu_l,const field <vec> &nu_r,const mat &stop_param,
                                          const mat &penal_param){



  // Rcpp::Rcout<<"U"<< endl;

  unsigned N=nu_l.n_elem;


  field<vec>nu_l_s(N);
  field<vec>nu_r_s(N);


  for (unsigned i = 0; i < N; ++i) {



    // vec stop_param_i=stop_param.col(i);

    nu_l_s(i)=nu_l(i)-penalty_fun_s(stop_param(1,i),penal_param.col(i));
    nu_r_s(i)=nu_r(i)-penalty_fun_s(stop_param(1,i),penal_param.col(i));





  }



  field<field<vec>> result(2);
  result(0)=nu_l_s;
  result(1)=nu_r_s;

  return result;


}






///////////////////////////////////////////////////////////////////////////////// Numerical Integration related Vectors ///////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////// Integral Versions///////////////////////////////////////////////////////////////////////





// log density (Stop Trial-Integral Version)

vec log_dens_stop_I(double sigma, double SSD, const vec &u,
                    const double DEL_s, double b_stop, double nu_stop) {


  vec u_s = u - SSD- DEL_s ;

  vec result(u.n_elem);

  for (uword i = 0; i < u.n_elem; ++i) {
    if (u_s(i) <= 0) {
      result(i) = trunc_log(0);
    } else {
      double DF = (exp(b_stop)) - (exp(nu_stop) * u_s(i));
      result(i) = b_stop - ( log(2 * datum::pi * sigma)/ 2)
        - (1.5 * log(u_s(i)))
        - (pow(DF, 2) / (2 * sigma * u_s(i)));
    }
  }

  if (result.has_nan()){
    Rcpp::Rcout<<"nan in log_dens_stop"<<endl;
  }

  return result;
}





// Exponent1 function

inline vec exponent_1_I(const double sigma,const vec &u,const vec &m,const double b){

  return (exp(b)-m)/(sqrt(sigma*u));
}


// log density for Failed Stop trial (u<ssd)



vec log_density_I(const double sigma, const vec &u, const double b,const vec &m) {

  // Rcpp::Rcout<<"A"<< endl;

  uvec Ind=find(u<=0);



  vec x = exponent_1_I(sigma,u,m,b);

  vec result = b-(log(sigma)/2)- ((3 * trunc_log(u)) / 2)+
               log_normpdf(x);

  result(Ind).fill(trunc_log(0)); // If u is negative or <=0,






  return  result;

}



// Exponent2 function

inline vec exponent_2_I(const double sigma,const vec &u,const vec &m,const double b){


  return (-exp(b)-m)/(sqrt(sigma*u));
}

// phi_rat function

vec phi_rat_I(const vec &q1,const bool lt = 1){

  vec y1 = q1;
  y1.transform([&lt](double E) { return R::pnorm(E, 0, 1, lt, 1); });

  vec norm_dens=log_normpdf(q1);

  return (exp(y1-norm_dens));



}



// SSD_Dif function

inline vec SSD_Dif_I(const double sigma,const vec &u,double SSD,const double nu1,const double nu2,const double DEL,const double DEL_s){

  return ((2*(SSD+DEL_s-DEL)*(exp(nu2)-exp(nu1)))/(sqrt(sigma*u)));
  // return (((SSD+DEL_s-DEL)*(exp(nu2)-exp(nu1)))/(sqrt(sigma*u)));



}




// log density (Go process-Failed Stop trial)

vec log_dens_s_I(const double sigma,const vec &u,const double SSD,const double DEL,double DEL_s,const double b, const double nu1,
                 const double nu2,const bool lt = 1){

  vec u_d=u-DEL;

  vec u_s=u-SSD-DEL_s;

  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);

  vec log_dens=log_density_I(sigma,u_d,b,m);

  vec diff=SSD_Dif_I(sigma,u_d,SSD,nu1,nu2,DEL,DEL_s);

  vec q2 = exponent_2_I(sigma,u_d,m,b);

  vec phi_ratio=phi_rat_I(q2,lt);

  vec val=phi_ratio%diff;

  val.clamp(-0.999,datum::inf);


  vec lp = log1p(val);

  vec result= (log_dens+lp);

  return result;

}




// log Survival (Failed Stop Trial-Integral Version)

vec log_survival_s_I(const double sigma, const vec &u,const double SSD,double DEL,double DEL_s,
                     const double b,const double nu1,
                     const double nu2,const bool lt = 1) {

  // Rcpp::Rcout << "D" << std::endl;

  vec u_d=u-DEL;

  vec u_s = u - SSD - DEL_s;

  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);

  vec q1 = exponent_1_I(sigma,u_d,m,b);

  vec y1 = q1;
  y1.transform([&lt](double E) { return R::pnorm(E, 0, 1, lt, 1); });


  vec C=(2*exp(b)*m)/(sigma*u_d);

  vec q2 =exponent_2_I(sigma,u_d,m,b);

  vec y2 = q2;
  y2.transform([&lt](double F) { return R::pnorm(F, 0, 1, lt, 1); });


  vec y3 = C + y2;

  vec val = -exp(y3 - y1);

  val.clamp(-0.999,datum::inf);

  vec result= (y1 + log1p(val));

  return result;
}





///////////////////////////////////////////////////////////////////////////////////////////////likelihood /////////////////////////////////////////////////////////////////////////////////////////////////


// Integrand function (Trapizoidal Integration)

vec IntegrandFunc_likelihood(const vec &u, double sigma,double SSD,double DEL,double DEL_s, double b_stop, double nu_stop,
                             double b_CR,double nu1_CR,const double nu2_CR,
                             double b_IR,double nu1_IR,const double nu2_IR, const bool lt = 1 ) {


  vec log_dens = log_dens_stop_I(sigma,SSD,u,DEL_s,b_stop,nu_stop);
  vec log_surv_CR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);
  vec log_surv_IR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec result = exp(log_dens + log_surv_CR + log_surv_IR);

  return result;
}





// Integration using Trapezoidal Integration

double integrate_integral_likelihood(double sigma,double SSD,double DEL,double DEL_s,
                                     const vec &stop_param,
                                     double b_CR,double nu_1_CR, double nu_2_CR,
                                     double b_IR,double nu_1_IR, double nu_2_IR, double lower_bound, double upper_bound,
                                     bool lt = 1) {

  // Generate the u values
  double lower_bound_new=lower_bound+0.00004;

  int n_points=100;
  vec u = linspace(lower_bound_new, upper_bound, n_points);

  // Compute the integrand values
  vec integrand_values = IntegrandFunc_likelihood(u,sigma,SSD,DEL,DEL_s,stop_param(0),stop_param(1),
                                                  b_CR,nu_1_CR,nu_2_CR,b_IR,nu_1_IR,nu_2_IR,lt);


  // Perform the trapezoidal integration
  mat integral = trapz(u, integrand_values);

  double res = integral(0,0);
  return res;
}



// Saving the Integral of likelihood

field<vec> integral_likelihood(const double sigma,
                               const vec &SSD,const double DEL,const double DEL_s,
                               const vec &stop_param,const vec &penal_param,const vec &prob_param,
                               const vec &b_l,const vec &nu_l,const vec &nu_l_s,
                               const vec &b_r,const vec &nu_r_p,const vec &nu_r_s,
                               const vec &nu_r,const vec &nu_l_p,
                               const uvec &Ind_L,const uvec &Ind_R,const uvec &Ind_I_L,const uvec &Ind_I_R,
                               const vec &lower_bound,const double upper_bound,const bool lt = true) {


  // Rcpp::Rcout<<"Y"<< endl;

  vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );
  vec SSD_I_R=g(SSD, Ind_R, Ind_I_R );

  vec b_l_I_L=g(b_l, Ind_L, Ind_I_L );
  vec b_l_I_R=g(b_l, Ind_R, Ind_I_R );

  vec b_r_I_L=g(b_r, Ind_L, Ind_I_L );
  vec b_r_I_R=g(b_r, Ind_R, Ind_I_R );

  vec nu_l_I_L=g(nu_l, Ind_L, Ind_I_L );
  vec nu_r_I_R=g(nu_r, Ind_R, Ind_I_R );

  vec nu_l_s_I_L=g(nu_l_s, Ind_L, Ind_I_L );
  vec nu_l_s_I_R=g(nu_l_s, Ind_R, Ind_I_R );

  vec nu_r_s_I_L=g(nu_r_s, Ind_L, Ind_I_L );
  vec nu_r_s_I_R=g(nu_r_s, Ind_R, Ind_I_R );

  vec nu_r_p_I_L=h(nu_r_p, Ind_I_L );
  vec nu_l_p_I_R=h(nu_l_p, Ind_I_R );

  vec lower_bound_I_L=g(lower_bound, Ind_L, Ind_I_L );
  vec lower_bound_I_R=g(lower_bound, Ind_R, Ind_I_R );


  // Initialize result vector
  vec Stim_L(b_l_I_L.size());


  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_l_I_L.size(); ++i) {
    // Perform integration for current parameter values

    double int_dens=trunc_log(integrate_integral_likelihood(sigma,SSD_I_L(i),DEL,DEL_s,stop_param,
                                                            b_l_I_L(i),nu_l_I_L(i), nu_l_s_I_L(i),
                                                            b_r_I_L(i),nu_r_p_I_L(i),nu_r_s_I_L(i),
                                                            lower_bound_I_L(i),upper_bound,lt ));




    // cout<<"int_dens: "<<int_dens<<endl;

    Stim_L[i] = mlpack::LogAdd(prob_param(0),(prob_param(1)+int_dens));

  }


  // Initialize result vector
  vec Stim_R(b_r_I_R.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_r_I_R.size(); ++i) {
    // Perform integration for current parameter values

    double int_dens=trunc_log(integrate_integral_likelihood(sigma,SSD_I_R(i),DEL,DEL_s,stop_param,
                                                            b_r_I_R(i),nu_r_I_R(i),nu_r_s_I_R(i),
                                                            b_l_I_R(i),nu_l_p_I_R(i),nu_l_s_I_R(i),
                                                            lower_bound_I_R(i),upper_bound,lt));
    // cout<<"int_dens2: "<<int_dens<<endl;

    Stim_R[i] =mlpack::LogAdd(prob_param(0),(prob_param(1)+int_dens));


  }



  field<vec> result(2);

  result(0)=Stim_L;
  result(1)=Stim_R;

  return result;
}



////////////////////////////////////////////////////////////////////////////////////gama/////////////////////////////////////////////////////////////////////////


// deriv_multiplier

inline vec deriv_multiplier_I(const vec &SSD_diff,const vec &rat,const vec &q2){

  vec result=SSD_diff%(1+(rat%q2));

  return result;
}




// derivative of exponent w.r.t. gama

inline vec deriv_exp_gama_I(double sigma,const vec &u, const double &b){

  return (exp(b)/(sqrt(sigma*u)));

}




// Derivative of log dense w.r.t. gama (Go Proces-Failed Stop Trial)


vec deriv_log_dens_gama_s_I(const vec &u,const double sigma,
                            const double SSD,const double DEL, const double DEL_s,
                            const double b,const double nu1,const double nu2,const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;

  vec u_d=u-DEL;
  vec u_s=u-SSD-DEL_s;


  // Defining necessary vectors
  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);

  vec q1=exponent_1_I(sigma,u_d,m,b);

  vec q2 = exponent_2_I(sigma,u_d,m,b);

  vec rat=phi_rat(q2,lt);

  vec SSD_diff=SSD_Dif_I(sigma,u_d,SSD,nu1,nu2,DEL,DEL_s);

  vec deriv_gama=deriv_exp_gama_I(sigma,u_d,b);

  // calculating small vectors

  vec c1=q1%deriv_gama;

  vec deriv_gama_B=deriv_multiplier_I(SSD_diff,rat,q2)%(-deriv_gama);

  vec B=1+(rat%SSD_diff);

  return (1-c1+(deriv_gama_B/B));

}




// Integrand function (Trapezoidal Integration)
vec IntegrandFunc_gama(const vec &u,double sigma, double SSD,double DEL,double DEL_s, double b_stop, double nu_stop,
                       double b_CR,double nu1_CR,const double nu2_CR,
                       double b_IR,double nu1_IR,const double nu2_IR,
                       const bool lt=1) {


  // Rcpp::Rcout<<"AAAE"<< endl;

  vec dens = log_dens_stop_I(sigma,SSD,u,DEL_s,b_stop,nu_stop);
  vec surv_CR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);
  vec surv_IR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec A = exp(dens + surv_CR + surv_IR);

  vec log_dens_s =log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);

  vec W_L = exp((2*log_dens_s) - surv_CR);

  vec Deriv_gama = deriv_log_dens_gama_s_I(u,sigma,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);


  /*vec weight_grad_gama = W_L % Deriv_gama;

   return A % (-weight_grad_gama);*/

  return - (A% W_L % Deriv_gama);

}





// Integration (Trapezoidal Integration)


double  integrate_integral_gama(double sigma,double SSD,double DEL,double DEL_s,
                                const vec &stop_param,
                                double b_CR,double nu1_CR,const double nu2_CR,
                                double b_IR,double nu1_IR,const double nu2_IR,
                                double lower_bound,double upper_bound,double lt=1) {



  // Rcpp::Rcout<<"AAAB"<< endl;


  double lower_bound_new=lower_bound+0.00004;

  unsigned  n_points=100;

  vec u = linspace(lower_bound_new,upper_bound, n_points);


  // mat integrand_values_mat(n_points, 2);

  // Compute the integrand values
  vec integrand_values = IntegrandFunc_gama(u,sigma,DEL,DEL_s,stop_param(0),stop_param(1),
                                            b_CR,nu1_CR,nu2_CR,b_IR,nu1_IR,nu2_IR,lt);


  // Perform the trapezoidal integration
  mat integral = trapz(u, integrand_values);




  double res=integral(0,0);

  return res;
}



// Integral (Left Stimulus)

field<vec> integrate_gama_Left_Stim(const field<vec> &integral_likelihood, const double sigma,
                                    const vec &SSD,const double DEL,const double DEL_s,
                                    const vec &stop_param,const vec &penal_param,const vec &prob_param,
                                    const vec &b_l, const vec &nu_l,const vec &nu_l_s,
                                    const vec &b_r, const vec &nu_r_p,const vec &nu_r_s,
                                    const uvec &Ind_L,const uvec &Ind_I_L,
                                    const vec lower_bound,const double upper_bound,const bool lt = true) {


  // Rcpp::Rcout<<"gama_left"<<endl;


  // Rcpp:: vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );

  vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );

  vec b_l_I_L=g(b_l, Ind_L, Ind_I_L );

  vec b_r_I_L=g(b_r, Ind_L, Ind_I_L );

  vec nu_l_I_L=g(nu_l, Ind_L, Ind_I_L );

  vec nu_l_s_I_L=g(nu_l_s, Ind_L, Ind_I_L );

  vec nu_r_s_I_L=g(nu_r_s, Ind_L, Ind_I_L );

  vec nu_r_p_I_L=h(nu_r_p, Ind_I_L );

  vec lower_bound_I_L=g(lower_bound, Ind_L, Ind_I_L );



  // Initialize result vectors
  vec Gama_L(b_l_I_L.size());
  vec Gama_R(b_r_I_L.size());


  // Parallelized loop for Gama_L
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_l_I_L.size(); ++i) {
    Gama_L[i] =  integrate_integral_gama(sigma, SSD_I_L[i],DEL,DEL_s, stop_param,
                                         b_l_I_L[i], nu_l_I_L[i],nu_l_s_I_L[i],
                                         b_r_I_L[i], nu_r_p_I_L[i],nu_r_s_I_L[i],
                                         lower_bound_I_L[i], upper_bound,lt);


    Gama_R[i] =  integrate_integral_gama(sigma,SSD_I_L[i],DEL,DEL_s, stop_param,
                                         b_r_I_L[i], nu_r_p_I_L[i],nu_r_s_I_L[i],
                                         b_l_I_L[i], nu_l_I_L[i],nu_l_s_I_L[i],
                                         lower_bound_I_L[i], upper_bound,lt);


  }





  vec A_L = exp(prob_param(1)-integral_likelihood(0));
  vec Gama_L_final = Gama_L % A_L;
  vec Gama_R_final = Gama_R % A_L;

  field<vec> result(2);
  result(0) = Gama_L_final;
  result(1) = Gama_R_final;

  return result;
}




// Integral (Right Stimulus)


field<vec> integrate_gama_Right_Stim(const field<vec> &integral_likelihood,const double sigma,
                                     const vec &SSD,const double DEL,const double DEL_s,
                                     const vec &stop_param,const vec &penal_param,const vec &prob_param,
                                     const vec &b_l, const vec &nu_l_p,const vec &nu_l_s,
                                     const vec &b_r, const vec &nu_r,const vec &nu_r_s,
                                     const uvec &Ind_R,const uvec &Ind_I_R,
                                     const vec &lower_bound,const double upper_bound,const bool lt = true){



  // Rcpp::Rcout<<"gama_right"<<endl;

  // Extract relevant subsets based on indices

  vec SSD_I_R=g(SSD, Ind_R, Ind_I_R );

  vec b_l_I_R=g(b_l, Ind_R, Ind_I_R );

  vec b_r_I_R=g(b_r, Ind_R, Ind_I_R );

  vec nu_r_I_R=g(nu_r, Ind_R, Ind_I_R );

  vec nu_l_s_I_R=g(nu_l_s, Ind_R, Ind_I_R );

  vec nu_r_s_I_R=g(nu_r_s, Ind_R, Ind_I_R );

  vec nu_l_p_I_R=h(nu_l_p, Ind_I_R );

  vec lower_bound_I_R=g(lower_bound, Ind_R, Ind_I_R );


  // Initialize result vectors
  vec Gama_L(b_r_I_R.size());
  vec Gama_R(b_r_I_R.size());



  // Parallelized loop for Gama_L
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_r_I_R.size(); ++i) {
    // Perform integration for current parameter values
    Gama_L[i] =  integrate_integral_gama(sigma, SSD_I_R[i],DEL,DEL_s, stop_param,
                                         b_l_I_R[i], nu_l_p_I_R[i],nu_l_s_I_R[i],
                                         b_r_I_R[i],nu_r_I_R[i], nu_r_s_I_R[i],
                                         lower_bound_I_R[i], upper_bound);


    Gama_R[i] =  integrate_integral_gama(sigma, SSD_I_R[i],DEL,DEL_s, stop_param,
                                         b_r_I_R[i],nu_r_I_R[i], nu_r_s_I_R[i],
                                         b_l_I_R[i],nu_l_p_I_R[i], nu_l_s_I_R[i],
                                         lower_bound_I_R[i], upper_bound);

  }


  // Compute final results
  vec A_R = exp(prob_param(1)-integral_likelihood(1));
  vec Gama_L_final = Gama_L % A_R;
  vec Gama_R_final = Gama_R % A_R;

  // Prepare and return the result field
  field<vec> result(2);
  result(0) = Gama_L_final;
  result(1) = Gama_R_final;

  return result;
}


//////////////////////////////////////////////////////////////////////////////// Beta //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// derivative of exponent w.r.t. beta (without penalty-Integral Version)

inline vec deriv_exp_beta_I(double sigma,const vec &u,const vec &m){

  return (m/(sqrt(sigma*u)));

}




// beta derivative (without penalty) (Go Process-Stop trial-Integral Version)

vec deriv_log_dens_beta_s_I(const double sigma,const vec &u,
                            double SSD,double DEL,double DEL_s, const double b,const double nu1,const double nu2,const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;

  vec u_d=u-DEL;
  vec u_s=u-SSD-DEL_s;


  // Defining necessary vectors
  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);
  vec q1=exponent_1_I(sigma,u_d,m,b);
  vec q2 = exponent_2_I(sigma,u_d,m,b);
  vec rat=phi_rat_I(q2,lt);

  vec SSD_diff=SSD_Dif_I(sigma,u_d,SSD,nu1,nu2,DEL,DEL_s);

  vec deriv_beta=deriv_exp_beta_I(sigma,u_d,m);


  vec c1=q1%deriv_beta;

  vec deriv_beta_B=(deriv_multiplier_I(SSD_diff,rat,q2)%(-deriv_beta))+(rat%SSD_diff);

  vec B=1+(rat%SSD_diff);

  vec result= (c1+(deriv_beta_B/B));


  return result;

}





// Derivative of penalty w.r.t beta (Integral Version)


double g_beta_stim_penal_I(double nu,double nu_squared,double lambda_prime,double alpha_prime){


  // Rcpp::Rcout<<"AAAG"<< endl;


  double X=(exp(2*lambda_prime))*nu_squared;
  double Y=exp(2*alpha_prime);
  double hpot=sqrt(X+Y);

  double numerator = exp(lambda_prime)-((exp(2*lambda_prime)*nu)/hpot);

  double denominator=pow((-(exp(lambda_prime)*nu)+hpot),2);

  return numerator/denominator;


}

//derivative of exponent w.r.t. beta (with penalty)

inline vec deriv_exp_beta_p_I(const double sigma,const vec &u,const double SSD,const double nu1,const double nu_stim,
                              const double nu_squared,double lambda_prime,double alpha_prime,const double DEL, const double DEL_s){

  return (((SSD+DEL_s-DEL)*exp(nu1)*g_beta_stim_penal_I(nu_stim,nu_squared,lambda_prime,alpha_prime))/(sqrt(sigma*u)));

}


// beta derivative (without penalty) (Go Proces-Failed Stop Trial)

vec deriv_log_dens_beta_p_s_I(const double sigma,const vec &u,
                              const double SSD,const double DEL,const double DEL_s, const double b,const double nu1,const double nu2,
                              const double nu_stim, const double nu_squared,
                              double lambda_prime,double alpha_prime,const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;

  vec u_d=u-DEL;

  vec u_s=u-SSD-DEL_s;


  // Defining necessary vectors

  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);

  vec q1=exponent_1_I(sigma,u_d,m,b);

  vec q2 = exponent_2_I(sigma,u_d,m,b);

  vec rat=phi_rat_I(q2,lt);

  vec SSD_diff=SSD_Dif_I(sigma,u_d,SSD,nu1,nu2,DEL,DEL_s);

  vec deriv_beta=deriv_exp_beta_p_I(sigma,u_d,SSD,nu1,nu_stim,nu_squared,lambda_prime,alpha_prime,DEL,DEL_s);


  // calculating small vectors

  vec deriv_beta_B=deriv_multiplier_I(SSD_diff,rat,q2)+(2*rat);

  vec B=1+(rat%SSD_diff);

  vec result= (-q1+(deriv_beta_B/B))%deriv_beta;

  return result;


}



//  Integrand function (Trapezoidal Integration)

vec IntegrandFunc_beta_CR(double sigma,const vec &u,double SSD,double DEL,double DEL_s,
                          double b_stop,double nu_stop,
                          double b_CR,double nu1_CR,const double nu2_CR,
                          double b_IR,double nu1_IR,const double nu2_IR,
                          const double nu_stim, const double nu_squared,
                          double lambda_prime,double alpha_prime,const bool lt=1){


  vec dens = log_dens_stop_I(sigma,SSD,u,DEL_s,b_stop,nu_stop);
  vec surv_CR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);
  vec surv_IR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);



  vec A=exp(dens+surv_CR+surv_IR);


  vec dens_s_CR=log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);


  vec W_CR=exp((2*dens_s_CR)-surv_CR);

  vec Deriv_beta_CR=deriv_log_dens_beta_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);


  vec weight_grad_beta_CR=W_CR%Deriv_beta_CR;

  vec dens_s_IR=log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec W_IR=exp((2*dens_s_IR)-surv_IR);

  vec Deriv_beta_IR=deriv_log_dens_beta_p_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,
                                              nu_stim,nu_squared,lambda_prime,alpha_prime,lt);


  vec weight_grad_beta_IR=W_IR%Deriv_beta_IR;


  vec result=A%(-weight_grad_beta_CR-weight_grad_beta_IR);

  return result;
}





// Integration (Trapezoidal Integration)


double  integrate_integral_beta_CR(double sigma,double SSD,double DEL,double DEL_s,
                                   const vec &stop_param,
                                   double b_CR,double nu1_CR,const double nu2_CR,
                                   double b_IR,double nu1_IR,const double nu2_IR,
                                   const double nu_stim, const double nu_squared,
                                   const vec &penal_param,double lower_bound,double upper_bound,const bool lt=1){



  double lower_bound_new=lower_bound+0.00004;

  double  n_points=100;

  vec u = linspace(lower_bound_new,upper_bound, n_points);


  // Compute the integrand values
  vec integrand_values =IntegrandFunc_beta_CR(sigma,u,SSD,DEL,DEL_s,stop_param(0),stop_param(1),
                                              b_CR,nu1_CR,nu2_CR,
                                              b_IR,nu1_IR,nu2_IR,
                                              nu_stim,nu_squared,
                                              penal_param(0),penal_param(1),lt);;


  // Perform the trapezoidal integration
  mat integral = trapz(u, integrand_values);

  double res=integral(0,0);

  return res;
}




//  Integrand function (Trapezoidal Integration)

vec IntegrandFunc_beta_IR(double sigma,const vec &u,double SSD,double DEL,double DEL_s,
                          double b_stop,double nu_stop,
                          double b_CR,double nu1_CR,const double nu2_CR,
                          double b_IR,double nu1_IR,const double nu2_IR,const bool lt=1){


  vec dens = log_dens_stop_I(sigma,SSD,u,DEL_s,b_stop,nu_stop);
  vec surv_CR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);
  vec surv_IR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec A=exp(dens+surv_CR+surv_IR);


  vec dens_s_IR= log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec W_IR=exp((2*dens_s_IR)-surv_IR);



  vec Deriv_beta_IR=deriv_log_dens_beta_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec weight_grad_beta_IR=W_IR%Deriv_beta_IR;

  vec result=A%(-weight_grad_beta_IR);

  return result;
}





// Integration (Trapezoidal Integration)


double integrate_integral_beta_IR(double sigma,double SSD,double DEL,double DEL_s,
                                  const vec &stop_param,
                                  double b_CR,double nu1_CR,const double nu2_CR,
                                  double b_IR,double nu1_IR,const double nu2_IR,
                                  double lower_bound,double upper_bound,const bool lt=1){



  double lower_bound_new=lower_bound+0.00004;

  double  n_points=100;

  vec u = linspace(lower_bound_new,upper_bound, n_points);



  // Compute the integrand values
  vec integrand_values =IntegrandFunc_beta_IR(sigma,u,SSD,DEL,DEL_s,stop_param(0),stop_param(1),
                                              b_CR,nu1_CR,nu2_CR,
                                              b_IR,nu1_IR,nu2_IR,lt);


  // Perform the trapezoidal integration
  mat integral = trapz(u, integrand_values);



  double res=integral(0,0);

  return res;
}




// Integral (Beta_l)

field<vec> integrate_beta_l(const field<vec> &integral_likelihood,const double sigma,
                            const vec &SSD,const double DEL,const double DEL_s,
                            const vec &stop_param,const vec &penal_param,const vec &prob_param,
                            const vec &b_l,const vec &b_r,
                            const vec &nu_l,const vec &nu_r,
                            const vec &nu_l_s,const vec &nu_r_s,
                            const vec &nu_r_p, const vec &nu_l_p,
                            const vec &nu_l_squared,
                            const uvec &Ind_L,const uvec &Ind_R,const uvec &Ind_I_L,const uvec &Ind_I_R,
                            const vec &lower_bound,const double upper_bound,const bool lt = true) {


  // Rcpp::Rcout<<"AAAI"<< endl;

  vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );
  vec SSD_I_R=g(SSD, Ind_R, Ind_I_R );

  vec b_l_I_L=g(b_l, Ind_L, Ind_I_L );
  vec b_l_I_R=g(b_l, Ind_R, Ind_I_R );

  vec b_r_I_L=g(b_r, Ind_L, Ind_I_L );
  vec b_r_I_R=g(b_r, Ind_R, Ind_I_R );

  vec nu_l_I_L=g(nu_l, Ind_L, Ind_I_L );
  vec nu_r_I_R=g(nu_r, Ind_R, Ind_I_R );

  vec nu_l_squared_I_L=g(nu_l_squared, Ind_L, Ind_I_L );

  vec nu_l_s_I_L=g(nu_l_s, Ind_L, Ind_I_L );
  vec nu_l_s_I_R=g(nu_l_s, Ind_R, Ind_I_R );

  vec nu_r_s_I_L=g(nu_r_s, Ind_L, Ind_I_L );
  vec nu_r_s_I_R=g(nu_r_s, Ind_R, Ind_I_R );

  vec nu_r_p_I_L=h(nu_r_p, Ind_I_L );
  vec nu_l_p_I_R=h(nu_l_p, Ind_I_R );

  vec lower_bound_I_L=g(lower_bound, Ind_L, Ind_I_L );
  vec lower_bound_I_R=g(lower_bound, Ind_R, Ind_I_R );


  // Precompute multiplier outside the loop


  // Initialize result vector
  vec Stim_L(b_l_I_L.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_l_I_L.size(); ++i) {
    // Perform integration for current parameter values
    Stim_L[i] = (integrate_integral_beta_CR(sigma,SSD_I_L(i),DEL,DEL_s,stop_param,
                                            b_l_I_L(i),nu_l_I_L(i),nu_l_s_I_L (i),
                                            b_r_I_L(i),nu_r_p_I_L(i),nu_r_s_I_L(i),
                                            nu_l_I_L(i), nu_l_squared_I_L(i),
                                            penal_param,lower_bound_I_L(i),upper_bound,lt));


    // // Save the result
    // Stim_L[i] = result;
    //


  }


  // Initialize result vector
  vec Stim_R(b_r_I_R.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_r_I_R.size(); ++i) {
    // Perform integration for current parameter values
    Stim_R[i] =(integrate_integral_beta_IR(sigma,SSD_I_R(i),DEL,DEL_s,stop_param,
                                           b_r_I_R(i),nu_r_I_R(i),nu_r_s_I_R(i),
                                           b_l_I_R(i),nu_l_p_I_R(i),nu_l_s_I_R(i),
                                           lower_bound_I_R(i),upper_bound,lt));


    // // Save the result
    // Stim_R[i] = result;

  }

  vec A_L=exp(prob_param(1)-integral_likelihood(0));
  vec A_R=exp(prob_param(1)-integral_likelihood(1));


  vec Stim_L_final=Stim_L%A_L;
  vec Stim_R_final=Stim_R%A_R;


  field<vec> result(2);

  result(0)=Stim_L_final;
  result(1)=Stim_R_final;



  return result;
}



// Integral (Beta_R)




field<vec> integrate_beta_r(const field<vec> &integral_likelihood,const double sigma,
                            const vec &SSD,const double DEL,const double DEL_s,
                            const vec &stop_param,const vec &penal_param,const vec &prob_param,
                            const vec &b_l,const vec &b_r,
                            const vec &nu_l,const vec &nu_r,
                            const vec &nu_l_s,const vec &nu_r_s,
                            const vec &nu_r_p,const vec &nu_l_p,
                            const vec &nu_r_squared,
                            const uvec &Ind_L,const uvec &Ind_R,const uvec &Ind_I_L,const uvec &Ind_I_R,
                            const vec &lower_bound,const double upper_bound,const bool lt = true) {


  // Rcpp::Rcout<<"AAAJ"<< endl;

  vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );
  vec SSD_I_R=g(SSD, Ind_R, Ind_I_R );

  vec b_l_I_L=g(b_l, Ind_L, Ind_I_L );
  vec b_l_I_R=g(b_l, Ind_R, Ind_I_R );

  vec b_r_I_L=g(b_r, Ind_L, Ind_I_L );
  vec b_r_I_R=g(b_r, Ind_R, Ind_I_R );

  vec nu_l_I_L=g(nu_l, Ind_L, Ind_I_L );
  vec nu_r_I_R=g(nu_r, Ind_R, Ind_I_R );

  vec nu_r_squared_I_R=g(nu_r_squared, Ind_R, Ind_I_R );

  vec nu_l_s_I_L=g(nu_l_s, Ind_L, Ind_I_L );
  vec nu_l_s_I_R=g(nu_l_s, Ind_R, Ind_I_R );

  vec nu_r_s_I_L=g(nu_r_s, Ind_L, Ind_I_L );
  vec nu_r_s_I_R=g(nu_r_s, Ind_R, Ind_I_R );

  vec nu_r_p_I_L=h(nu_r_p, Ind_I_L );
  vec nu_l_p_I_R=h(nu_l_p, Ind_I_R );

  vec lower_bound_I_L=g(lower_bound, Ind_L, Ind_I_L );
  vec lower_bound_I_R=g(lower_bound, Ind_R, Ind_I_R );




  // Initialize result vector
  vec Stim_L(b_l_I_L.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_l_I_L.size(); ++i) {
    // Perform integration for current parameter values
    Stim_L[i] = (integrate_integral_beta_IR(sigma,SSD_I_L(i),DEL,DEL_s,stop_param,
                                            b_l_I_L(i),nu_l_I_L(i),nu_l_s_I_L(i),
                                            b_r_I_L(i),nu_r_p_I_L(i),nu_r_s_I_L(i),
                                            lower_bound_I_L(i),upper_bound,lt));


    // // Save the result
    // Stim_L[i] = result;

  }


  // Initialize result vector
  vec Stim_R(b_r_I_R.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_r_I_R.size(); ++i) {
    // Perform integration for current parameter values
    Stim_R[i]  =(integrate_integral_beta_CR(sigma,SSD_I_R(i),DEL,DEL_s,stop_param,
                                            b_r_I_R(i),nu_r_I_R(i),nu_r_s_I_R(i),
                                            b_l_I_R(i),nu_l_p_I_R(i),nu_l_s_I_R(i),
                                            nu_r_I_R(i),nu_r_squared_I_R(i),
                                            penal_param,lower_bound_I_R(i),upper_bound,lt));

    // // Save the result
    // Stim_R[i] = result;

  }

  vec A_L=exp(prob_param(1)-integral_likelihood(0));
  vec A_R=exp(prob_param(1)-integral_likelihood(1));

  vec Stim_L_final=Stim_L%A_L;
  vec Stim_R_final=Stim_R%A_R;


  field<vec> result(2);

  result(0)=Stim_L_final;
  result(1)=Stim_R_final;



  return result;
}


//////////////////////////////////////////////////////////////////////////////////////// delta_prime /////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// derivative of exponent w.r.t. delta prime

vec deriv_exp_delta_prime1_I(double sigma,const vec &u,double b,const vec &m,double nu1){


  vec A=(sqrt(u)*exp(nu1))+(((exp(b)-m))/(2*sqrt(u)));

  return (A/(sqrt(sigma)*u));

}



vec deriv_exp_delta_prime2_I(double sigma,const vec &u,double b,const vec &m,double nu1){


  vec A=(sqrt(u)*exp(nu1))+(((-exp(b)-m))/(2*sqrt(u)));

  return (A/(sqrt(sigma)*u));

}


inline vec deriv_nu_diff_delta_prime_I(const vec &u,const double SSD,double nu1,double nu2,const double DEL, const double DEL_s){

  return ((SSD+DEL_s-DEL-(2*u))*(exp(nu2)-exp(nu1)))/(2*pow(u,3/2));




}




// Derivative of log dense w.r.t. delta (Go Proces-Failed Stop Trial)

vec deriv_log_dens_delta_prime_I(const double sigma,const vec &u,
                                 const double SSD,const double DEL,const double DEL_s,
                                 const double b,const double nu1,const double nu2,
                                 const double deriv_delta,
                                 const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;

  vec u_d=u-DEL;
  vec u_s=u-SSD-DEL_s;

  // Defining necessary vectors
  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);

  vec q1=exponent_1_I(sigma,u_d,m,b);

  vec q2 = exponent_2_I(sigma,u_d,m,b);

  vec rat=phi_rat_I(q2,lt);

  vec SSD_diff=SSD_Dif_I(sigma,u_d,SSD,nu1,nu2,DEL,DEL_s);

  vec deriv_delta_prime1=deriv_exp_delta_prime1_I(sigma,u_d,b,m,nu1);

  vec deriv_delta_prime2=deriv_exp_delta_prime2_I(sigma,u_d,b,m,nu1);

  vec deriv_nu_diff_delta_prime=deriv_nu_diff_delta_prime_I(u_d,SSD,nu1,nu2,DEL,DEL_s);

  // calculating small vectors

  vec A=(3/2)*(1/u_d);

  vec c1=q1%deriv_delta_prime1;

  vec deriv_delta_prime_B=(deriv_multiplier_I(SSD_diff,rat,q2)%deriv_delta_prime2)+(2*rat%deriv_nu_diff_delta_prime/sqrt(sigma));

  vec B=1+(rat%SSD_diff);

  vec C=(A-c1+(deriv_delta_prime_B/B))*deriv_delta;


  return C;


}





// Integrand function (Trapezoidal Integration)


vec IntegrandFunc_delta_prime(const vec &u,double sigma,double SSD,double DEL,double DEL_s,
                              double b_stop,double nu_stop,
                              double b_CR,double nu1_CR,const double nu2_CR,
                              double b_IR,double nu1_IR,const double nu2_IR,const double deriv_delta,
                              const bool lt=1){


  // Rcpp::Rcout<<"delta_prime"<<endl;

  vec dens = log_dens_stop_I(sigma,SSD,u,DEL_s,b_stop,nu_stop);
  vec surv_CR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);
  vec surv_IR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec A=exp(dens+surv_CR+surv_IR);

  vec dens_s_CR=log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);

  vec W_CR=exp((2*dens_s_CR)-surv_CR);

  vec Deriv_delta_prime_CR=deriv_log_dens_delta_prime_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,deriv_delta,lt);

  vec weight_grad_delta_prime_CR=W_CR%Deriv_delta_prime_CR;

  vec dens_s_IR=log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec W_IR=exp((2*dens_s_IR)-surv_IR);

  vec Deriv_delta_prime_IR=deriv_log_dens_delta_prime_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,deriv_delta,lt);

  vec weight_grad_delta_prime_IR=W_IR%Deriv_delta_prime_IR;


  vec result=A%(-weight_grad_delta_prime_CR-weight_grad_delta_prime_IR);


  return result;
}



// Integration (Trapezoidal Integration)


double integrate_integral_delta_prime(double sigma,double SSD,double DEL,double DEL_s,const vec &stop_param,
                                      double b_CR,double nu1_CR,const double nu2_CR,
                                      double b_IR,double nu1_IR,const double nu2_IR,
                                      double deriv_delta,
                                      double lower_bound,double upper_bound,const bool lt=1) {


  double lower_bound_new=lower_bound+0.00004;

  // double lower_bound_new=lower_bound;


  double  n_points=100;

  vec u = linspace(lower_bound_new,upper_bound, n_points);



  // Compute the integrand values
  vec integrand_values =IntegrandFunc_delta_prime(u,sigma,SSD,DEL,DEL_s,stop_param(0),stop_param(1),
                                                  b_CR,nu1_CR,nu2_CR,b_IR,nu1_IR,nu2_IR,lt);





  // Perform the trapezoidal integration
  mat integral = trapz(u, integrand_values);




  double res=integral(0,0);

  return res;
}



// Integral (delta_prime)



field<vec> integrate_delta_prime(const field<vec> &integral_likelihood,const double sigma,
                                 const vec &stop_param,const vec &penal_param,const vec &prob_param,
                                 const vec &SSD,const double DEL,const double DEL_s,
                                 const vec &b_l,const vec &b_r,
                                 const vec &nu_l,const vec &nu_r,
                                 const vec &nu_l_s,const vec &nu_r_s,
                                 const vec &nu_r_p,const vec &nu_l_p,
                                 const double deriv_delta,
                                 const uvec &Ind_L,const uvec &Ind_R,const uvec &Ind_I_L,const uvec &Ind_I_R,
                                 const vec &lower_bound,const double upper_bound,const bool lt =1) {


  // Rcpp::Rcout<<"AAAM"<< endl;

  vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );
  vec SSD_I_R=g(SSD, Ind_R, Ind_I_R );

  vec b_l_I_L=g(b_l, Ind_L, Ind_I_L );
  vec b_l_I_R=g(b_l, Ind_R, Ind_I_R );

  vec b_r_I_L=g(b_r, Ind_L, Ind_I_L );
  vec b_r_I_R=g(b_r, Ind_R, Ind_I_R );

  vec nu_l_I_L=g(nu_l, Ind_L, Ind_I_L );
  vec nu_r_I_R=g(nu_r, Ind_R, Ind_I_R );


  vec nu_l_s_I_L=g(nu_l_s, Ind_L, Ind_I_L );
  vec nu_l_s_I_R=g(nu_l_s, Ind_R, Ind_I_R );

  vec nu_r_s_I_L=g(nu_r_s, Ind_L, Ind_I_L );
  vec nu_r_s_I_R=g(nu_r_s, Ind_R, Ind_I_R );

  vec nu_r_p_I_L=h(nu_r_p, Ind_I_L );
  vec nu_l_p_I_R=h(nu_l_p, Ind_I_R );

  vec lower_bound_I_L=g(lower_bound, Ind_L, Ind_I_L );
  vec lower_bound_I_R=g(lower_bound, Ind_R, Ind_I_R );




  // Initialize result vector
  vec Stim_L(b_l_I_L.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_l_I_L.size(); ++i) {
    // Perform integration for current parameter values
    double result = (integrate_integral_delta_prime(sigma,SSD_I_L(i),DEL,DEL_s,stop_param,
                                                    b_l_I_L(i),nu_l_I_L(i),nu_l_s_I_L(i),
                                                    b_r_I_L(i),nu_r_p_I_L(i),nu_r_s_I_L(i),
                                                    deriv_delta,
                                                    lower_bound_I_L(i),upper_bound,lt));



    // Save the result
    Stim_L[i] = result;
  }




  // Initialize result vector
  vec Stim_R(b_r_I_R.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_r_I_R.size(); ++i) {
    // Perform integration for current parameter values
    double result =(integrate_integral_delta_prime(sigma,SSD_I_R(i),DEL,DEL_s,stop_param,
                                                   b_r_I_R(i),nu_r_I_R(i),nu_r_s_I_R(i),
                                                   b_l_I_R(i),nu_l_p_I_R(i),nu_l_s_I_R(i),
                                                   deriv_delta,
                                                   lower_bound_I_R(i),upper_bound,lt));






    // Save the result
    Stim_R[i] = result;
  }

  vec A_L=exp(prob_param(1)-integral_likelihood(0));
  vec A_R=exp(prob_param(1)-integral_likelihood(1));

  vec Stim_L_final=Stim_L%A_L;
  vec Stim_R_final=Stim_R%A_R;



  field<vec> result(2);

  result(0)=Stim_L_final;
  result(1)=Stim_R_final;



  return result;
}



//////////////////////////////////////////////////////////////////////////////////////// delta_prime_s /////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Derivative of log dense stop w.r.t. delta_prime_s

vec deriv_log_dens_stop_delta_prime_s_I(const vec &u,
                                        const double sigma,const double SSD, const double DEL_s,
                                        const double &b_stop, const double &nu_stop,
                                        const double deriv_delta_s) {

  // Compute DEL_s
  // double DEL_s = V * EXPITE(delta_param(0));

  // Compute shifted u
  vec u_s = u - DEL_s - SSD;


  // Initialize output vector
  vec C(u_s.n_elem);


  for (uword i = 0; i < u_s.n_elem; ++i) {
    if (u_s(i) <= 0) {
      C(i) = 0;

    } else {
      double u_si = u_s(i);
      double u_sq_i = pow(u_si,2);

      C(i) = ((1.5 / u_si) -
        (exp(2*b_stop) / (2 * sigma * u_sq_i)) +
        (exp(2* nu_stop) / (2* sigma))) * deriv_delta_s;
    }
  }

  return C;
}





// Derivative of log dense w.r.t. delta (Go Proces-Failed Stop Trial)

vec deriv_log_dens_delta_prime_s_I(const double sigma,const vec &u,
                                 const double SSD,const double DEL,const double DEL_s,
                                 const double b,const double nu1,const double nu2,
                                 const double deriv_delta_s,
                                 const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;

  vec u_d=u-DEL;
  vec u_s=u-SSD-DEL_s;

  // Defining necessary vectors
  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);
  vec q1=exponent_1_I(sigma,u_d,m,b);

  vec q2 = exponent_2_I(sigma,u_d,m,b);

  vec rat=phi_rat_I(q2,lt);

  vec SSD_diff=SSD_Dif_I(sigma,u_d,SSD,nu1,nu2,DEL,DEL_s);


  // calculating small vectors

  vec deriv_delta_prime_B=deriv_multiplier_I(SSD_diff,rat,q2)+(2*rat);

  vec B=1+(rat%SSD_diff);

  vec deriv=(exp(nu2)-exp(nu1))/(sqrt(sigma*u_d));

  vec C=((-q1+(deriv_delta_prime_B/B))%deriv)*deriv_delta_s;


  return C;


}


// Integrand function (Trapezoidal Integration)

vec IntegrandFunc_delta_prime_s(const vec &u,double sigma,double SSD,double DEL,double DEL_s,
                                double b_stop,double nu_stop,
                                double b_CR,double nu1_CR,const double nu2_CR,
                                double b_IR,double nu1_IR,const double nu2_IR,
                                const double deriv_delta_s,const double deriv_delta_s_stop,
                                const bool lt=1){

  // Rcpp::Rcout<<"delta_prime_s"<<endl;


  vec dens = log_dens_stop_I(sigma,SSD,u,DEL_s,b_stop,nu_stop);
  vec surv_CR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);
  vec surv_IR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec A=exp(dens+surv_CR+surv_IR);

  vec deriv_delta_prime_s_stop=deriv_log_dens_stop_delta_prime_s_I(u,sigma,SSD,DEL_s,b_stop,nu_stop,deriv_delta_s_stop);


  vec dens_s_CR=log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);

  vec W_CR=exp((2*dens_s_CR)-surv_CR);

  vec Deriv_delta_prime_CR=deriv_log_dens_delta_prime_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,deriv_delta_s,lt);


  vec weight_grad_delta_prime_CR=W_CR%Deriv_delta_prime_CR;

  vec dens_s_IR=log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec W_IR=exp((2*dens_s_IR)-surv_IR);

  vec Deriv_delta_prime_IR=deriv_log_dens_delta_prime_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,deriv_delta_s,lt);

  vec weight_grad_delta_prime_IR=W_IR%Deriv_delta_prime_IR;


  vec result=A%(deriv_delta_prime_s_stop-weight_grad_delta_prime_CR-weight_grad_delta_prime_IR);





  return result;
}





// Integration (Trapezoidal Integration)


double integrate_integral_delta_prime_s(double sigma,double SSD,double DEL,double DEL_s,const vec &stop_param,
                                        double b_CR,double nu1_CR,const double nu2_CR,
                                        double b_IR,double nu1_IR,const double nu2_IR,
                                        const double deriv_delta_s,const double deriv_delta_s_stop,
                                        double lower_bound,double upper_bound,const bool lt=1) {

  double lower_bound_new=lower_bound;
  // double lower_bound_new=lower_bound+0.00004;

  double  n_points=100;

  vec u = linspace(lower_bound_new,upper_bound, n_points);



  // Compute the integrand values
  vec integrand_values =IntegrandFunc_delta_prime_s(u,sigma,SSD,DEL,DEL_s,stop_param(0),stop_param(1),
                                                    b_CR,nu1_CR,nu2_CR,b_IR,nu1_IR,nu2_IR, deriv_delta_s,lt);




  // Perform the trapezoidal integration
  mat integral = trapz(u, integrand_values);




  double res=integral(0,0);

  return res;
}





// Integral (delta_prime_s)

field<vec> integrate_delta_prime_s(const field<vec> &integral_likelihood,const double sigma,
                                   const vec &stop_param,const vec &penal_param,const vec &prob_param,
                                   const vec &SSD,const double DEL,const double DEL_s,
                                   const vec &b_l,const vec &b_r,
                                   const vec &nu_l,const vec &nu_r,
                                   const vec &nu_l_s,const vec &nu_r_s,
                                   const vec &nu_r_p,const vec &nu_l_p,
                                   const double deriv_delta_s,const double deriv_delta_s_stop,
                                   const uvec &Ind_L,const uvec &Ind_R,const uvec &Ind_I_L,const uvec &Ind_I_R,
                                   const vec &lower_bound,const double upper_bound,const bool lt =1) {


  // Rcpp::Rcout<<"AAAS"<< endl;

  vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );
  vec SSD_I_R=g(SSD, Ind_R, Ind_I_R );

  vec b_l_I_L=g(b_l, Ind_L, Ind_I_L );
  vec b_l_I_R=g(b_l, Ind_R, Ind_I_R );

  vec b_r_I_L=g(b_r, Ind_L, Ind_I_L );
  vec b_r_I_R=g(b_r, Ind_R, Ind_I_R );

  vec nu_l_I_L=g(nu_l, Ind_L, Ind_I_L );
  vec nu_r_I_R=g(nu_r, Ind_R, Ind_I_R );


  vec nu_l_s_I_L=g(nu_l_s, Ind_L, Ind_I_L );
  vec nu_l_s_I_R=g(nu_l_s, Ind_R, Ind_I_R );

  vec nu_r_s_I_L=g(nu_r_s, Ind_L, Ind_I_L );
  vec nu_r_s_I_R=g(nu_r_s, Ind_R, Ind_I_R );

  vec nu_r_p_I_L=h(nu_r_p, Ind_I_L );
  vec nu_l_p_I_R=h(nu_l_p, Ind_I_R );

  vec lower_bound_I_L=g(lower_bound, Ind_L, Ind_I_L );
  vec lower_bound_I_R=g(lower_bound, Ind_R, Ind_I_R );





  // Initialize result vector
  vec Stim_L(b_l_I_L.size());


  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_l_I_L.size(); ++i) {
    // Perform integration for current parameter values
    double result = (integrate_integral_delta_prime_s(sigma,SSD_I_L(i),DEL,DEL_s,stop_param,
                                                      b_l_I_L(i),nu_l_I_L(i),nu_l_s_I_L(i),
                                                      b_r_I_L(i),nu_r_p_I_L(i),nu_r_s_I_L(i),
                                                      deriv_delta_s,deriv_delta_s_stop,
                                                      lower_bound_I_L(i),upper_bound,lt));



    // Save the result
    Stim_L[i] = result;



  }


  // Initialize result vector
  vec Stim_R(b_r_I_R.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_r_I_R.size(); ++i) {
    // Perform integration for current parameter values
    double result =(integrate_integral_delta_prime_s(sigma,SSD_I_R(i),DEL,DEL_s,stop_param,
                                                     b_r_I_R(i),nu_r_I_R(i),nu_r_s_I_R(i),
                                                     b_l_I_R(i),nu_l_p_I_R(i),nu_l_s_I_R(i),
                                                     deriv_delta_s,deriv_delta_s_stop,
                                                     lower_bound_I_R(i),upper_bound,lt));






    // Save the result
    Stim_R[i] = result;



  }





  vec A_L=exp(prob_param(1)-integral_likelihood(0));
  vec A_R=exp(prob_param(1)-integral_likelihood(1));

  vec Stim_L_final=Stim_L%A_L;
  vec Stim_R_final=Stim_R%A_R;



  field<vec> result(2);

  result(0)=Stim_L_final;
  result(1)=Stim_R_final;



  return result;
}








///////////////////////////////////////////////////////////////////////////////////// Penalty  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Derivative of nu_stop penalty w.r.t lambda (Go Process-Failed Stop Trial)


double g_nu_stop_stim_penal_lambda(const double &nu_stop, double lambda_prime,double alpha_prime){

  double X=(exp(2*(lambda_prime+nu_stop)));
  double Y=exp(2*alpha_prime);
  double hpot=sqrt(X+Y);

  double numerator =(exp(nu_stop))-((exp((2*nu_stop)+lambda_prime))/hpot);

  double denominator=pow((-(exp(lambda_prime+nu_stop))+hpot),2);

  return ((numerator/denominator)*exp(lambda_prime));


}



// Derivative of nu_stop penalty w.r.t alpha (Go Process-Failed Stop Trial)

double g_nu_stop_stim_penal_alpha(const double &nu_stop, double lambda_prime,double alpha_prime){

  double X=(exp(2*(lambda_prime+nu_stop)));
  double Y=exp(2*alpha_prime);
  double hpot=sqrt(X+Y);

  double numerator =-(exp(2*alpha_prime))/hpot;

  double denominator=pow((-(exp(lambda_prime+nu_stop))+hpot),2);

  return numerator/denominator;


}

// penal derivative (without penalty) (Go Process-Failed Stop Trial)


vec g_penal_com_s_I(const double sigma,const vec &u,
                    const double SSD,const double DEL,const double DEL_s, const double b,const double nu1,const double nu2,
                    double lambda_prime,double alpha_prime,double nu_stop,const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;

  vec u_d=u-DEL;
  vec u_s=u-SSD-DEL_s;
  // Defining necessary vectors

  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);
  vec q1=exponent_1_I(sigma,u_d,m,b);

  vec q2 = exponent_2_I(sigma,u_d,m,b);


  vec rat=phi_rat_I(q2,lt);

  vec SSD_diff=SSD_Dif_I(sigma,u_d,SSD,nu1,nu2,DEL,DEL_s);


  vec deriv_m=u_s/(sqrt(sigma*u_d));


  // calculating small vectors

  vec c1=-(q1%deriv_m);

  vec deriv_lambda_B=(deriv_multiplier_I(SSD_diff,rat,q2)%deriv_m)-((2*rat*(SSD+DEL_s-DEL))/sqrt(sigma*u_d));

  vec B=1+(rat%SSD_diff);

  return ((c1+(deriv_lambda_B/B))*exp(nu2));


}


// Derivative of beta penalty w.r.t lambda (Go Process-Failed Stop Trial)


double g_beta_stim_penal_lambda_I(const double nu_stim,const double nu_squared,double lambda_prime,double alpha_prime){

  double X=(exp(2*lambda_prime))*nu_squared;
  double Y=exp(2*alpha_prime);
  double hpot=sqrt(X+Y);

  double numerator = nu_stim- ((exp(lambda_prime)*nu_squared)/hpot);

  double denominator=pow((-(exp(lambda_prime)*nu_stim)+hpot),2);

  return ((numerator/denominator)*exp(lambda_prime));


}



// Derivative of beta penalty w.r.t alpha (Go Process-Failed Stop Trial)


double g_beta_stim_penal_alpha_I(const double nu_stim,const double nu_squared,double lambda_prime,double alpha_prime){

  double X=(exp(2*lambda_prime))*nu_squared;
  double Y=exp(2*alpha_prime);
  double hpot=sqrt(X+Y);

  double numerator = -(exp(alpha_prime)/hpot);

  double denominator=pow((-(exp(lambda_prime)*nu_stim)+hpot),2);

  double result=(numerator/denominator)*exp(alpha_prime);

  return result;
}



// derivative of exponent w.r.t. lambda (with penalty)

inline vec deriv_exp_penal_p_I(const vec &u_d,const vec &u_s,const double sigma,const double SSD,const double nu1,const double nu2,
                               double beta_deriv_penal,double nu_stop_deriv_penal,const double DEL,const double DEL_s){


  return ((((SSD+DEL_s-DEL)*exp(nu1)*beta_deriv_penal)+(u_s*exp(nu2)*nu_stop_deriv_penal))/(sqrt(sigma*u_d)));



}



// derivative of nu_diff w.r.t. lambda (with penalty)

double deriv_nu_diff_penal_p_I(const double SSD,const double nu1,const double nu2,double beta_deriv_penal,double nu_stop_deriv_penal,
                               const double DEL, const double DEL_s){


  return ((SSD+DEL_s-DEL)*((exp(nu1)*beta_deriv_penal)-(exp(nu2)*nu_stop_deriv_penal)));


}





// lambda derivative (with penalty) (Go Proces-Failed Stop Trial)


vec g_penal_com_p_s_I(const double sigma,const vec &u,
                      const double SSD,const double DEL,const double DEL_s,
                      const double b,const double nu1,const double nu2,
                      double beta_deriv_penal,double nu_stop_deriv_penal,const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;

  vec u_d=u-DEL;

  vec u_s=u-SSD-DEL_s;

  // Defining necessary vectors

  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);
  vec q1=exponent_1_I(sigma,u_d,m,b);

  vec q2 = exponent_2_I(sigma,u_d,m,b);

  vec rat=phi_rat_I(q2,lt);

  vec SSD_diff=SSD_Dif_I(sigma,u_d,SSD,nu1,nu2,DEL,DEL_s);

  vec deriv_lambda=deriv_exp_penal_p_I(u_d,u_s,sigma,SSD,nu1,nu2,
                                       beta_deriv_penal,nu_stop_deriv_penal,DEL,DEL_s);



  double deriv_lambda_nu_diff=deriv_nu_diff_penal_p_I(SSD,nu1,nu2,beta_deriv_penal,nu_stop_deriv_penal,DEL,DEL_s);


  // calculating small vectors

  vec c1=q1%deriv_lambda;

  vec deriv_lambda_B=(deriv_multiplier_I(SSD_diff,rat,q2)%(deriv_lambda))+((2*rat*deriv_lambda_nu_diff)/sqrt(sigma*u_d));

  vec B=1+(rat%SSD_diff);

  return (-c1+(deriv_lambda_B/B));


}




//  Integrand function (Trapezoidal Integration)

mat IntegrandFunc_penal(double sigma,const vec &u,double SSD,double DEL,double DEL_s,
                        double b_CR,double nu1_CR,const double nu2_CR,
                        double b_IR,double nu1_IR,const double nu2_IR,
                        double nu_stim,double nu_squared,
                        double lambda_prime,double alpha_prime,
                        double b_stop,double nu_stop,const bool lt=1){



  vec dens = log_dens_stop_I(sigma,SSD,u,DEL_s,b_stop,nu_stop);
  vec surv_CR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);
  vec surv_IR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec A=exp(dens+surv_CR+surv_IR);

  vec dens_s_CR=log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);

  vec W_CR=exp((2*dens_s_CR)-surv_CR);

  vec dens_s_IR=log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec W_IR=exp((2*dens_s_IR)-surv_IR);

  vec penal_C= g_penal_com_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,
                               lambda_prime,alpha_prime,nu_stop,lt);




  double beta_penal_lambda=g_beta_stim_penal_lambda_I(nu_stim,nu_squared,lambda_prime,alpha_prime);
  double nu_stop_penal_lambda=g_nu_stop_stim_penal_lambda(nu_stop,lambda_prime,alpha_prime);

  double beta_penal_alpha=g_beta_stim_penal_alpha_I(nu_stim, nu_squared,lambda_prime,alpha_prime);
  double nu_stop_penal_alpha=g_nu_stop_stim_penal_alpha(nu_stop, lambda_prime,alpha_prime);


  vec lambda_IC=g_penal_com_p_s_I(sigma,u,
                                  SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,
                                  beta_penal_lambda,nu_stop_penal_lambda, lt);

  vec alpha_IC=g_penal_com_p_s_I(sigma,u,
                                 SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,
                                 beta_penal_alpha,nu_stop_penal_alpha, lt);



  vec lambda=((penal_C*nu_stop_penal_lambda)%W_CR)+(lambda_IC%W_IR);

  vec alpha=((penal_C*nu_stop_penal_alpha)%W_CR)+(alpha_IC%W_IR);


  mat penal=join_rows(lambda,alpha);


  mat result=((penal).each_col())%(-A);

  return result;
}






// Integration (Trapezoidal Integration)


mat integrate_integral_penal(double sigma,double SSD,double DEL,double DEL_s,
                             double b_CR,double nu1_CR,const double nu2_CR,
                             double b_IR,double nu1_IR,const double nu2_IR,
                             const double nu_stim, const double nu_squared,
                             const vec &penal_param,const vec &stop_param,
                             double lower_bound,double upper_bound,const bool lt=1){



  double lower_bound_new=lower_bound+0.00004;

  double  n_points=100;

  vec u = linspace(lower_bound_new,upper_bound, n_points);



  // Compute the integrand values
  mat integrand_values =  IntegrandFunc_penal(sigma,u,SSD,DEL,DEL_s,
                                              b_CR,nu1_CR,nu2_CR,
                                              b_IR,nu1_IR,nu2_IR,
                                              nu_stim,nu_squared,
                                              penal_param(0),penal_param(1),
                                              stop_param(0),stop_param(1),lt);


  // Perform the trapezoidal integration
  mat integral = trapz(u, integrand_values);




  return integral;
}






// Integral (Left Stimulus)

mat integrate_penalty_Left_Stim(const field<vec> &integral_likelihood,const double sigma,
                                const vec &SSD,const double DEL,const double DEL_s,
                                const vec &penal_param,const vec &stop_param,const vec &prob_param,
                                const vec &b_l,const vec &b_r,
                                const vec &nu_l,const vec &nu_r,
                                const vec &nu_l_s,const vec &nu_r_s,
                                const vec &nu_r_p,
                                const vec &nu_l_squared,

                                const uvec &Ind_L,const uvec &Ind_I_L,
                                const vec &lower_bound,const double upper_bound,const bool lt = true) {



  // Rcpp::Rcout<<"AAAI"<< endl;

  vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );

  vec b_l_I_L=g(b_l, Ind_L, Ind_I_L );

  vec b_r_I_L=g(b_r, Ind_L, Ind_I_L );

  vec nu_l_I_L=g(nu_l, Ind_L, Ind_I_L );

  vec nu_l_squared_I_L=g(nu_l_squared, Ind_L, Ind_I_L );


  vec nu_l_s_I_L=g(nu_l_s, Ind_L, Ind_I_L );

  vec nu_r_s_I_L=g(nu_r_s, Ind_L, Ind_I_L );

  vec nu_r_p_I_L=h(nu_r_p, Ind_I_L );

  vec lower_bound_I_L=g(lower_bound, Ind_L, Ind_I_L );



  // Initialize result matrix
  mat Penal(SSD_I_L.n_rows, 2);


  // Loop over each row in b_I_L
  for (unsigned i = 0; i < SSD_I_L.n_rows; ++i) {


    Penal.row(i) = integrate_integral_penal(sigma,SSD_I_L(i),DEL,DEL_s,
                                            b_l_I_L(i),nu_l_I_L(i),nu_l_s_I_L(i),
                                            b_r_I_L(i),nu_r_p_I_L(i),nu_r_s_I_L(i),
                                            nu_l_I_L(i), nu_l_squared_I_L(i),
                                            penal_param,stop_param,
                                            lower_bound_I_L(i), upper_bound, lt);
  }


  // Compute final result

  vec A_L = exp(prob_param(1)-integral_likelihood(0));

  Penal.each_col() %= A_L;


  return Penal;
}



// Integral (Right Stimulus)

mat integrate_penal_Right_Stim(const field<vec> &integral_likelihood,const double sigma,
                               const vec &SSD, const double DEL,const double DEL_s,const vec &penal_param,const vec &stop_param,const vec &prob_param,
                               const vec &b_l,const vec &b_r,
                               const vec &nu_l,const vec &nu_r,
                               const vec &nu_l_s,const vec &nu_r_s,
                               const vec &nu_l_p,

                               const vec &nu_r_squared,
                               const uvec &Ind_R,const uvec &Ind_I_R,
                               const vec &lower_bound,const double upper_bound,const bool lt = true) {


  // Rcpp::Rcout<<"AAAJ"<< endl;

  vec SSD_I_R=g(SSD, Ind_R, Ind_I_R );

  vec b_l_I_R=g(b_l, Ind_R, Ind_I_R );

  vec b_r_I_R=g(b_r, Ind_R, Ind_I_R );

  vec nu_r_I_R=g(nu_r, Ind_R, Ind_I_R );


  vec nu_r_squared_I_R=g(nu_r_squared, Ind_R, Ind_I_R );


  vec nu_l_s_I_R=g(nu_l_s, Ind_R, Ind_I_R );

  vec nu_r_s_I_R=g(nu_r_s, Ind_R, Ind_I_R );

  vec nu_l_p_I_R=h(nu_l_p, Ind_I_R );

  vec lower_bound_I_R=g(lower_bound, Ind_R, Ind_I_R );





  // Initialize result matrix
  mat Penal(SSD_I_R.n_rows, 2);

  // Parallelized loop for Gama_L
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < SSD_I_R.n_rows; ++i) {
    // Perform integration for current parameter values
    Penal.row(i) =integrate_integral_penal(sigma,SSD_I_R(i),DEL, DEL_s,
                                           b_r_I_R(i),nu_r_I_R(i),nu_r_s_I_R(i),
                                           b_l_I_R(i),nu_l_p_I_R(i),nu_l_s_I_R(i),
                                           nu_r_I_R(i), nu_r_squared_I_R(i),
                                           penal_param,stop_param,
                                           lower_bound_I_R(i), upper_bound, lt);

  }

  // Compute final result

  vec A_R = exp(prob_param(1)-integral_likelihood(1));

  Penal.each_col() %= A_R;


  return Penal;
}









//////////////////////////////////////////////////////////////////////////////////////// b_stop /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Derivative of log dense w.r.t. b_stop (Integral Version)

vec deriv_log_dens_b_stop_I(const vec u,double DEL_s,double SSD,double sigma,double b_stop,double nu_stop){

  // Rcpp::Rcout<<"AAAK"<< endl;

   vec u_s=u-SSD-DEL_s;


  vec C= 1-(((exp(2*b_stop))-(exp(b_stop+nu_stop)*u_s))/(sigma*u_s));

  return C; //length(C)

}



// Integrand function(Trapezoidal Integration)

vec IntegrandFunc_b_stop(const vec &u,double sigma,double SSD,double DEL,double DEL_s,
                         double b_stop,double nu_stop,
                         double b_CR,double nu1_CR,const double nu2_CR,
                         double b_IR,double nu1_IR,const double nu2_IR,const bool lt){



  vec dens = log_dens_stop_I(sigma,SSD,u,DEL_s,b_stop,nu_stop);
  vec surv_CR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);
  vec surv_IR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);


  vec A=exp(dens+surv_CR+surv_IR);

  vec deriv_b_stop=deriv_log_dens_b_stop_I(u,DEL_s,SSD,sigma,b_stop,nu_stop);

  vec result=A%deriv_b_stop;


  return result;
}




// Integration (Trapezoidal Integration)


double integrate_integral_b_stop(double sigma,double SSD,double DEL,double DEL_s,
                                 const vec &stop_param,
                                 double b_CR,double nu1_CR,const double nu2_CR,
                                 double b_IR,double nu1_IR,const double nu2_IR,
                                 double lower_bound,double upper_bound,const bool lt=1) {


  double lower_bound_new=lower_bound+0.00004;

  double  n_points=100;

  vec u = linspace(lower_bound_new,upper_bound, n_points);


  // Compute the integrand values
  vec integrand_values = IntegrandFunc_b_stop(u,sigma,SSD,DEL,DEL_s,stop_param(0),stop_param(1),
                                              b_CR,nu1_CR,nu2_CR,b_IR,nu1_IR,nu2_IR,lt);





  // Perform the trapezoidal integration
  mat integral = trapz(u, integrand_values);


  double res=integral(0,0);

  return res;
}





// Integral (b_stop)


field<vec> integrate_b_stop(const field<vec> &integral_likelihood,const double sigma,
                            const vec &SSD,const double DEL,const double DEL_s,
                            const vec &stop_param,const vec &penal_param,const vec &prob_param,
                            const vec &b_l,const vec &b_r,
                            const vec &nu_l,const vec &nu_r,
                            const vec &nu_l_s,const vec &nu_r_s,
                            const vec &nu_r_p,const vec &nu_l_p,
                            const uvec &Ind_L,const uvec &Ind_R,const uvec &Ind_I_L,const uvec &Ind_I_R,
                            const vec &lower_bound,const double upper_bound,const bool lt =1) {


  // Rcpp::Rcout<<"AAAO"<< endl;

  vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );
  vec SSD_I_R=g(SSD, Ind_R, Ind_I_R );

  vec b_l_I_L=g(b_l, Ind_L, Ind_I_L );
  vec b_l_I_R=g(b_l, Ind_R, Ind_I_R );

  vec b_r_I_L=g(b_r, Ind_L, Ind_I_L );
  vec b_r_I_R=g(b_r, Ind_R, Ind_I_R );

  vec nu_l_I_L=g(nu_l, Ind_L, Ind_I_L );
  vec nu_r_I_R=g(nu_r, Ind_R, Ind_I_R );

  vec nu_l_s_I_L=g(nu_l_s, Ind_L, Ind_I_L );
  vec nu_l_s_I_R=g(nu_l_s, Ind_R, Ind_I_R );

  vec nu_r_s_I_L=g(nu_r_s, Ind_L, Ind_I_L );
  vec nu_r_s_I_R=g(nu_r_s, Ind_R, Ind_I_R );

  vec nu_r_p_I_L=h(nu_r_p, Ind_I_L );
  vec nu_l_p_I_R=h(nu_l_p, Ind_I_R );

  vec lower_bound_I_L=g(lower_bound, Ind_L, Ind_I_L );
  vec lower_bound_I_R=g(lower_bound, Ind_R, Ind_I_R );


  // Initialize result vector
  vec Stim_L(b_l_I_L.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_l_I_L.size(); ++i) {
    // Perform integration for current parameter values
    Stim_L[i] = (integrate_integral_b_stop(sigma,SSD_I_L[i],DEL,DEL_s,stop_param,
                                           b_l_I_L[i],nu_l_I_L(i),nu_l_s_I_L(i),
                                           b_r_I_L[i],nu_r_p_I_L(i),nu_r_s_I_L(i),
                                           lower_bound_I_L(i),upper_bound));



  }



  // Initialize result vector
  vec Stim_R(b_r_I_R.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_r_I_R.size(); ++i) {
    // Perform integration for current parameter values
    Stim_R[i] =(integrate_integral_b_stop(sigma,SSD_I_R[i],DEL,DEL_s,stop_param,
                                          b_r_I_R[i],nu_r_I_R(i),nu_r_s_I_R(i),
                                          b_l_I_R(i),nu_l_p_I_R(i),nu_l_s_I_R(i),
                                          lower_bound_I_R(i),upper_bound));

  }

  vec A_L=exp(prob_param(1)-integral_likelihood(0));
  vec A_R=exp(prob_param(1)-integral_likelihood(1));

  vec Stim_L_final=Stim_L%A_L;
  vec Stim_R_final=Stim_R%A_R;


  field<vec> result(2);

  result(0)=Stim_L_final;
  result(1)=Stim_R_final;



  return result;
}



////////////////////////////////////////////////////////////////////////////////////// nu_stop ///////////////////////////////////////////////////////////////////////////////////////////////////////////


// Derivative of log dense w.r.t. nu_stop (Stop Trial-Integral version)


vec deriv_log_dens_nu_stop_I(const vec u,double sigma,double SSD,double DEL_s, double b_stop,double nu_stop){

  // Rcpp::Rcout<<"AAAL"<< endl;

  vec u_s=u-SSD-DEL_s;


  vec C= ((exp(b_stop+nu_stop))-((exp(2*nu_stop))*u_s))/(sigma);

  return C;

}




// Derivative of nu_stop penalty w.r.t nu_stop (Go Process-Failed Stop Trial)


double g_nu_stop_stim_penal_I(const double &nu_stop, double lambda_prime,double alpha_prime){

  double X=(exp(2*(lambda_prime+nu_stop)));
  double Y=exp(2*alpha_prime);
  double hpot=sqrt(X+Y);

  double numerator = (exp(lambda_prime+nu_stop))-((exp(2*(lambda_prime+nu_stop)))/hpot);

  double denominator=pow((-(exp(lambda_prime+nu_stop))+hpot),2);

  return numerator/denominator;


}


// derivative of exponent w.r.t. beta (without penalty)

double deriv_exp_nu_stop_I(const double nu2,
                           const double &nu_stop, double lambda_prime,double alpha_prime){

  return ((exp(nu2))*g_nu_stop_stim_penal_I(nu_stop,lambda_prime,alpha_prime));

}




// beta derivative (without penalty) (Go Process-Failed Stop Trial)


vec deriv_log_dens_nu_stop_s_I(const double sigma,const vec &u,
                               const double SSD,const double DEL,
                               const double DEL_s, const double b,const double nu1,const double nu2,
                               const double &nu_stop, double lambda_prime,double alpha_prime,const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;

  vec u_d=u-DEL;
  vec u_s=u-SSD-DEL_s;

  // Defining necessary vectors


  vec m=(exp(nu1)*(SSD+DEL_s-DEL))+(exp(nu2)*u_s);
  vec q1=exponent_1_I(sigma,u_d,m,b);

  vec q2 = exponent_2_I(sigma,u_d,m,b);

  vec rat=phi_rat_I(q2,lt);

  vec SSD_diff=SSD_Dif_I(sigma,u_d,SSD,nu1,nu2,DEL,DEL_s);

  double deriv_nu_stop=deriv_exp_nu_stop_I(nu2,nu_stop,lambda_prime,alpha_prime);


  vec w=u_s/(sqrt(sigma*u_d));

  vec c1=q1%w;

  vec deriv_nu_stop_B=(deriv_multiplier_I(SSD_diff,rat,q2)%w)-((2*rat*(SSD+DEL_s-DEL))/(sqrt(sigma*u_d)));

  vec B=1+(rat%SSD_diff);

  return ((-c1+(deriv_nu_stop_B/B))*deriv_nu_stop);



}






// Integrand function (Trapezoidal Integration)

vec  IntegrandFunc_nu_stop(const vec &u,double sigma,double SSD,double DEL,double DEL_s,
                           double b_stop,double nu_stop,
                           double b_CR,double nu1_CR,const double nu2_CR,
                           double b_IR,double nu1_IR,const double nu2_IR,
                           const double nu_stim, const double nu_squared,
                           double lambda_prime,double alpha_prime,const bool lt=1){


  vec dens = log_dens_stop_I(sigma,SSD,u,DEL_s,b_stop,nu_stop);
  vec surv_CR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);
  vec surv_IR = log_survival_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec A=exp(dens+surv_CR+surv_IR);

  vec deriv_nu_stop_stop=deriv_log_dens_nu_stop_I(u,sigma,SSD,DEL_s,b_stop,nu_stop);


  vec log_dens_s_CR=2*log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,lt);

  vec W_CR=exp(log_dens_s_CR-surv_CR);

  vec Deriv_nu_stop_CR=deriv_log_dens_nu_stop_s_I(sigma,u,SSD,DEL,DEL_s,b_CR,nu1_CR,nu2_CR,nu_stop,
                                                  lambda_prime,alpha_prime,lt);


  vec weight_grad_nu_stop_CR=W_CR%Deriv_nu_stop_CR;


  vec log_dens_s_IR=2*log_dens_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,lt);

  vec W_IR=exp(log_dens_s_IR-surv_IR);

  vec Deriv_nu_stop_IR=deriv_log_dens_nu_stop_s_I(sigma,u,SSD,DEL,DEL_s,b_IR,nu1_IR,nu2_IR,nu_stop,
                                                  lambda_prime,alpha_prime,lt);

  vec weight_grad_nu_stop_IR=W_IR % Deriv_nu_stop_IR;

  vec result=A%(deriv_nu_stop_stop-weight_grad_nu_stop_CR-weight_grad_nu_stop_IR);


  return result;
}



// Integration (Trapezoidal Integration)


double integrate_integral_nu_stop(double sigma,double SSD,double DEL,
                                  double DEL_s,const vec &stop_param,
                                  double b_CR,double nu1_CR,const double nu2_CR,
                                  double b_IR,double nu1_IR,const double nu2_IR,
                                  const double nu_stim, const double nu_squared,
                                  double lambda_prime,double alpha_prime,double lower_bound,double upper_bound,double lt=1) {



  double lower_bound_new=lower_bound+0.00004;

  double  n_points=100;

  vec u = linspace(lower_bound_new,upper_bound, n_points);


  // Compute the integrand values
  vec integrand_values =IntegrandFunc_nu_stop(u,sigma,SSD,DEL,DEL_s,stop_param(0),stop_param(1),
                                              b_CR,nu1_CR,nu2_CR,
                                              b_IR,nu1_IR,nu2_IR,
                                              nu_stim,nu_squared,
                                              lambda_prime,alpha_prime,lt);



  // Perform the trapezoidal integration
  mat integral = trapz(u, integrand_values);



  double res=integral(0,0);

  return res;
}





// Integral (nu_stop)



field<vec> integrate_nu_stop(const field<vec> &integral_likelihood,const double sigma,
                             const vec &SSD,const double DEL,const double DEL_s,
                             const vec &stop_param,const vec &penal_param,const vec &prob_param,
                             const vec &b_l,const vec &b_r,
                             const vec &nu_l,const vec &nu_r,
                             const vec &nu_l_s,const vec &nu_r_s,
                             const vec &nu_r_p,const vec &nu_l_p,
                             const vec &nu_l_squared,const vec &nu_r_squared,
                             const uvec &Ind_L,const uvec &Ind_R,const uvec &Ind_I_L,const uvec &Ind_I_R,
                             const vec &lower_bound,const double upper_bound,const bool lt =1) {


  // Rcpp::Rcout<<"AAAR"<< endl;

  vec SSD_I_L=g(SSD, Ind_L, Ind_I_L );
  vec SSD_I_R=g(SSD, Ind_R, Ind_I_R );

  vec b_l_I_L=g(b_l, Ind_L, Ind_I_L );
  vec b_l_I_R=g(b_l, Ind_R, Ind_I_R );

  vec b_r_I_L=g(b_r, Ind_L, Ind_I_L );
  vec b_r_I_R=g(b_r, Ind_R, Ind_I_R );

  vec nu_l_I_L=g(nu_l, Ind_L, Ind_I_L );
  vec nu_r_I_R=g(nu_r, Ind_R, Ind_I_R );

  vec nu_l_squared_I_L=g(nu_l_squared, Ind_L, Ind_I_L );
  vec nu_r_squared_I_R=g(nu_r_squared, Ind_R, Ind_I_R);

  vec nu_l_s_I_L=g(nu_l_s, Ind_L, Ind_I_L );
  vec nu_l_s_I_R=g(nu_l_s, Ind_R, Ind_I_R );

  vec nu_r_s_I_L=g(nu_r_s, Ind_L, Ind_I_L );
  vec nu_r_s_I_R=g(nu_r_s, Ind_R, Ind_I_R );

  vec nu_r_p_I_L=h(nu_r_p, Ind_I_L );
  vec nu_l_p_I_R=h(nu_l_p, Ind_I_R );

  vec lower_bound_I_L=g(lower_bound, Ind_L, Ind_I_L );
  vec lower_bound_I_R=g(lower_bound, Ind_R, Ind_I_R );


  // Initialize result vector
  vec Stim_L(b_l_I_L.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_l_I_L.size(); ++i) {
    // Perform integration for current parameter values
    Stim_L[i] = (integrate_integral_nu_stop(sigma,SSD_I_L(i),DEL,DEL_s,stop_param,
                                            b_l_I_L(i),nu_l_I_L(i),nu_l_s_I_L(i),
                                            b_r_I_L(i),nu_r_p_I_L(i),nu_r_s_I_L(i),
                                            nu_l_I_L(i),nu_l_squared_I_L(i),
                                            penal_param(0),penal_param(1),lower_bound_I_L(i),upper_bound,lt));

  }



  // Initialize result vector
  vec Stim_R(b_r_I_R.size());

  // Loop over parameter vectors
  // #pragma omp parallel num_threads(2)
  for (unsigned i = 0; i < b_r_I_R.size(); ++i) {
    // Perform integration for current parameter values
    Stim_R[i]=(integrate_integral_nu_stop(sigma,SSD_I_R(i),DEL,DEL_s,stop_param,
                                          b_r_I_R(i),nu_r_I_R(i),nu_r_s_I_R(i),
                                          b_l_I_R(i),nu_l_p_I_R(i),nu_l_s_I_R(i),
                                          nu_r_I_R(i),nu_r_squared_I_R(i),
                                          penal_param(0),penal_param(1),lower_bound_I_R(i),upper_bound,lt));


  }


  vec A_L=exp(prob_param(1)-integral_likelihood(0));
  vec A_R=exp(prob_param(1)-integral_likelihood(1));


  vec Stim_L_final=Stim_L%A_L;
  vec Stim_R_final=Stim_R%A_R;


  field<vec> result(2);

  result(0)=Stim_L_final;
  result(1)=Stim_R_final;



  return result;
}



////////////////////////////////////////////////////////////////////////////////////// // Final Integration Vectors //////////////////////////////////////////////////////////////////////////


// Function-1


// likelihood Sum

field <vec> update_integral_likelihood_sum(const field<field<vec>>  &Inhib_likelihood){


  // Rcpp::Rcout<<"AR"<< endl;

  unsigned N=Inhib_likelihood.n_elem;

  vec Inhib_sum(N);


  for (unsigned i = 0; i < N; ++i) {


    Inhib_sum(i)=accu(Inhib_likelihood(i)(0))+accu(Inhib_likelihood(i)(1));

  }

  field<vec> result(1);

  result(0)=Inhib_sum;

  return result;

}


//likelihood


field<field<vec>> update_likelihood(const double sigma,
                                    const field<vec> &SSD, const vec &DEL,const vec &DEL_s,
                                    const mat &stop_param,const mat &penal_param,const mat &prob_param,
                                    const field<vec> &b_l, const field<vec> &b_r,
                                    const field<vec> &nu_l, const field<vec> &nu_r, const field<vec> &nu_l_s,
                                    const field<vec> &nu_r_s, const field<vec> &nu_r_p, const field<vec> &nu_l_p,
                                    const field<uvec> &Ind_L, const field<uvec> &Ind_R, const field<uvec> &Ind_I_L,
                                    const field<uvec> &Ind_I_R, const field<vec> &lower_bound,
                                    const double upper_bound, unsigned nparall, const bool lt = 1) {


  // Rcpp::Rcout << "AS" << endl;

  unsigned N=b_l.n_elem;

  field<field<vec>> Inhib_likelihood(N);
  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall)
  for (unsigned i = 0; i < N; ++i) {

    Inhib_likelihood(i) = integral_likelihood(sigma,SSD(i),DEL(i),DEL_s(i),
                     stop_param.col(i),penal_param.col(i),prob_param.col(i),
                     b_l(i), nu_l(i), nu_l_s(i), b_r(i), nu_r_p(i), nu_r_s(i),
                     nu_r(i), nu_l_p(i),
                     Ind_L(i), Ind_R(i), Ind_I_L(i), Ind_I_R(i),
                     lower_bound(i), upper_bound, lt);
  }
  omp_set_num_threads(6);

  return Inhib_likelihood;
}








//Gama Gradient

field <field<field<vec>>>update_gama_grad(const double sigma,const field<field<vec>> &Int_dens,
                                          const field <vec> &SSD,const vec &DEL,const vec &DEL_s,
                                          const mat &stop_param,const mat &penal_param,const mat &prob_param,
                                          const field <vec> &b_l,const field <vec> &b_r,
                                          const field <vec> &nu_l,const field <vec> &nu_r,
                                          const field <vec> &nu_l_s,const field <vec> &nu_r_s,
                                          const field <vec> &nu_l_p,const field <vec> &nu_r_p,
                                          const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                                          const field <uvec> &Ind_I_L,const field <uvec>&Ind_I_R,
                                          const field <vec> &lower_bound,const double upper_bound,
                                          unsigned nparall,const bool lt =1){



  // Rcpp::Rcout<<"AT"<< endl;

  unsigned N=b_l.n_elem;


  field<field<vec>>gama_LS(N);
  field<field<vec>>gama_RS(N);

  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall)
  for (unsigned i = 0; i < N; ++i) {

    gama_LS(i)=integrate_gama_Left_Stim(Int_dens(i),sigma,
            SSD(i),DEL(i),DEL_s(i),stop_param.col(i),penal_param.col(i),prob_param.col(i),
            b_l(i), nu_l(i),nu_l_s(i),
            b_r(i), nu_r_p(i),nu_r_s(i),
            Ind_L(i),Ind_I_L(i),
            lower_bound(i),upper_bound,lt);


    gama_RS(i)=integrate_gama_Right_Stim(Int_dens(i),sigma,
            SSD(i),DEL(i),DEL_s(i),stop_param.col(i),penal_param.col(i),prob_param.col(i),
            b_l(i),nu_l_p(i),nu_l_s(i),
            b_r(i),nu_r(i),nu_r_s(i),
            Ind_R(i),Ind_I_R(i),
            lower_bound(i),upper_bound,lt);


  }

  omp_set_num_threads(6);


  field <field<field<vec>>> result(2);


  result(0)=gama_LS;
  result(1)=gama_RS;
  return result;

}



//Beta Gradient

field <field<field<vec>>>update_beta_grad(const double sigma,
                                          const field<field<vec>> &Int_dens,
                                          const field <vec> &SSD,const vec &DEL,const vec &DEL_s,
                                          const mat &stop_param,const mat &penal_param,const mat &prob_param,
                                          const field <vec> &b_l,const field <vec> &b_r,
                                          const field <vec> &nu_l,const field <vec> &nu_r,
                                          const field <vec> &nu_l_s,const field <vec> &nu_r_s,
                                          const field <vec> &nu_l_p,const field <vec> &nu_r_p,
                                          const field <vec> &nu_l_squared,const field <vec> &nu_r_squared,
                                          const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                                          const field <uvec> &Ind_I_L,const field <uvec>&Ind_I_R,
                                          const field <vec> &lower_bound,const double upper_bound,
                                          unsigned nparall,const bool lt =1){



  // Rcpp::Rcout<<"AU"<< endl;

  unsigned N=b_l.n_elem;


  field<field<vec>>beta_l_Stim(N);
  field<field<vec>>beta_r_Stim(N);

  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall)
  for (unsigned i = 0; i < N; ++i) {

    beta_l_Stim(i)=  integrate_beta_l(Int_dens(i),sigma,
                SSD(i),DEL(i),DEL_s(i),stop_param.col(i),penal_param.col(i),prob_param.col(i),
                b_l(i),b_r(i),nu_l(i),nu_r(i),nu_l_s(i),nu_r_s(i),nu_r_p(i),nu_l_p(i),
                nu_l_squared(i),
                Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                lower_bound(i),upper_bound,lt);



    beta_r_Stim(i)=  integrate_beta_r(Int_dens(i),sigma,
                SSD(i),DEL(i),DEL_s(i),stop_param.col(i),penal_param.col(i),prob_param.col(i),
                b_l(i),b_r(i),nu_l(i),nu_r(i),nu_l_s(i),nu_r_s(i),nu_r_p(i),nu_l_p(i),
                nu_r_squared(i),
                Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                lower_bound(i),upper_bound,lt);

  }
  omp_set_num_threads(6);

  field <field<field<vec>>> result(2);


  result(0)=beta_l_Stim;
  result(1)=beta_r_Stim;

  return result;

}



/////////////////////////////////////////////////////////////////////////// Gradient Functions ////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////// Gama //////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////// Gama //////////////////////////////////////////////////////////////////////////////////////////////


// Derivative of log dense w.r.t. gama (Go Trial)


mat deriv_log_dens_gama(const mat &t,const vec &tau,const double sigma,
                        const vec &nu, const vec &b){

  //Rcpp::Rcout<<"AV"<< endl;

  if (t.n_rows != tau.n_elem || t.n_rows != nu.n_elem)
    Rcpp::stop("t.n_rows != tau.n_elem || t.n_rows != nu.n_elem in --g_log_dens_theta-- ");

  vec C=1-((exp(2*b))/(sigma*tau))+((exp(b+nu))/(sigma));

  return (t.each_col()%C).t(); //length(C)


}


// deriv_multiplier

inline vec deriv_multiplier(const vec &SSD_diff,const vec &rat,const vec &q2){

  return (SSD_diff%(1+(rat%q2)));

}




// derivative of exponent w.r.t. gama

inline vec deriv_exp_gama(double sigma,const vec &tau,const vec &b,const vec &m){

  return (exp(b)/(sqrt(sigma*tau)));

}



// Derivative of log dense w.r.t. gama (Go Proces-Failed Stop Trial)

mat deriv_log_dens_gama_s(const mat &t,const vec &tau,const vec &tau_s,const double sigma,
                          const vec &SSD, const vec &b,const vec &nu1,const vec &nu2,
                          const double DEL,const double DEL_s,const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;


  // Defining necessary vectors

  vec m=(exp(nu1)%(SSD+DEL_s-DEL))+(exp(nu2)%tau_s);

  vec q1 = exponent_1(sigma,tau,m,b);

  vec q2 = exponent_2(sigma,tau,m,b);

  vec rat=phi_rat(q2,lt);

  vec SSD_diff=SSD_Dif(sigma,tau,SSD,nu1,nu2,DEL,DEL_s);


  vec deriv_gama=deriv_exp_gama(sigma,tau,b,m);

  // calculating small vectors

  vec c1=q1%deriv_gama;

  vec deriv_gama_B=deriv_multiplier(SSD_diff,rat,q2)%(-deriv_gama);

  vec B=1+(rat%SSD_diff);

  vec C=1-c1+(deriv_gama_B/B);


  // Multiplying by t
  mat result= (t.each_col()%C);


  return result.t();


}



// Cube consisting of derivative of log dense w.r.t. gama for Correct and Incorrect responses (Go Trial)


cube grad_cube(const mat &t,const vec &tau,const double sigma,const vec &nu_CR,const vec &nu_IR,
               const vec &b_CR,const vec &b_IR){

  //Rcpp::Rcout<<"AX"<< endl;

  mat CR=deriv_log_dens_gama(t,tau,sigma,nu_CR,b_CR);
  mat IR=deriv_log_dens_gama(t,tau,sigma,nu_IR,b_IR);

  return join_slices(CR,IR);

}


// Cube consisting of derivative of log dense w.r.t. gama for Correct and Incorrect responses (Go Trial- Stop Process)

cube grad_cube_s(const mat &t,const vec &tau,const vec &tau_s,const double sigma,const vec &SSD,
                 const vec &b_CR,const vec &nu_1_CR,const vec &nu_2_CR,
                 const vec &b_IR,const vec &nu_1_IR,const vec &nu_2_IR,const double DEL, const double DEL_s){

  //Rcpp::Rcout<<"AY"<< endl;


  mat CR=deriv_log_dens_gama_s(t,tau,tau_s,sigma,SSD,b_CR,nu_1_CR,nu_2_CR,DEL,DEL_s);
  mat IR=deriv_log_dens_gama_s(t,tau,tau_s,sigma,SSD,b_IR,nu_1_IR,nu_2_IR,DEL,DEL_s);

  return join_slices(CR,IR);

}






// Gradient function for Gama

mat grad_gama(const mat &t, const vec &tau,const vec &tau_s, const double sigma,const vec &SSD,
              const double DEL, const double DEL_s,
              const vec &nu_l, const vec &nu_r,const vec &b_l, const vec &b_r,const vec &nu_l_p, const vec &nu_r_p,
              const vec &nu_l_s, const vec &nu_r_s,
              const vec &weight_LCR, const vec &weight_LIR,const vec &weight_RCR, const vec &weight_RIR,
              const vec &weight_LCR_G_FS, const vec &weight_LIR_G_FS,const vec &weight_RCR_G_FS, const vec &weight_RIR_G_FS,
              const vec &weight_LCR_S, const vec &weight_LIR_S,const vec &weight_RCR_S, const vec &weight_RIR_S,

              const uvec &Ind_L,const uvec &Ind_R,
              const uvec &Ind_LCR,const uvec &Ind_LIR,const uvec &Ind_RCR,const uvec &Ind_RIR,
              const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,const uvec &Ind_S_RCR,const uvec &Ind_S_RIR,
              const uvec &Ind_I_L,const uvec &Ind_I_R,

              const vec &diff_X_LCR,const vec &diff_Y_LCR,const vec &diff_X_LIR,const vec &diff_Y_LIR,
              const vec &diff_X_RCR,const vec &diff_Y_RCR,const vec &diff_X_RIR,const vec &diff_Y_RIR,
              const field<vec> &gama_LS,const field<vec> &gama_RS) {



  // cout<<"AZ"<< endl;


  //Go Trial
  cube grad_L = grad_cube(t.rows(Ind_L), tau(Ind_L), sigma, nu_l(Ind_L),nu_r_p, b_l(Ind_L),b_r(Ind_L));
  cube grad_R = grad_cube(t.rows(Ind_R), tau(Ind_R), sigma, nu_r(Ind_R),nu_l_p, b_r(Ind_R), b_l(Ind_R));


  mat LCR = grad_L.slice(0);
  mat LIR = grad_L.slice(1);

  mat RCR = grad_R.slice(0);
  mat RIR = grad_R.slice(1);




  //Go Process-Stop Trial


  cube grad_L_S=grad_cube_s(t.rows(Ind_L),tau(Ind_L),tau_s(Ind_L),sigma,SSD(Ind_L),b_l(Ind_L),nu_l(Ind_L),nu_l_s(Ind_L),
                            b_r(Ind_L),nu_r_p,nu_r_s(Ind_L),DEL,DEL_s);

  cube grad_R_S=grad_cube_s(t.rows(Ind_R),tau(Ind_R),tau_s(Ind_R),sigma,SSD(Ind_R),b_r(Ind_R),nu_r(Ind_R),nu_r_s(Ind_R),
                            b_l(Ind_R),nu_l_p,nu_l_s(Ind_R),DEL,DEL_s);


  mat LCR_S = grad_L_S.slice(0);
  mat LIR_S = grad_L_S.slice(1);

  mat RCR_S = grad_R_S.slice(0);
  mat RIR_S = grad_R_S.slice(1);




  // Stop Process Inhibit trial

  mat t_l=t.rows(Ind_L);
  mat t_r=t.rows(Ind_R);

  mat t_I_L=t_l.rows(Ind_I_L);
  mat t_I_R=t_r.rows(Ind_I_R);

  vec gama_l_LS=(t_I_L.t())*gama_LS(0);

  vec gama_l_RS=(t_I_R.t())*gama_RS(0);


  vec gama_r_LS=(t_I_L.t())*gama_LS(1);

  vec gama_r_RS=(t_I_R.t())*gama_RS(1);



  mat gama_l_grad=sum(LCR.cols(Ind_LCR),1)-((LCR.cols(Ind_LIR))*weight_LIR)-
    ((RIR.cols(Ind_RCR))*weight_RCR)+ sum(RIR.cols(Ind_RIR),1)+

    ((LCR.cols(Ind_S_LCR))*diff_X_LCR)+((LCR_S.cols(Ind_S_LCR))*diff_Y_LCR)-
    ((LCR.cols(Ind_S_LIR))*(weight_LIR_G_FS%diff_X_LIR))-((LCR_S.cols(Ind_S_LIR))*(weight_LIR_S%diff_Y_LIR))-
    ((RIR.cols(Ind_S_RCR))*(weight_RCR_G_FS%diff_X_RCR))-((RIR_S.cols(Ind_S_RCR))*(weight_RCR_S%diff_Y_RCR))+
    ((RIR.cols(Ind_S_RIR))*diff_X_RIR)+((RIR_S.cols(Ind_S_RIR))*diff_Y_RIR)+
    gama_l_LS+gama_l_RS;



  mat gama_r_grad = sum(RCR.cols(Ind_RCR),1)-((RCR.cols(Ind_RIR))*weight_RIR)+
    sum(LIR.cols(Ind_LIR),1)-((LIR.cols(Ind_LCR))*weight_LCR)+

    ((RCR.cols(Ind_S_RCR))*diff_X_RCR)+((RCR_S.cols(Ind_S_RCR))*diff_Y_RCR)-
    ((RCR.cols(Ind_S_RIR))*(weight_RIR_G_FS%diff_X_RIR))-((RCR_S.cols(Ind_S_RIR))*(weight_RIR_S%diff_Y_RIR))+
    ((LIR.cols(Ind_S_LIR))*diff_X_LIR)+((LIR_S.cols(Ind_S_LIR))*diff_Y_LIR)-
    ((LIR.cols(Ind_S_LCR))*(weight_LCR_G_FS%diff_X_LCR))-((LIR_S.cols(Ind_S_LCR))*(weight_LCR_S%diff_Y_LCR))+
    gama_r_LS+gama_r_RS;


  // Rcpp::Rcout << "--- Checking components of gama_r_grad ---\n";
  //
  //
  // check_nan_inf(gama_r_LS, "gama_r_LS");
  // check_nan_inf(gama_r_RS, "gama_r_RS");
  //
  // Rcpp::Rcout << "--- Done checking gama_r_grad ---\n";
  //
  mat result=join_rows(gama_l_grad,gama_r_grad);


  return result;

}

//Gama_grad_prior

mat grad_gama_prior(const mat &gama,double eta_b_l,double eta_b_r,
                    const vec &mu_gl,const mat &sigma_gl_inv,
                    const vec &mu_gr,const mat &sigma_gr_inv){



  vec gama_l_prior=-(eta_b_l*(sigma_gl_inv*gama.col(0)));

  vec gama_r_prior=-(eta_b_r*(sigma_gr_inv*gama.col(1)));

  mat result=join_rows(gama_l_prior,gama_r_prior);

  return result;

}


//Derivative of Random effect prior

mat grad_Ind_prior(const mat &gama,
                   const vec &S_gama_l,const vec &S_gama_r,
                   const double alpha_gama_l,const double alpha_gama_r){

  // Rcpp::Rcout<<"AZ"<< endl;


  vec gama_l_prior=-(gama.col(0)*exp(-alpha_gama_l))/(exp(S_gama_l));

  vec gama_r_prior=-(gama.col(1)*exp(-alpha_gama_r))/(exp(S_gama_r));

  mat result=join_rows(gama_l_prior,gama_r_prior);

  return result;

}


////////////////////////////////////////////////////////////////////////////////////////////////////// Beta //////////////////////////////////////////////////////////////////////////////////////////////

// Common quantity for beta derivative (without penalty)

vec com_quan(const vec &tau, const vec &b, const vec &nu,double sigma){

  //Rcpp::Rcout<<"AAA"<< endl;

  if (tau.n_elem !=nu.n_elem)
    Rcpp::stop("tau.n_elem !=nu.n_elem --com_quan-- ");

  return((exp(nu+b)-(exp(2*nu)%tau))/sigma);   // nu=t*beta for no penalty // nu=t*beta_r-penalty(t*beta_s))

}




// beta derivative (without penalty) (Go Trial)

mat deriv_log_dens_beta(const mat &t,const vec &tau,const double sigma,
                        const vec &nu, const vec &b){

  // Rcpp::Rcout<<"AAC"<< endl;

  return (t.each_col()%com_quan(tau,b,nu,sigma)).t(); //length(C)

}




// Derivative of penalty w.r.t beta


vec g_beta_stim_penal(const vec &nu, const vec &nu_squared,double lambda_prime,double alpha_prime){


  //Rcpp::Rcout<<"AAE"<< endl;

  vec X=(exp(2*lambda_prime))*nu_squared;
  double Y=exp(2*alpha_prime);
  vec hpot=sqrt(X+Y);

  vec numerator = exp(lambda_prime)-((exp(2*lambda_prime)*nu)/hpot);

  vec denominator=square((-(exp(lambda_prime)*nu)+hpot));

  return numerator/denominator;


}



// beta derivative (with penalty) (Go Trial)


mat deriv_log_dens_beta_p(const mat &t,const vec &nu,const vec &nu_s,const vec &nu_s_squared,
                          const vec &tau,const double sigma,const vec &b,double lambda_prime,double alpha_prime){


  //Rcpp::Rcout<<"AAF"<< endl;

  vec D= (com_quan(tau,b,nu,sigma)%(-g_beta_stim_penal(nu_s,nu_s_squared,lambda_prime,alpha_prime)));

  return (t.each_col()%D).t();

}



// derivative of exponent w.r.t. beta (without penalty)

inline vec deriv_exp_beta(double sigma,const vec &tau,const vec &m){

  return (m/(sqrt(sigma*tau)));

}




// beta derivative (without penalty) (Go Proces-Failed Stop Trial)

mat deriv_log_dens_beta_s(const mat &t,const vec &tau,const vec &tau_s,const double sigma,
                          const vec &SSD, const vec &b,const vec &nu1,const vec &nu2,const double DEL, const double DEL_s,
                          const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;


  // Defining necessary vectors
  vec m=(exp(nu1)%(SSD+DEL_s-DEL))+(exp(nu2)%tau_s);

  vec q1=exponent_1(sigma,tau,m,b);

  vec q2 = exponent_2(sigma,tau,m,b);

  vec rat=phi_rat(q2,lt);

  vec SSD_diff=SSD_Dif(sigma,tau,SSD,nu1,nu2,DEL,DEL_s);


  vec deriv_beta=deriv_exp_beta(sigma,tau,m);

  // calculating small vectors

  vec c1=q1%deriv_beta;

  vec deriv_beta_B=(deriv_multiplier(SSD_diff,rat,q2)%(-deriv_beta))+(rat%SSD_diff);

  vec B=1+(rat%SSD_diff);

  vec C=c1+(deriv_beta_B/B);


  // Multiplying by t
  mat result= (t.each_col()%C);

  return result.t();


}




// derivative of exponent w.r.t. beta (with penalty)

inline vec deriv_exp_beta_p(const double sigma,const vec &tau,const vec &SSD,
                            const vec &nu1,const vec &nu_stim, const vec &nu_squared,
                            double lambda_prime,double alpha_prime,const double DEL,const double DEL_s){

  return (((SSD+DEL_s-DEL)%exp(nu1)%g_beta_stim_penal(nu_stim,nu_squared,lambda_prime,alpha_prime))/(sqrt(sigma*tau)));

}



// beta derivative (without penalty) (Go Proces-Failed Stop Trial)

mat deriv_log_dens_beta_p_s(const mat &t,const vec &tau,const vec &tau_s,const double sigma,
                            const vec &SSD, const vec &b,const vec &nu1,const vec &nu2,
                            const vec &nu_stim, const vec &nu_squared,
                            double lambda_prime,double alpha_prime,const double DEL, const double DEL_s,const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;


  // Defining necessary vectors
  vec m=(exp(nu1)%(SSD+DEL_s-DEL))+(exp(nu2)%tau_s);


  vec q1=exponent_1(sigma,tau,m,b);

  vec q2 = exponent_2(sigma,tau,m,b);

  vec rat=phi_rat(q2,lt);

  vec SSD_diff=SSD_Dif(sigma,tau,SSD,nu1,nu2,DEL,DEL_s);


  vec deriv_beta=deriv_exp_beta_p(sigma,tau,SSD,nu1,nu_stim,nu_squared,lambda_prime,alpha_prime,DEL,DEL_s);



  // calculating small vectors

  vec deriv_beta_B=deriv_multiplier(SSD_diff,rat,q2)+(2*rat);

  vec B=1+(rat%SSD_diff);

  vec C=(-q1+(deriv_beta_B/B))%deriv_beta;


  // Multiplying by t
  mat result= (t.each_col()%C);


  return result.t();


}



// Gradient Function for Beta

mat grad_beta(const mat &t,const vec &tau,const vec &tau_s,const double sigma,const vec &SSD,const double DEL,const double DEL_s,
              const vec &nu_l,const vec &nu_l_p,const vec &nu_r,const vec &nu_r_p,
              const vec &nu_l_s,const vec &nu_r_s,
              const vec &b_l,const vec &b_r,
              const vec &nu_l_squared,const vec &nu_r_squared,
              const vec &penal_param,const uvec &Ind_L,const uvec &Ind_R,
              const uvec &Ind_LCR,const uvec &Ind_LIR,const uvec &Ind_RIR,const uvec &Ind_RCR,
              const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,const uvec &Ind_S_RIR,const uvec &Ind_S_RCR,

              const uvec &Ind_I_L,const uvec &Ind_I_R,


              const field<vec> &beta_l_Stim,const field<vec> &beta_r_Stim,

              const vec &weight_LCR, const vec &weight_LIR,const vec &weight_RCR, const vec &weight_RIR,
              const vec &weight_LCR_G_FS, const vec &weight_LIR_G_FS,const vec &weight_RCR_G_FS, const vec &weight_RIR_G_FS,
              const vec &weight_LCR_S, const vec &weight_LIR_S,const vec &weight_RCR_S, const vec &weight_RIR_S,
              const vec &diff_X_LCR,const vec &diff_Y_LCR,const vec &diff_X_LIR,const vec &diff_Y_LIR,
              const vec &diff_X_RCR,const vec &diff_Y_RCR,const vec &diff_X_RIR,const vec &diff_Y_RIR,const bool lt=1){




  // Rcpp::Rcout<<"AAH"<< endl;

  //Go Trial
  mat LS_CR=deriv_log_dens_beta(t.rows(Ind_L),tau(Ind_L),sigma,nu_l(Ind_L),b_l(Ind_L));
  mat LS_P=deriv_log_dens_beta_p(t.rows(Ind_L),nu_r_p,nu_l(Ind_L),nu_l_squared(Ind_L),
                                 tau(Ind_L),sigma,b_r(Ind_L),penal_param(0),penal_param(1));
  mat RS=deriv_log_dens_beta(t.rows(Ind_R),tau(Ind_R),sigma,nu_l_p,b_l(Ind_R));




  // Failed Stop Trial
  mat LS_CR_S=deriv_log_dens_beta_s(t.rows(Ind_L),tau(Ind_L),tau_s(Ind_L),sigma,SSD(Ind_L),b_l(Ind_L),nu_l(Ind_L),nu_l_s(Ind_L),DEL,DEL_s,lt);



  mat LS_P_S=deriv_log_dens_beta_p_s(t.rows(Ind_L),tau(Ind_L),tau_s(Ind_L),sigma,SSD(Ind_L),b_r(Ind_L),nu_r_p,nu_r_s(Ind_L),
                                     nu_l(Ind_L),nu_l_squared(Ind_L),
                                     penal_param(0),penal_param(1),DEL,DEL_s,lt);

  mat RS_S=deriv_log_dens_beta_s(t.rows(Ind_R),tau(Ind_R),tau_s(Ind_R),sigma,SSD(Ind_R),b_l(Ind_R),nu_l_p,nu_l_s(Ind_R),DEL,DEL_s,lt);






  // Inhibit trial

  mat t_l=t.rows(Ind_L);
  mat t_r=t.rows(Ind_R);

  mat t_I_L=t_l.rows(Ind_I_L);
  mat t_I_R=t_r.rows(Ind_I_R);



  vec beta_l_LS=(t_I_L.t())*beta_l_Stim(0);

  vec beta_l_RS=(t_I_R.t())*beta_l_Stim(1);


  vec beta_r_LS=(t_I_L.t())*beta_r_Stim(0);

  vec beta_r_RS=(t_I_R.t())*beta_r_Stim(1);








  // beta_l_Gradient
  vec beta_l_grad = sum(LS_CR.cols(Ind_LCR),1)-((LS_P.cols(Ind_LCR))*weight_LCR)+
    sum(LS_P.cols(Ind_LIR),1)-((LS_CR.cols(Ind_LIR))*weight_LIR)+
    sum(RS.cols(Ind_RIR),1)-((RS.cols(Ind_RCR))*weight_RCR)+


    ((LS_CR.cols(Ind_S_LCR))*diff_X_LCR)-((LS_P.cols(Ind_S_LCR))*(weight_LCR_G_FS%diff_X_LCR))+
    ((LS_CR_S.cols(Ind_S_LCR))*diff_Y_LCR)-((LS_P_S.cols(Ind_S_LCR))*(weight_LCR_S%diff_Y_LCR))+

    ((LS_P.cols(Ind_S_LIR))*diff_X_LIR)-((LS_CR.cols(Ind_S_LIR))*(weight_LIR_G_FS%diff_X_LIR))+
    ((LS_P_S.cols(Ind_S_LIR))*diff_Y_LIR)-((LS_CR_S.cols(Ind_S_LIR))*(weight_LIR_S%diff_Y_LIR))+

    ((RS.cols(Ind_S_RIR))*diff_X_RIR)+sum(RS_S.cols(Ind_S_RIR)*diff_Y_RIR)-
    ((RS.cols(Ind_S_RCR))*(weight_RCR_G_FS%diff_X_RCR))-((RS_S.cols(Ind_S_RCR))*(weight_RCR_S%diff_Y_RCR))+
    beta_l_LS+beta_l_RS;




  // Go Trial
  mat LS=deriv_log_dens_beta(t.rows(Ind_L),tau(Ind_L),sigma,nu_r_p,b_r(Ind_L));
  mat RS_CR=deriv_log_dens_beta(t.rows(Ind_R),tau(Ind_R),sigma,nu_r(Ind_R),b_r(Ind_R));
  mat RS_P=deriv_log_dens_beta_p(t.rows(Ind_R),nu_l_p,nu_r(Ind_R),nu_r_squared(Ind_R),
                                 tau(Ind_R),sigma,b_l(Ind_R),penal_param(0),penal_param(1));



  // Failed Stop Trial

  mat LS_S=deriv_log_dens_beta_s(t.rows(Ind_L),tau(Ind_L),tau_s(Ind_L),sigma,SSD(Ind_L),b_r(Ind_L),nu_r_p,nu_r_s(Ind_L),DEL,DEL_s,lt);


  mat RS_CR_S=deriv_log_dens_beta_s(t.rows(Ind_R),tau(Ind_R),tau_s(Ind_R),sigma,SSD(Ind_R),b_r(Ind_R),nu_r(Ind_R),nu_r_s(Ind_R),DEL,DEL_s,lt);


  mat RS_P_S=deriv_log_dens_beta_p_s(t.rows(Ind_R),tau(Ind_R),tau_s(Ind_R),sigma,SSD(Ind_R),b_l(Ind_R),nu_l_p,nu_l_s(Ind_R),
                                     nu_r(Ind_R),nu_r_squared(Ind_R),
                                     penal_param(0),penal_param(1),DEL,DEL_s,lt);





  // beta_r_Gradient
  vec beta_r_grad = sum(LS.cols(Ind_LIR),1)-((LS.cols(Ind_LCR))*weight_LCR)+
    sum(RS_P.cols(Ind_RIR),1)-((RS_CR.cols(Ind_RIR))*weight_RIR)+
    sum(RS_CR.cols(Ind_RCR),1)-((RS_P.cols(Ind_RCR))*weight_RCR)-

    ((LS.cols(Ind_S_LCR))*(weight_LCR_G_FS%diff_X_LCR))-((LS_S.cols(Ind_S_LCR))*(weight_LCR_S%diff_Y_LCR))+
    ((LS.cols(Ind_S_LIR))*diff_X_LIR)+((LS_S.cols(Ind_S_LIR))*diff_Y_LIR)+

    ((RS_P.cols(Ind_S_RIR))*diff_X_RIR)-((RS_CR.cols(Ind_S_RIR))*(weight_RIR_G_FS%diff_X_RIR))+
    ((RS_P_S.cols(Ind_S_RIR))*diff_Y_RIR)-((RS_CR_S.cols(Ind_S_RIR))*(weight_RIR_S%diff_Y_RIR))+

    ((RS_CR.cols(Ind_S_RCR))*diff_X_RCR)-((RS_P.cols(Ind_S_RCR))*(weight_RCR_G_FS%diff_X_RCR))+
    ((RS_CR_S.cols(Ind_S_RCR))*diff_Y_RCR)-((RS_P_S.cols(Ind_S_RCR))*(weight_RCR_S%diff_Y_RCR))+
    beta_r_LS+beta_r_RS;

  return join_rows(beta_l_grad, beta_r_grad);



}


// beta_grad_prior

mat grad_beta_prior(const mat &beta,double eta_nu_l,double eta_nu_r,
                    const vec &mu_bl,const mat &sigma_bl_inv,const vec &mu_br,const mat &sigma_br_inv){


  mat beta_L_prior=-(eta_nu_l*-sigma_bl_inv*beta.col(0));
  mat beta_R_prior=-(eta_nu_r*sigma_br_inv*beta.col(1));

  return join_rows(beta_L_prior, beta_R_prior);

}







///////////////////////////////////////////////////////////////////////////////// Penalty Parameter ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Derivative of penalty w.r.t lambda (Go Process)


vec g_lambda_penal_com(const vec &tau,double sigma, const vec &b,const vec &nu,
                       const vec &nu_stim,const vec &nu_squared, double lambda_prime, double alpha_prime){

  //Rcpp::Rcout<<"AAI"<< endl;


  vec X=(exp(2*lambda_prime))*nu_squared;
  double Y=exp(2*alpha_prime);
  vec hpot=sqrt(X+Y);

  vec numerator = nu_stim- ((exp(lambda_prime)*nu_squared)/hpot);

  vec denominator=square((-(exp(lambda_prime)*nu_stim)+hpot));

  vec result=(numerator/denominator)*exp(lambda_prime);

  return (-com_quan(tau, b, nu,sigma)%result);

}



// Derivative of penalty w.r.t alpha (Go Trial)

vec g_alpha_penal_com(const vec &tau, const double sigma, const vec &b,const vec &nu,
                      const vec &nu_stim,const vec &nu_squared,double lambda_prime, double alpha_prime){

  // Rcpp::Rcout<<"AAK"<< endl;

  vec X=(exp(2*lambda_prime))*nu_squared;
  double Y=exp(2*alpha_prime);
  vec hpot=sqrt(X+Y);

  vec numerator = -(exp(alpha_prime)/hpot);

  vec denominator=square(-(exp(lambda_prime)*nu_stim)+hpot);

  vec result=(numerator/denominator)*exp(alpha_prime);

  return (-com_quan(tau, b, nu,sigma)%result);

}





// Penalty derivative (without penalty) (Go Process-Failed Stop Trial)

vec g_penal_com_s(const vec &tau, const vec &tau_s, const double sigma,
                  const vec &SSD, const vec &b, const vec &nu1, const vec &nu2,
                  const double DEL, const double DEL_s, const bool lt = 1) {

  vec m = (exp(nu1) % (SSD + DEL_s - DEL)) + (exp(nu2) % tau_s);

  vec q1 = exponent_1(sigma, tau, m, b);

  vec q2 = exponent_2(sigma, tau, m, b);

  vec rat = phi_rat(q2, lt);

  vec SSD_diff = SSD_Dif(sigma, tau, SSD, nu1, nu2, DEL, DEL_s);

  vec w = tau_s / sqrt(sigma * tau);

  vec c1 = q1 % w;

  vec deriv_lambda_B = (deriv_multiplier(SSD_diff, rat, q2) % w) -((2 * rat % (SSD + DEL_s-DEL)) / sqrt(sigma * tau));

  vec B = 1 + (rat % SSD_diff);

  vec result = ((-c1 + (deriv_lambda_B / B)) % exp(nu2));


  return result;
}




// Derivative of beta penalty w.r.t lambda (Go Process-Failed Stop Trial)


vec g_beta_stim_penal_lambda(const vec &nu_stim,const vec &nu_squared,double lambda_prime,double alpha_prime){

  vec X=(exp(2*lambda_prime))*nu_squared;
  double Y=exp(2*alpha_prime);
  vec hpot=sqrt(X+Y);

  vec numerator = nu_stim- ((exp(lambda_prime)*nu_squared)/hpot);

  vec denominator=square((-(exp(lambda_prime)*nu_stim)+hpot));

  vec result=(numerator/denominator)*exp(lambda_prime);

  return result;
}




// Derivative of beta penalty w.r.t alpha (Go Process-Failed Stop Trial)

vec g_beta_stim_penal_alpha(const vec &nu_stim,const vec &nu_squared,double lambda_prime,double alpha_prime){

  vec X=(exp(2*lambda_prime))*nu_squared;
  double Y=exp(2*alpha_prime);
  vec hpot=sqrt(X+Y);

  vec numerator = -(exp(alpha_prime)/hpot);

  vec denominator=square(-(exp(lambda_prime)*nu_stim)+hpot);

  vec result=(numerator/denominator)*exp(alpha_prime);

  return result;
}



// derivative of exponent w.r.t. penalty (with penalty)

inline vec deriv_exp_penal_p(double sigma,const vec &tau,const vec &tau_s,const vec &SSD,const vec &nu1,const vec &nu2,
                             const vec &beta_deriv_penalty,const double nu_stop_deriv_penalty,const double DEL,const double DEL_s){


  return ((((SSD+DEL_s-DEL) % exp(nu1) % beta_deriv_penalty) + (tau_s % exp(nu2) * nu_stop_deriv_penalty)) / sqrt(sigma * tau));

}



// derivative of nu_diff w.r.t. lambda (with penalty)

inline vec deriv_nu_diff_penal_p(double sigma,const vec &SSD,const vec &nu1,const vec &nu2,
                                 const vec &beta_deriv_penalty,const double nu_stop_deriv_penalty,
                                 const double DEL, const double DEL_s){

        return ((SSD+DEL_s-DEL)%((exp(nu1)%beta_deriv_penalty)-(exp(nu2)*nu_stop_deriv_penalty)));


}




// lambda derivative (with penalty) (Go Proces-Failed Stop Trial)
vec g_penal_com_p_s(const vec &tau,const vec &tau_s,const double sigma,
                    const vec &SSD, const vec &b,const vec &nu1,const vec &nu2,
                    const vec &beta_deriv_penalty,const double nu_stop_deriv_penalty,
                    const double DEL, const double DEL_s,const bool lt=1){


  // Defining necessary vectors
  vec m=(exp(nu1)%(SSD+DEL_s-DEL))+(exp(nu2)%tau_s);


  vec q1=exponent_1(sigma,tau,m,b);

  vec q2 = exponent_2(sigma,tau,m,b);

  vec rat=phi_rat(q2,lt);

  vec SSD_diff=SSD_Dif(sigma,tau,SSD,nu1,nu2,DEL,DEL_s);


  vec deriv_penal_exp=deriv_exp_penal_p(sigma,tau,tau_s,SSD,nu1,nu2,
                                        beta_deriv_penalty,nu_stop_deriv_penalty,DEL,DEL_s);


  vec deriv_penal_nu_diff=deriv_nu_diff_penal_p(sigma,SSD,nu1,nu2,
                                                beta_deriv_penalty,nu_stop_deriv_penalty,DEL,DEL_s);


  // calculating small vectors

  vec c1=q1%deriv_penal_exp;

  vec deriv_penal_B=(deriv_multiplier(SSD_diff,rat,q2)%deriv_penal_exp)+((2*rat%deriv_penal_nu_diff)/sqrt(sigma*tau));

  vec B=1+(rat%SSD_diff);

  return (-c1+(deriv_penal_B/B));


}


// Gradient Function for penalty parameter


vec grad_penal_param(const vec &tau,const vec &tau_s, const double sigma,const vec &SSD,const double DEL, const double DEL_s,
                     const vec &b_l,const vec &b_r,
                     const vec &nu_l,const vec &nu_r,
                     const vec &nu_r_p,const vec &nu_l_p,
                     const vec &nu_l_squared,const vec &nu_r_squared,const vec &nu_l_s,const vec &nu_r_s,
                     const uvec &Ind_L,const uvec &Ind_R,
                     const uvec &Ind_LCR,const uvec &Ind_LIR,const uvec &Ind_RIR,const uvec &Ind_RCR,
                     const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,const uvec &Ind_S_RIR,const uvec &Ind_S_RCR,

                     const uvec &Ind_I_L,const uvec &Ind_I_R,

                     const mat &Penal_LS,const mat &Penal_RS,

                     const vec &weight_LCR,const vec &weight_RCR,

                     const vec &weight_LCR_G_FS,const vec &weight_RCR_G_FS,

                     const vec &weight_LCR_S,const vec &weight_LIR_S,
                     const vec &weight_RCR_S,const vec &weight_RIR_S,

                     const vec &penal_param,const vec &stop_param,
                     const double mu_lambda_prime,const double sigma_lambda_prime,
                     const double mu_alpha_prime,const double sigma_alpha_prime,
                     const vec &diff_X_LCR,const vec &diff_Y_LCR,const vec &diff_X_LIR,const vec &diff_Y_LIR,
                     const vec &diff_X_RCR,const vec &diff_Y_RCR,const vec &diff_X_RIR,const vec &diff_Y_RIR,const bool lt=1){




  // Rcpp::Rcout<<"AAM"<< endl;


  vec beta_deriv_lambda_nu_l=g_beta_stim_penal_lambda(nu_l(Ind_L),nu_l_squared(Ind_L),penal_param(0),penal_param(1));
  vec beta_deriv_lambda_nu_r=g_beta_stim_penal_lambda(nu_r(Ind_R),nu_r_squared(Ind_R),penal_param(0),penal_param(1));
  double nu_stop_penal_lambda=g_nu_stop_stim_penal_lambda(stop_param(1),penal_param(0),penal_param(1));

  vec beta_deriv_alpha_nu_l=g_beta_stim_penal_alpha(nu_l(Ind_L),nu_l_squared(Ind_L),penal_param(0),penal_param(1));
  vec beta_deriv_alpha_nu_r=g_beta_stim_penal_alpha(nu_r(Ind_R),nu_r_squared(Ind_R),penal_param(0),penal_param(1));
  double nu_stop_penal_alpha=g_nu_stop_stim_penal_alpha(stop_param(1), penal_param(0),penal_param(1));



  //lambda Gradient

  // Go Trial
  vec L_lambda=g_lambda_penal_com(tau(Ind_L),sigma,b_r(Ind_L),nu_r_p,
                                  nu_l(Ind_L),nu_l_squared(Ind_L),penal_param(0),penal_param(1));

  vec R_lambda=g_lambda_penal_com(tau(Ind_R),sigma,b_l(Ind_R),nu_l_p,
                                  nu_r(Ind_R),nu_r_squared(Ind_R),penal_param(0),penal_param(1));



  // Failed Stop Trial


  vec L_lambda_S=(g_penal_com_s(tau(Ind_L),tau_s(Ind_L),sigma,SSD(Ind_L),
                                b_l(Ind_L),nu_l(Ind_L),nu_l_s(Ind_L),DEL,DEL_s,lt)*nu_stop_penal_lambda);

  vec L_lambda_P_S=g_penal_com_p_s(tau(Ind_L),tau_s(Ind_L),sigma,SSD(Ind_L),
                                   b_r(Ind_L),nu_r_p,nu_r_s(Ind_L),
                                   beta_deriv_lambda_nu_l,nu_stop_penal_lambda,DEL,DEL_s,lt);

  vec R_lambda_S=(g_penal_com_s(tau(Ind_R),tau_s(Ind_R),sigma,
                                SSD(Ind_R), b_r(Ind_R),nu_r(Ind_R),nu_r_s(Ind_R),DEL,DEL_s,lt)*nu_stop_penal_lambda);

  vec R_lambda_P_S=g_penal_com_p_s(tau(Ind_R),tau_s(Ind_R),sigma,SSD(Ind_R),
                                   b_l(Ind_R),nu_l_p,nu_l_s(Ind_R),
                                   beta_deriv_lambda_nu_r,nu_stop_penal_lambda,DEL,DEL_s,lt);






  // Inhibit trial


  vec lambda_LS=Penal_LS.col(0);

  vec lambda_RS=Penal_RS.col(0);



  double grad_lambda = - accu(L_lambda(Ind_LCR) % weight_LCR)+ accu(L_lambda(Ind_LIR))+
                         accu(R_lambda(Ind_RIR))- accu(R_lambda(Ind_RCR) % weight_RCR)-
                         ((penal_param(0) - mu_lambda_prime) / sigma_lambda_prime)-

                         accu(L_lambda(Ind_S_LCR) % weight_LCR_G_FS % diff_X_LCR)+
                         accu((L_lambda_S(Ind_S_LCR)) % diff_Y_LCR)-accu(L_lambda_P_S(Ind_S_LCR) % weight_LCR_S % diff_Y_LCR)+

                         accu(L_lambda(Ind_S_LIR) % diff_X_LIR)+
                         accu(L_lambda_P_S(Ind_S_LIR) % diff_Y_LIR)- accu(L_lambda_S(Ind_S_LIR) % weight_LIR_S % diff_Y_LIR)+


                         accu(R_lambda(Ind_S_RIR)%diff_X_RIR)+
                         accu(R_lambda_P_S(Ind_S_RIR)%diff_Y_RIR)- accu(R_lambda_S(Ind_S_RIR) % weight_RIR_S % diff_Y_RIR)-

                         accu(R_lambda(Ind_S_RCR) % weight_RCR_G_FS %diff_X_RCR)+
                         accu(R_lambda_S(Ind_S_RCR) % diff_Y_RCR)- accu(R_lambda_P_S(Ind_S_RCR) % weight_RCR_S % diff_Y_RCR)+

                         accu(lambda_LS)+ accu(lambda_RS);






  //alpha Gradient


  // Go Trial
  vec L_alpha=g_alpha_penal_com(tau(Ind_L),sigma,b_r(Ind_L),nu_r_p,
                                nu_l(Ind_L),nu_l_squared(Ind_L),penal_param(0),penal_param(1));


  vec R_alpha=g_alpha_penal_com(tau(Ind_R),sigma,b_l(Ind_R),nu_l_p,
                                nu_r(Ind_R),nu_r_squared(Ind_R),penal_param(0),penal_param(1));



  // Failed Stop Trial

  vec L_alpha_S=(g_penal_com_s(tau(Ind_L),tau_s(Ind_L),sigma,
                               SSD(Ind_L), b_l(Ind_L),nu_l(Ind_L),nu_l_s(Ind_L),DEL,DEL_s,lt)*nu_stop_penal_alpha);



  vec L_alpha_P_S=g_penal_com_p_s(tau(Ind_L),tau_s(Ind_L),sigma,
                                  SSD(Ind_L),b_r(Ind_L),nu_r_p,nu_r_s(Ind_L),
                                  beta_deriv_alpha_nu_l,nu_stop_penal_alpha,DEL,DEL_s,lt);



  vec R_alpha_S=(g_penal_com_s(tau(Ind_R),tau_s(Ind_R),sigma,
                               SSD(Ind_R), b_r(Ind_R),nu_r(Ind_R),nu_r_s(Ind_R),DEL,DEL_s,lt)*nu_stop_penal_alpha);



  vec R_alpha_P_S=g_penal_com_p_s(tau(Ind_R),tau_s(Ind_R),sigma,
                                  SSD(Ind_R),b_l(Ind_R),nu_l_p,nu_l_s(Ind_R),
                                  beta_deriv_alpha_nu_r,nu_stop_penal_alpha,DEL,DEL_s,lt);



  //Inhibit Trial

  vec alpha_LS=Penal_LS.col(1);

  vec alpha_RS=Penal_RS.col(1);



  double grad_alpha = - accu(L_alpha(Ind_LCR) % weight_LCR)+ accu(L_alpha(Ind_LIR))
                      + accu(R_alpha(Ind_RIR))- accu(R_alpha(Ind_RCR) % weight_RCR)
                      - ((penal_param(1) - mu_alpha_prime) / sigma_alpha_prime)-

                      accu(L_alpha(Ind_S_LCR) % weight_LCR_G_FS % diff_X_LCR)+
                      accu(L_alpha_S(Ind_S_LCR) % diff_Y_LCR)- accu(L_alpha_P_S(Ind_S_LCR) % weight_LCR_S % diff_Y_LCR)+

                      accu(L_alpha(Ind_S_LIR) % diff_X_LIR)+
                      accu(L_alpha_P_S(Ind_S_LIR) % diff_Y_LIR)- accu(L_alpha_S(Ind_S_LIR) % weight_LIR_S % diff_Y_LIR)+

                      accu(R_alpha(Ind_S_RIR) % diff_X_RIR)+
                      accu(R_alpha_P_S(Ind_S_RIR) % diff_Y_RIR)- accu(R_alpha_S(Ind_S_RIR) % weight_RIR_S % diff_Y_RIR)-

                      accu(R_alpha(Ind_S_RCR) % weight_RCR_G_FS % diff_X_RCR )+
                      accu(R_alpha_S(Ind_S_RCR) % diff_Y_RCR )- accu(R_alpha_P_S(Ind_S_RCR) % weight_RCR_S % diff_Y_RCR)+

                      accu(alpha_LS) + accu(alpha_RS);



  vec grad = {grad_lambda,grad_alpha};

  return grad;
}




///////////////////////////////////////////////////////////////////////////////// delta_prime////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Derivative of log dense w.r.t. delta_prime (Go Process)


inline vec deriv_log_dens_delta_prime(const vec &tau,const double sigma,const vec &nu, const vec &b, const double deriv_delta){

  // Rcpp::Rcout<<"AAN"<< endl;

   return (((3/2)*(1/tau)-((exp(2*b))/(2*sigma*pow(tau,2)))+((exp(2*nu))/(2*sigma)))*deriv_delta);





}






// derivative of exponent w.r.t. delta


vec deriv_exp_delta_prime1(const double sigma,const vec &tau,
                           const vec &b,const vec &m,const vec &nu1){

  // Rcpp::Rcout<<"AX"<< endl;


  vec num = (sqrt(tau) % exp(nu1)) + ((exp(b) - m) / (2 * sqrt(tau)));

  vec den=sqrt(sigma)*tau;

  return (num/den);

}



vec deriv_exp_delta_prime2(const double sigma,const vec &tau,
                           const vec &b,const vec &m,const vec &nu1){

  // Rcpp::Rcout<<"AX"<< endl;


  vec num = (sqrt(tau) % exp(nu1)) + ((-exp(b) - m) / (2 * sqrt(tau)));

  vec den=sqrt(sigma)*tau;

  return (num/den);

}




inline vec deriv_nu_diff_delta_prime(double sigma,const vec &tau,const vec &nu1,const vec &SSD_diff){

  return (((sqrt(tau)%exp(nu1))+ ((sigma*SSD_diff)/4))/tau);

}


inline vec deriv_nu_diff_delta_prime(const vec &tau,const vec &SSD,const vec &nu1,const vec &nu2,const double DEL, const double DEL_s){

  return ((SSD+DEL_s-DEL-(2*tau))%(exp(nu2)-exp(nu1)))/(2*pow(tau,3/2));
}



// Derivative of log dense w.r.t. delta_prime (Go Proces-Failed Stop Trial)


mat deriv_log_dens_delta_fs(const double sigma,const vec &tau,const vec &tau_s,
                            const vec &SSD, const vec &b,const vec &nu1,const vec &nu2,const double deriv_delta,
                            const double DEL, const double DEL_s,
                            const bool lt=1){

  // Rcpp::Rcout<<"AS"<< endl;


  // Defining necessary vectors
  vec m=(exp(nu1)%(SSD+DEL_s-DEL))+(exp(nu2)%tau_s);


  vec q1=exponent_1(sigma,tau,m,b);

  vec q2 = exponent_2(sigma,tau,m,b);

  vec rat=phi_rat(q2,lt);

  vec SSD_diff=SSD_Dif(sigma,tau,SSD,nu1,nu2,DEL,DEL_s);


  vec deriv_delta_prime1=deriv_exp_delta_prime1(sigma,tau,b,m,nu1);

  vec deriv_delta_prime2=deriv_exp_delta_prime2(sigma,tau,b,m,nu1);

  vec deriv_nu_diff=deriv_nu_diff_delta_prime(tau,SSD,nu1,nu2,DEL,DEL_s);


  // calculating small vectors

  vec A=(3/2)*(1/tau);

  vec c1=q1%deriv_delta_prime1;

  vec deriv_delta_prime_B=(deriv_multiplier(SSD_diff,rat,q2)%deriv_delta_prime2)+((2*rat%deriv_nu_diff)/(sqrt(sigma)));

  vec B=1+(rat%SSD_diff);

  vec C=(A-c1+(deriv_delta_prime_B/B))*deriv_delta;


  return C;


}




// Cube consisting of derivative of log dense w.r.t. delta_prime for Correct and Incorrect responses (Go Trial)

mat grad_vec(const vec &tau,const double sigma,const vec &nu_CR,const vec &nu_IR,
                 const vec &b_CR,const vec &b_IR,const double deriv_delta){

  //Rcpp::Rcout<<"AAQ"<< endl;


  vec CR= deriv_log_dens_delta_prime(tau,sigma,nu_CR,b_CR,deriv_delta);

  vec IR= deriv_log_dens_delta_prime(tau,sigma,nu_IR,b_IR,deriv_delta);


  mat result=join_rows(CR,IR);

  return result;

}




// Cube consisting of derivative of log dense w.r.t. delta_prime for Correct and Incorrect responses (Go Trial- Stop Process)

mat grad_vec_delta(const double sigma,const vec &tau,const vec &tau_s,const vec &SSD,const vec &b_CR,const vec &nu1_CR,const vec &nu2_CR,
               const vec &b_IR,const vec &nu1_IR,const vec &nu2_IR,const double deriv_delta,const double DEL,const double DEL_s,const bool lt=1){

  //Rcpp::Rcout<<"AAQ"<< endl;


  vec CR= deriv_log_dens_delta_fs(sigma,tau,tau_s,SSD,b_CR,nu1_CR,nu2_CR,deriv_delta,DEL,DEL_s,lt);
  vec IR= deriv_log_dens_delta_fs(sigma,tau,tau_s,SSD,b_IR,nu1_IR,nu2_IR,deriv_delta,DEL,DEL_s,lt);

  mat result=join_rows(CR,IR);

  return result;

}





// Derivative of log dense stop w.r.t. delta_prime_s


vec deriv_log_dens_stop_delta_prime_s(const vec &tau_s, const double sigma, const double &b_stop, const double &nu_stop,
                                      const double deriv_delta) {

  // Rcpp::Rcout<<"AAQ"<< endl;

  vec C(tau_s.n_elem, fill::zeros);

  // Find indices where tau_s > 0
  uvec valid_idx = find(tau_s > 0);

  if (!valid_idx.is_empty()) {
    vec tau_valid = tau_s.elem(valid_idx);
    vec tau_sq = square(tau_valid);

    vec partial_C = ((3 / 2) * (1 / tau_valid) -
      (exp(b_stop) / (2 * sigma * tau_sq)) +
      (exp(2 * nu_stop) / (2 * sigma))) * deriv_delta;

    C.elem(valid_idx) = partial_C;
  }

  return C;
}


// Derivative of log dense w.r.t. delta_prime (Go Proces-Failed Stop Trial)


mat deriv_log_dens_delta_s_fs(const double sigma,const vec &tau,const vec &tau_s,
                            const vec &SSD, const vec &b,const vec &nu1,const vec &nu2,const double deriv_delta_s,
                            const double DEL, const double DEL_s,
                            const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;


  // Defining necessary vectors
  vec m=(exp(nu1)%(SSD+DEL_s-DEL))+(exp(nu2)%tau_s);


  vec q1=exponent_1(sigma,tau,m,b);

  vec q2 = exponent_2(sigma,tau,m,b);

  vec rat=phi_rat(q2,lt);

  vec SSD_diff=SSD_Dif(sigma,tau,SSD,nu1,nu2,DEL,DEL_s);


  // calculating small vectors

  vec deriv_delta_prime_B=deriv_multiplier(SSD_diff,rat,q2)+(2*rat);

  vec B=1+(rat%SSD_diff);

  vec deriv=(exp(nu2)-exp(nu1))/(sqrt(sigma*tau));

  vec C=((-q1+(deriv_delta_prime_B/B))%deriv)*deriv_delta_s;


  return C;


}


// Cube consisting of derivative of log dense w.r.t. delta_prime for Correct and Incorrect responses (Go Trial- Stop Process)

mat grad_vec_delta_s(const double sigma,const vec &tau,const vec &tau_s,const vec &SSD,const vec &b_CR,const vec &nu1_CR,const vec &nu2_CR,
                   const vec &b_IR,const vec &nu1_IR,const vec &nu2_IR,const double deriv_delta_s,const double DEL, const double DEL_s,
                   const bool lt=1){

  //Rcpp::Rcout<<"AAQ"<< endl;


  vec CR= deriv_log_dens_delta_s_fs(sigma,tau,tau_s,SSD,b_CR,nu1_CR,nu2_CR,deriv_delta_s,DEL,DEL_s,lt);
  vec IR= deriv_log_dens_delta_s_fs(sigma,tau,tau_s,SSD,b_IR,nu1_IR,nu2_IR,deriv_delta_s,DEL,DEL_s,lt);

  mat result=join_rows(CR,IR);

  return result;

}



// Gradient function for delta_prime

vec grad_delta(const vec &tau,const vec &tau_s,const double sigma,const vec &SSD,const double DEL,const double DEL_s,
               const vec &delta_param,const vec &stop_param,
               const vec &nu_l, const vec &nu_r,const vec &b_l, const vec &b_r,
               const vec &nu_l_p, const vec &nu_r_p, const vec &nu_l_s, const vec &nu_r_s,

               const field<vec> &delta_prime_Integral,const field<vec> &delta_s_Integral,
               const vec &weight_LIR, const vec &weight_RCR,const vec &weight_LCR, const vec &weight_RIR,
               const vec &weight_LCR_G_FS, const vec &weight_LIR_G_FS,const vec &weight_RCR_G_FS, const vec &weight_RIR_G_FS,
               const vec &weight_LIR_S, const vec &weight_RCR_S,const vec &weight_LCR_S, const vec &weight_RIR_S,
               const vec &weight_FS_LCR,const vec &weight_FS_LIR,const vec &weight_FS_RCR,const vec &weight_FS_RIR,

               const vec &diff_X_LCR,const vec &diff_Y_LCR,const vec &diff_X_LIR,const vec &diff_Y_LIR,
               const vec &diff_X_RCR,const vec &diff_Y_RCR,const vec &diff_X_RIR,const vec &diff_Y_RIR,

               const double deriv_delta,const double deriv_delta_s,const double deriv_delta_s_stop,

               const uvec &Ind_L,const uvec &Ind_R,
               const uvec &Ind_LCR,const uvec &Ind_LIR,const uvec &Ind_RCR,const uvec &Ind_RIR,
               const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,const uvec &Ind_S_RCR,const uvec &Ind_S_RIR,
               const bool lt=1){


  // Rcpp::Rcout<<"AAR"<< endl;


  //delta_prime_s


  mat grad_L_delta_prime_s = grad_vec(tau(Ind_L),sigma,nu_l(Ind_L),nu_r_p,b_l(Ind_L),b_r(Ind_L),deriv_delta_s);
  mat grad_R_delta_prime_s = grad_vec(tau(Ind_R),sigma,nu_r(Ind_R),nu_l_p, b_r(Ind_R),b_l(Ind_R),deriv_delta_s);




  vec LCR_delta_prime_s = grad_L_delta_prime_s.col(0);
  vec LIR_delta_prime_s = grad_L_delta_prime_s.col(1);

  vec RCR_delta_prime_s = grad_R_delta_prime_s.col(0);
  vec RIR_delta_prime_s = grad_R_delta_prime_s.col(1);



  mat grad_LS_delta_prime_s=grad_vec_delta_s(sigma,tau(Ind_L),tau_s(Ind_L),SSD(Ind_L),b_l(Ind_L),nu_l(Ind_L),nu_l_s(Ind_L),
                                       b_r(Ind_L),nu_r_p,nu_r_s(Ind_L),deriv_delta_s,DEL,DEL_s,lt);

  mat grad_RS_delta_prime_s=grad_vec_delta_s(sigma,tau(Ind_R),tau_s(Ind_R),SSD(Ind_R),b_r(Ind_R),nu_r(Ind_R),nu_r_s(Ind_R),
                                       b_l(Ind_R),nu_l_p,nu_l_s(Ind_R),deriv_delta_s,DEL,DEL_s,lt);


  vec LCR_S_delta_prime_s = grad_LS_delta_prime_s.col(0);
  vec LIR_S_delta_prime_s = grad_LS_delta_prime_s.col(1);

  vec RCR_S_delta_prime_s = grad_RS_delta_prime_s.col(0);
  vec RIR_S_delta_prime_s = grad_RS_delta_prime_s.col(1);





  vec deriv_LCR_delta_prime_s=(LCR_S_delta_prime_s(Ind_S_LCR))-(LIR_S_delta_prime_s(Ind_S_LCR)%weight_LCR_S);

  vec deriv_LIR_delta_prime_s=(LIR_S_delta_prime_s(Ind_S_LIR))-(LCR_S_delta_prime_s(Ind_S_LIR)%weight_LIR_S);

  vec deriv_RCR_delta_prime_s=(RCR_S_delta_prime_s(Ind_S_RCR))-(RIR_S_delta_prime_s(Ind_S_RCR)%weight_RCR_S);

  vec deriv_RIR_delta_prime_s=(RIR_S_delta_prime_s(Ind_S_RIR))-(RCR_S_delta_prime_s(Ind_S_RIR)%weight_RIR_S);





  vec deriv_d_s_LCR=deriv_log_dens_stop_delta_prime_s(g(tau_s, Ind_L, Ind_S_LCR ),sigma,stop_param(0),stop_param(1),deriv_delta_s_stop);

  vec deriv_d_s_LIR=deriv_log_dens_stop_delta_prime_s(g(tau_s, Ind_L, Ind_S_LIR ),sigma,stop_param(0),stop_param(1),deriv_delta_s_stop);

  vec deriv_d_s_RCR=deriv_log_dens_stop_delta_prime_s(g(tau_s, Ind_R, Ind_S_RCR ),sigma,stop_param(0),stop_param(1),deriv_delta_s_stop);

  vec deriv_d_s_RIR=deriv_log_dens_stop_delta_prime_s(g(tau_s, Ind_R, Ind_S_RIR ),sigma,stop_param(0),stop_param(1),deriv_delta_s_stop);






  vec FS_LCR= weight_FS_LCR %  diff_Y_LCR;

  vec FS_LIR= weight_FS_LIR %  diff_Y_LIR;

  vec FS_RCR= weight_FS_RCR %  diff_Y_RCR;

  vec FS_RIR= weight_FS_RIR %  diff_Y_RIR;

  double grad_delta_prime_s = accu(LCR_delta_prime_s.elem(Ind_LCR)) - accu(LIR_delta_prime_s.elem(Ind_LCR) % weight_LCR) +
                              accu(LIR_delta_prime_s.elem(Ind_LIR)) - accu(LCR_delta_prime_s.elem(Ind_LIR) % weight_LIR) +
                              accu(RIR_delta_prime_s.elem(Ind_RIR)) - accu(RCR_delta_prime_s.elem(Ind_RIR) % weight_RIR) +
                              accu(RCR_delta_prime_s.elem(Ind_RCR)) - accu(RIR_delta_prime_s.elem(Ind_RCR) % weight_RCR) +


                              accu(LCR_delta_prime_s.elem(Ind_S_LCR) % diff_X_LCR) -
                              accu(LIR_delta_prime_s.elem(Ind_S_LCR) % (weight_LCR_G_FS % diff_X_LCR)) +

                              accu(LIR_delta_prime_s.elem(Ind_S_LIR) % diff_X_LIR) -
                              accu(LCR_delta_prime_s.elem(Ind_S_LIR) % (weight_LIR_G_FS % diff_X_LIR)) +

                              accu(RIR_delta_prime_s.elem(Ind_S_RIR) % diff_X_RIR) -
                              accu(RCR_delta_prime_s.elem(Ind_S_RIR) % (weight_RIR_G_FS % diff_X_RIR)) +

                              accu(RCR_delta_prime_s.elem(Ind_S_RCR) % diff_X_RCR) -
                              accu(RIR_delta_prime_s.elem(Ind_S_RCR) % (weight_RCR_G_FS % diff_X_RCR)) +

                              dot(diff_Y_LCR, deriv_LCR_delta_prime_s) +
                              dot(diff_Y_LIR, deriv_LIR_delta_prime_s) +
                              dot(diff_Y_RCR, deriv_RCR_delta_prime_s) +
                              dot(diff_Y_RIR, deriv_RIR_delta_prime_s) -

                              (dot(FS_LCR, deriv_d_s_LCR) +
                              dot(FS_LIR, deriv_d_s_LIR) +
                              dot(FS_RCR, deriv_d_s_RCR) +
                              dot(FS_RIR, deriv_d_s_RIR)) +

                              sum(delta_s_Integral(0)) + sum(delta_s_Integral(1)) +
                              1- (2*EXPITE(delta_param(0)));






  //delta_prime


  mat grad_L_delta_prime = grad_vec(tau(Ind_L),sigma,nu_l(Ind_L),nu_r_p,b_l(Ind_L),b_r(Ind_L),deriv_delta);
  mat grad_R_delta_prime = grad_vec(tau(Ind_R),sigma,nu_r(Ind_R),nu_l_p, b_r(Ind_R),b_l(Ind_R),deriv_delta);


  mat LCR_delta_prime = grad_L_delta_prime.col(0);
  mat LIR_delta_prime = grad_L_delta_prime.col(1);

  mat RCR_delta_prime = grad_R_delta_prime.col(0);
  mat RIR_delta_prime = grad_R_delta_prime.col(1);




  mat grad_LS_delta_prime=grad_vec_delta(sigma,tau(Ind_L),tau_s(Ind_L),SSD(Ind_L),b_l(Ind_L),nu_l(Ind_L),nu_l_s(Ind_L),
                                     b_r(Ind_L),nu_r_p,nu_r_s(Ind_L),deriv_delta,DEL,DEL_s,lt);

  mat grad_RS_delta_prime=grad_vec_delta(sigma,tau(Ind_R),tau_s(Ind_R),SSD(Ind_R),b_r(Ind_R),nu_r(Ind_R),nu_r_s(Ind_R),
                                     b_l(Ind_R),nu_l_p,nu_l_s(Ind_R),deriv_delta,DEL,DEL_s,lt);



  vec LCR_S_delta_prime = grad_LS_delta_prime.col(0);
  vec LIR_S_delta_prime = grad_LS_delta_prime.col(1);

  vec RCR_S_delta_prime = grad_RS_delta_prime.col(0);
  vec RIR_S_delta_prime = grad_RS_delta_prime.col(1);





  vec deriv_LCR_delta_prime=(LCR_S_delta_prime(Ind_S_LCR))-(LIR_S_delta_prime(Ind_S_LCR)%weight_LCR_S);

  vec deriv_LIR_delta_prime=(LIR_S_delta_prime(Ind_S_LIR))-(LCR_S_delta_prime(Ind_S_LIR)%weight_LIR_S);

  vec deriv_RCR_delta_prime=(RCR_S_delta_prime(Ind_S_RCR))-(RIR_S_delta_prime(Ind_S_RCR)%weight_RCR_S);

  vec deriv_RIR_delta_prime=(RIR_S_delta_prime(Ind_S_RIR))-(RCR_S_delta_prime(Ind_S_RIR)%weight_RIR_S);


  double grad_delta_prime =accu(LCR_delta_prime.elem(Ind_LCR)) - accu(LIR_delta_prime.elem(Ind_LCR) % weight_LCR) +
                           accu(LIR_delta_prime.elem(Ind_LIR)) - accu(LCR_delta_prime.elem(Ind_LIR) % weight_LIR) +
                           accu(RIR_delta_prime.elem(Ind_RIR)) - accu(RCR_delta_prime.elem(Ind_RIR) % weight_RIR) +
                           accu(RCR_delta_prime.elem(Ind_RCR)) - accu(RIR_delta_prime.elem(Ind_RCR) % weight_RCR) +

                            accu(LCR_delta_prime.elem(Ind_S_LCR) % diff_X_LCR) -
                            accu(LIR_delta_prime.elem(Ind_S_LCR) % (weight_LCR_G_FS % diff_X_LCR)) +

                            accu(LIR_delta_prime.elem(Ind_S_LIR) % diff_X_LIR) -
                            accu(LCR_delta_prime.elem(Ind_S_LIR) % (weight_LIR_G_FS % diff_X_LIR)) +

                            accu(RIR_delta_prime.elem(Ind_S_RIR) % diff_X_RIR) -
                            accu(RCR_delta_prime.elem(Ind_S_RIR) % (weight_RIR_G_FS % diff_X_RIR)) +

                            accu(RCR_delta_prime.elem(Ind_S_RCR) % diff_X_RCR) -
                            accu(RIR_delta_prime.elem(Ind_S_RCR) % (weight_RCR_G_FS % diff_X_RCR)) +

                            dot(diff_Y_LCR, deriv_LCR_delta_prime) +
                            dot(diff_Y_LIR, deriv_LIR_delta_prime) +
                            dot(diff_Y_RCR, deriv_RCR_delta_prime) +
                            dot(diff_Y_RIR, deriv_RIR_delta_prime) +

                            sum(delta_prime_Integral(0)) + sum(delta_prime_Integral(1)) +
                            1- (2*EXPITE(delta_param(1)));


  vec grad = {grad_delta_prime_s,grad_delta_prime};


  return grad;


}






///////////////////////////////////////////////////////////////////////////////// Stop Parameters////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Derivative of log dense w.r.t. b_stop


vec deriv_log_dens_b_stop(const vec &tau_s,const double sigma,const double &b_stop,const double &nu_stop){

  // Rcpp::Rcout<<"AAS"<< endl;

  vec C = 1 - (((exp(2 * b_stop)) - (exp(b_stop + nu_stop) * tau_s)) / (sigma * tau_s));

  return C;

}



// Derivative of log dense w.r.t. nu_stop (Stop Trial)


vec deriv_log_dens_nu_stop(const vec &tau_s,const double sigma,const double &b_stop,const double &nu_stop){

  // Rcpp::Rcout<<"AAU"<< endl;

  vec C= ((exp(b_stop+nu_stop))-((exp(2*nu_stop))*tau_s))/(sigma);

  return C; //length(C)

}



// Derivative of nu_stop penalty w.r.t nu_stop (Go Process-Failed Stop Trial)


double g_nu_stop_stim_penal(const double &nu_stop, double lambda_prime,double alpha_prime){

  double X=(exp(2*(lambda_prime+nu_stop)));
  double Y=exp(2*alpha_prime);
  double hpot=sqrt(X+Y);

  double numerator = (exp(lambda_prime+nu_stop))-((exp(2*(lambda_prime+nu_stop)))/hpot);

  double denominator=pow((-(exp(lambda_prime+nu_stop))+hpot),2);

  return numerator/denominator;


}


// derivative of exponent w.r.t. beta (without penalty)

inline vec deriv_exp_nu_stop(double sigma,const vec &tau,const vec &SSD,const vec &nu2,
                      const double &nu_stop, double lambda_prime,double alpha_prime){

  return (exp(nu2)*g_nu_stop_stim_penal(nu_stop,lambda_prime,alpha_prime));


}




// beta derivative (without penalty) (Go Proces-Failed Stop Trial)


vec deriv_log_dens_nu_stop_s(const vec &tau,const vec &tau_s,const double sigma,
                             const vec &SSD, const vec &b,const vec &nu1,const vec &nu2,
                             const double &nu_stop, double lambda_prime,double alpha_prime,
                             const double DEL, const double DEL_s,const bool lt=1){

  // Rcpp::Rcout<<"AW"<< endl;

  vec m=(exp(nu1)%(SSD+DEL_s-DEL))+(exp(nu2)%tau_s);


  vec q1=exponent_1(sigma,tau,m,b);

  vec q2 = exponent_2(sigma,tau,m,b);

  vec rat=phi_rat(q2,lt);

  vec SSD_diff=SSD_Dif(sigma,tau,SSD,nu1,nu2,DEL,DEL_s);


  vec deriv_nu_stop=deriv_exp_nu_stop(sigma,tau,SSD,nu2,nu_stop,lambda_prime,alpha_prime);


  vec w=tau_s/(sqrt(sigma*tau));

  vec c1=q1%w;

  vec deriv_nu_stop_B=(deriv_multiplier(SSD_diff,rat,q2)%w)-((2*rat%(SSD+DEL_s-DEL))/(sqrt(sigma*tau)));

  vec B=1+(rat%SSD_diff);

  vec C=(-c1+(deriv_nu_stop_B/B))%deriv_nu_stop;


  return C;


}







// Cube consisting of derivative of log dense w.r.t. nu_stop for Correct and Incorrect responses (Go Trial- Stop Process)


mat grad_vec_nu_stop(const vec &tau,const vec &tau_s,const double sigma,const vec &SSD,
                     const vec &b_CR,const vec &nu_1_CR,const vec &nu_2_CR,
                     const vec &b_IR,const vec &nu_1_IR,const vec &nu_2_IR,
                     double nu_stop,double lambda_prime,double alpha_prime,
                     const double DEL, const double DEL_s,const bool lt=1){

  // Rcpp::Rcout<<"AAX"<< endl;


  vec CR= deriv_log_dens_nu_stop_s(tau,tau_s,sigma,SSD, b_CR,nu_1_CR,nu_2_CR,nu_stop, lambda_prime,alpha_prime,DEL,DEL_s,lt);
  vec IR= deriv_log_dens_nu_stop_s(tau,tau_s,sigma,SSD, b_IR,nu_1_IR,nu_2_IR,nu_stop, lambda_prime,alpha_prime,DEL,DEL_s,lt);

  mat result=join_rows(CR,IR);

  return result;

}




// Gradient function for Stop Parameters

vec grad_stop_param(const vec &tau,const vec &tau_s,const double sigma,const vec &SSD,const double DEL, const double DEL_s,
                    const vec &nu_l, const vec &nu_r,const vec &b_l, const vec &b_r,
                    const vec &nu_l_p, const vec &nu_r_p, const vec &nu_l_s, const vec &nu_r_s,
                    const vec &weight_LCR_S, const vec &weight_LIR_S,const vec &weight_RCR_S, const vec &weight_RIR_S,
                    const vec &weight_FS_LCR,const vec &weight_FS_LIR,const vec &weight_FS_RCR,const vec &weight_FS_RIR,
                    const uvec &Ind_L,const uvec &Ind_R,
                    const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,const uvec &Ind_S_RCR,const uvec &Ind_S_RIR,

                    const field<vec> &nu_stop_Stim, const field<vec> &b_stop_Stim,

                    const vec &stop_param,const vec &penal_param,
                    const double mu_nu_stop,const double sigma_nu_stop,const double mu_b_stop,const double sigma_b_stop,
                    const vec &diff_Y_LCR,const vec &diff_Y_LIR,const vec &diff_Y_RCR,const vec &diff_Y_RIR,
                    const bool lt=1  ){




  // Rcpp::Rcout<<"AAY"<< endl;

  //b_stop gradient

  vec deriv_b_s_LCR=deriv_log_dens_b_stop(g(tau_s, Ind_L, Ind_S_LCR ),sigma,stop_param(0),stop_param(1));

  vec deriv_b_s_LIR=deriv_log_dens_b_stop(g(tau_s, Ind_L, Ind_S_LIR ),sigma,stop_param(0),stop_param(1));

  vec deriv_b_s_RCR=deriv_log_dens_b_stop(g(tau_s, Ind_R, Ind_S_RCR ),sigma,stop_param(0),stop_param(1));

  vec deriv_b_s_RIR=deriv_log_dens_b_stop(g(tau_s, Ind_R, Ind_S_RIR ),sigma,stop_param(0),stop_param(1));




  vec FS_LCR= weight_FS_LCR %  diff_Y_LCR;

  vec FS_LIR= weight_FS_LIR %  diff_Y_LIR;

  vec FS_RCR= weight_FS_RCR %  diff_Y_RCR;

  vec FS_RIR= weight_FS_RIR %  diff_Y_RIR;



  double dot_LCR = dot(FS_LCR, deriv_b_s_LCR);
  double dot_LIR = dot(FS_LIR, deriv_b_s_LIR);
  double dot_RCR = dot(FS_RCR, deriv_b_s_RCR);
  double dot_RIR = dot(FS_RIR, deriv_b_s_RIR);

  double prior_term = (stop_param(0) - mu_b_stop) / sigma_b_stop;
  double stim_sum = sum(b_stop_Stim(0)) + sum(b_stop_Stim(1));

  // Compute the gradient
  double grad_b_stop = -(dot_LCR + dot_LIR + dot_RCR + dot_RIR)
    - prior_term
  + stim_sum;




  //nu_stop gradient

  mat grad_L_S=grad_vec_nu_stop(tau(Ind_L),tau_s(Ind_L),sigma,SSD(Ind_L),
                                b_l(Ind_L),nu_l(Ind_L),nu_l_s(Ind_L),
                                b_r(Ind_L),nu_r_p,nu_r_s(Ind_L),
                                stop_param(1),penal_param(0),penal_param(1),DEL,DEL_s,lt);


  mat grad_R_S=  grad_vec_nu_stop(tau(Ind_R),tau_s(Ind_R),sigma,SSD(Ind_R),
                                  b_r(Ind_R),nu_r(Ind_R),nu_r_s(Ind_R),
                                  b_l(Ind_R),nu_l_p,nu_l_s(Ind_R),
                                  stop_param(1),penal_param(0),penal_param(1),DEL,DEL_s,lt);


  vec LCR_S = grad_L_S.col(0);
  vec LIR_S = grad_L_S.col(1);

  vec RCR_S = grad_R_S.col(0);
  vec RIR_S = grad_R_S.col(1);



  vec deriv_nu_s_LCR=deriv_log_dens_nu_stop(g(tau_s, Ind_L, Ind_S_LCR ),sigma,stop_param(0),stop_param(1));

  vec deriv_nu_s_LIR=deriv_log_dens_nu_stop(g(tau_s, Ind_L, Ind_S_LIR ),sigma,stop_param(0),stop_param(1));

  vec deriv_nu_s_RCR=deriv_log_dens_nu_stop(g(tau_s, Ind_R, Ind_S_RCR ),sigma,stop_param(0),stop_param(1));

  vec deriv_nu_s_RIR=deriv_log_dens_nu_stop(g(tau_s, Ind_R, Ind_S_RIR ),sigma,stop_param(0),stop_param(1));


  vec deriv_LCR=(LCR_S(Ind_S_LCR))-(LIR_S(Ind_S_LCR)%weight_LCR_S);

  vec deriv_LIR=(LIR_S(Ind_S_LIR))-(LCR_S(Ind_S_LIR)%weight_LIR_S);

  vec deriv_RCR=(RCR_S(Ind_S_RCR))-(RIR_S(Ind_S_RCR)%weight_RCR_S);

  vec deriv_RIR=(RIR_S(Ind_S_RIR))-(RCR_S(Ind_S_RIR)%weight_RIR_S);




  double grad_nu_stop=dot(diff_Y_LCR,deriv_LCR)-dot(FS_LCR,deriv_nu_s_LCR)+
                      dot(diff_Y_LIR,deriv_LIR)-dot(FS_LIR,deriv_nu_s_LIR)+
                      dot(diff_Y_RCR,deriv_RCR)-dot(FS_RCR,deriv_nu_s_RCR)+
                      dot(diff_Y_RIR,deriv_RIR)-dot(FS_RIR,deriv_nu_s_RIR)-

                      ((stop_param(1)-mu_nu_stop)/sigma_nu_stop)+
                      sum(nu_stop_Stim(0))+sum(nu_stop_Stim(1));


  vec grad = {grad_b_stop,grad_nu_stop};



  return grad;

}









// Gradient function for Probability Parameters

vec grad_prob_param(const vec &tau_s,const double sigma,
                    const field<vec> &Int_dens,
                    const vec &stop_param,const vec &prob_param,
                    const double sum_Prob,
                    const vec &a,

                    const double T_Go,const double k,const double T_total,
                    const vec &diff_X_LCR,const vec &diff_Y_LCR,const vec &diff_X_LIR,const vec &diff_Y_LIR,
                    const vec &diff_X_RCR,const vec &diff_Y_RCR,const vec &diff_X_RIR,const vec &diff_Y_RIR){


  // Rcpp::Rcout<<"grad_prob_param"<<endl;

  double w_01=prob_param(0);
  double w_00=prob_param(1);
  double w_10=prob_param(2);

  vec V_I_L=(w_00+Int_dens(0));
  vec V_I_R=(w_00+Int_dens(1));

  vec V_L=compute_log_add2(w_01,V_I_L);
  vec V_R=compute_log_add2(w_01,V_I_R);

  double u=mlpack::LogAdd(w_00,w_10);

  vec w_u=(T_Go-k)*exp(prob_param-u);
  vec w_logS=T_total*exp(prob_param-sum_Prob);

  vec prior=a-exp(prob_param);



  //GF Gradient

  double grad_w_01= k+sum(exp(w_01-V_L))+sum(exp(w_01-V_R));


  double grad_w_00=w_u(1),grad_w_10=w_u(2);

  if(diff_Y_LCR.n_elem){
    grad_w_00+=exp(mlpack::AccuLog(diff_Y_LCR));
    grad_w_10+=exp(mlpack::AccuLog(diff_X_LCR));  /// Since length(diff_Y_LCR)=length(diff_X_LCR)
  }

  if(diff_Y_LIR.n_elem){
    grad_w_00+=exp(mlpack::AccuLog(diff_Y_LIR));
    grad_w_10+=exp(mlpack::AccuLog(diff_X_LIR));
  }

  if(diff_Y_RCR.n_elem){
    grad_w_00+=exp(mlpack::AccuLog(diff_Y_RCR));
    grad_w_10+=exp(mlpack::AccuLog(diff_X_RCR));
  }

  if(diff_Y_RIR.n_elem){
    grad_w_00+=exp(mlpack::AccuLog(diff_Y_RIR));
    grad_w_10+=exp(mlpack::AccuLog(diff_X_RIR));
  }


  if(V_I_L.n_elem){
    grad_w_00+=exp(mlpack::AccuLog((V_I_L-V_L)));
  }

  if(V_I_R.n_elem){
    grad_w_00+=exp(mlpack::AccuLog((V_I_R-V_R)));
  }






  vec grad=-w_logS+prior;

  grad(0) += grad_w_01;

  grad(1) += grad_w_00;

  grad(2) += grad_w_10;




  return grad;

}


///////////////////////////////////////////////////////////////////////////////// rand_param //////////////////////////////////////////////////////////////////////////////////////////////


// Derivative of S w.r.t. l

inline vec grad_lam( const double nu,const double l,const vec &lam, const double D=1){
  return (nu*((D*exp(-2*l))-(4*pow(M_PI,2)*lam)))/((nu*exp(-2*l))+(2*pow(M_PI,2)*lam));
}


// Gradient function for Random_effect parameters

vec grad_rand(const unsigned N,const mat &gama_sq,const vec &rand_param,
              const vec &S,const vec &grad_S){


  // Rcpp::Rcout<<"grad_rand"<< endl;


  unsigned m=gama_sq.n_rows;


  //alpha gradient

  // double gama_S=dot(gama_sq,exp(-S));

  mat gama_S=gama_sq.each_col() % exp(-S);

  double D_gl=(exp(-rand_param(0))*accu(gama_S))/2;

  double alph_grad=D_gl-((m*N)/2);


  ///l gradient

  mat gama_grad_S=gama_sq.each_col()%(exp(-S)%grad_S);

  double prod_grad_l=exp(-rand_param(0))*accu(gama_grad_S);

  double l_grad=(prod_grad_l/2)-((N*accu(grad_S))/2);


  vec result = {alph_grad,l_grad};

  return result;



}


// grad_rand_prior

vec grad_rand_prior(const vec &rand_param, const double kappa) {


  double alph_grad=-(0.5 * tanh(rand_param(0)/2));

  double l_grad=-(1 - (kappa / exp(rand_param(1))));

  vec grad = {alph_grad,l_grad};

  return grad;
}




///////////////////////////////////////////////////////////////////////////////// Leapfrog and Parameter Update ///////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////// Main Effect /////////////////////////////////////////////////////////////////////////////////////////////

//Leapfrog algorithm

void leap_frog_main_effect(const field <mat> &t,
                           const field <vec> &Rand_gama_l,const field <vec> &Rand_gama_r,
                           const field <vec> &Rand_beta_l,const field <vec> &Rand_beta_r,
                           const field<vec> &tau,const field<vec> &tau_s, const double sigma,
                           const field <vec> &SSD,const vec &DEL,const vec &DEL_s,
                           mat &gama, mat &beta,const mat &delta_prime, const mat &penal_param,
                           const mat &stop_param,const mat &prob_param,
                           const field <uvec> &Ind_L, const field <uvec> &Ind_R,
                           const field <uvec> &Ind_LCR, const field <uvec> &Ind_LIR,
                           const field <uvec> &Ind_RCR, const field <uvec> &Ind_RIR,
                           const field <uvec> &Ind_S_LCR,const field <uvec> &Ind_S_LIR,
                           const field <uvec> &Ind_S_RCR,const field <uvec> &Ind_S_RIR,
                           const field <uvec>&Ind_I_L,const field <uvec>&Ind_I_R,

                           const field <mat> &gama_I,const field <mat> &beta_I,


                           double eta_b_l,double eta_b_r,double eta_nu_l,double eta_nu_r,
                           const vec &mu_gl,const mat &sigma_gl_inv,const vec &mu_gr,const mat &sigma_gr_inv,
                           const vec &mu_bl,const mat &sigma_bl_inv,const vec &mu_br,const mat &sigma_br_inv,

                           mat &v_old_g,mat &v_old_b, const double L, const double delta, mat &p_g,mat &p_b,
                           unsigned nparall,const field <vec> &lower_bound,const double upper_bound,const bool lt =1) {


  // Rcpp::Rcout << "AAZ" << std::endl;

  for (unsigned j = 0; j < L; ++j) {

    gama += delta * (p_g - (delta / 2) * v_old_g);


    gama.clamp(-5,100);

    beta += delta * (p_b - (delta/2)   * v_old_b);


    beta.clamp(-5,3);


    field <field<vec>> gama_ess = update_gama_param_ess1(t,Rand_gama_l,Rand_gama_r,gama);

    field <vec> b_l = gama_ess(0);
    field <vec> b_r = gama_ess(1);


    field <field<vec>> beta_ess=update_beta_param_ess1(t,Rand_beta_l,Rand_beta_r,beta);

    field <vec> nu_l=beta_ess(0);
    field <vec> nu_r=beta_ess(1);
    field <vec> nu_l_squared=beta_ess(2);
    field <vec> nu_r_squared=beta_ess(3);


    field<field<vec>> beta_ess2=update_beta_param_ess2(tau,sigma,b_l,b_r,nu_l,nu_r,nu_l_squared,nu_r_squared,
                                                       Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                                                       Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                                       penal_param);



    field <vec> nu_l_p=beta_ess2(0);
    field <vec> nu_r_p=beta_ess2(1);

    field <vec> weight_LCR=beta_ess2(2);
    field <vec> weight_LIR=beta_ess2(3);
    field <vec> weight_RCR=beta_ess2(4);
    field <vec> weight_RIR=beta_ess2(5);

    field <vec> weight_LCR_G_FS=beta_ess2(6);
    field <vec> weight_LIR_G_FS=beta_ess2(7);
    field <vec> weight_RCR_G_FS=beta_ess2(8);
    field <vec> weight_RIR_G_FS=beta_ess2(9);


    field <field<vec>> stop_param_ess= update_stop_param_ess(tau,tau_s,sigma,SSD,stop_param,b_l,b_r,nu_l,nu_r,nu_r_p,nu_l_p,
                                                             Ind_L,Ind_R,Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                                             penal_param,DEL,DEL_s,lt);




    field <vec> nu_l_s=stop_param_ess(0);
    field <vec> nu_r_s=stop_param_ess(1);
    field <vec> weight_LCR_S=stop_param_ess(2);
    field <vec> weight_LIR_S=stop_param_ess(3);
    field <vec> weight_RCR_S=stop_param_ess(4);
    field <vec> weight_RIR_S=stop_param_ess(5);



    field<field<vec>>Int_dens=update_likelihood(sigma,SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                                b_l,b_r,nu_l,nu_r,nu_l_s,nu_r_s,
                                                nu_r_p,nu_l_p,Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                lower_bound,upper_bound,nparall,lt);




    field <field<field<vec>>>gama_grad_integral=update_gama_grad(sigma,Int_dens,
                                                                 SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                                                 b_l,b_r,nu_l,nu_r,
                                                                 nu_l_s,nu_r_s,nu_l_p,nu_r_p,Ind_L,Ind_R,
                                                                 Ind_I_L,Ind_I_R,
                                                                 lower_bound,upper_bound,
                                                                 nparall,lt);


    field<field<vec>>gama_LS=gama_grad_integral(0);
    field<field<vec>>gama_RS=gama_grad_integral(1);




    field <field<field<vec>>>beta_grad_integral=update_beta_grad(sigma,Int_dens,
                                                                 SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                                                 b_l,b_r,nu_l,nu_r,
                                                                 nu_l_s,nu_r_s,nu_l_p,nu_r_p,
                                                                 nu_l_squared,nu_r_squared,
                                                                 Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                                 lower_bound,upper_bound,
                                                                 nparall,lt);


    field<field<vec>> beta_l_Stim=beta_grad_integral(0);
    field<field<vec>> beta_r_Stim=beta_grad_integral(1);



    field <field<vec>> lk_FS2=update_lk_FS2(tau,tau_s,sigma,SSD,b_l,b_r,
                                            nu_l,nu_r,nu_r_p,nu_l_p,nu_l_s,nu_r_s,
                                            Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                                            Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                            stop_param,prob_param,DEL,DEL_s,lt);



    field <vec> diff_X_LCR=lk_FS2(4);
    field <vec> diff_Y_LCR=lk_FS2(5);

    field <vec> diff_X_LIR=lk_FS2(6);
    field <vec> diff_Y_LIR=lk_FS2(7);

    field <vec> diff_X_RCR=lk_FS2(8);
    field <vec> diff_Y_RCR=lk_FS2(9);

    field <vec> diff_X_RIR=lk_FS2(10);
    field <vec> diff_Y_RIR=lk_FS2(11);




    unsigned N=tau.n_elem;

    // New gradient

    //Gama

    mat v_new_g(size(gama), fill::zeros);
    mat v_new_b(size(beta), fill::zeros);

    omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall)
    for (unsigned i = 0; i < N; ++i) {
      mat grad_gama_result = grad_gama(t(i), tau(i),tau_s(i), sigma, SSD(i),DEL(i),DEL_s(i),
                                       nu_l(i), nu_r(i), b_l(i), b_r(i),
                                       nu_l_p(i), nu_r_p(i), nu_l_s(i), nu_r_s(i),

                                       weight_LCR(i),weight_LIR(i),weight_RCR(i),weight_RIR(i),
                                       weight_LCR_G_FS(i),weight_LIR_G_FS(i),weight_RCR_G_FS(i),weight_RIR_G_FS(i),
                                       weight_LCR_S(i),weight_LIR_S(i),weight_RCR_S(i),weight_RIR_S(i),

                                       Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RCR(i), Ind_RIR(i),
                                       Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RCR(i), Ind_S_RIR(i),
                                       Ind_I_L(i), Ind_I_R(i),

                                       diff_X_LCR(i),diff_Y_LCR(i),diff_X_LIR(i),diff_Y_LIR(i),
                                       diff_X_RCR(i),diff_Y_RCR(i),diff_X_RIR(i),diff_Y_RIR(i),

                                       gama_LS(i), gama_RS(i));


      mat grad_beta_result = grad_beta(t(i),tau(i),tau_s(i),sigma,SSD(i),DEL(i),DEL_s(i),
                                       nu_l(i),nu_l_p(i),nu_r(i),nu_r_p(i),nu_l_s(i),nu_r_s(i),
                                       b_l(i),b_r(i),nu_l_squared(i),nu_r_squared(i),
                                       penal_param.col(i),Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RIR(i),Ind_RCR(i),
                                       Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RIR(i),Ind_S_RCR(i),
                                       Ind_I_L(i),Ind_I_R(i),
                                       beta_l_Stim(i),beta_r_Stim(i),
                                       weight_LCR(i),weight_LIR(i),weight_RCR(i),weight_RIR(i),
                                       weight_LCR_G_FS(i),weight_LIR_G_FS(i),weight_RCR_G_FS(i),weight_RIR_G_FS(i),
                                       weight_LCR_S(i), weight_LIR_S(i),weight_RCR_S(i), weight_RIR_S(i),
                                       diff_X_LCR(i),diff_Y_LCR(i),diff_X_LIR(i),diff_Y_LIR(i),
                                       diff_X_RCR(i),diff_Y_RCR(i),diff_X_RIR(i),diff_Y_RIR(i),lt);


#pragma omp critical
{
  v_new_g -= grad_gama_result;

  v_new_b -= grad_beta_result;

}

    }


    omp_set_num_threads(6);



    mat gama_prior=grad_gama_prior(gama,eta_b_l,eta_b_r,mu_gl,sigma_gl_inv,mu_gr,sigma_gr_inv);

    v_new_g -=gama_prior;

    // Momentum
    p_g -= (delta / 2) * (v_old_g + v_new_g);
    v_old_g = v_new_g;


    // Beta

    mat beta_prior=grad_beta_prior(beta,eta_nu_l,eta_nu_r,mu_bl,sigma_bl_inv,mu_br,sigma_br_inv);
    v_new_b -=beta_prior;


    //momentum
    p_b-=(delta/2)*(v_old_b+v_new_b);
    v_old_b=v_new_b;







  }
}




//Update Main_effect

void update_main_effect(const field <mat> &t,
                        const field <vec> &Rand_gama_l,const field <vec> &Rand_gama_r,
                        const field <vec> &Rand_beta_l,const field <vec> &Rand_beta_r,
                        const field <vec> &tau,const field<vec> &tau_s,const double sigma,
                        const field <vec> &SSD,const vec &DEL,const vec &DEL_s,const field <vec> &Ind_GF,
                        mat &gama,mat &beta,const mat &delta_param,
                        const mat &penal_param,const mat &stop_param,const mat &prob_param,
                        const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                        const field <uvec> &Ind_LCR, const field <uvec> &Ind_LIR,
                        const  field <uvec> &Ind_RCR,const field <uvec> &Ind_RIR,
                        const field <uvec> &Ind_S_LCR,const field <uvec> &Ind_S_LIR,
                        const field <uvec> &Ind_S_RCR,const field <uvec> &Ind_S_RIR,

                        const field <uvec>&Ind_I_L,const field <uvec>&Ind_I_R,

                        double eta_b_l,double eta_b_r, double eta_nu_l,double eta_nu_r,

                        const vec &mu_gl,const mat &sigma_gl_inv,const vec &mu_gr,const mat &sigma_gr_inv,
                        const vec &mu_bl,const mat &sigma_bl_inv,const vec &mu_br,const mat &sigma_br_inv,


                        const field <mat>  &gama_I,const field <mat>  &beta_I,
                        const vec &T_Go,const vec &k,

                        const vec &range,const double L,const int leapmax,unsigned &acceptance_main_effect,
                        const unsigned nparall,
                        const field <vec> &lower_bound,const double upper_bound,const bool lt =1){

  // Rcpp::Rcout<<"update_main_effect"<< endl;


  mat p_g(size(gama),fill::randn);
  mat p_b(size(beta),fill::randn);


  field <field<vec>> gama_ess = update_gama_param_ess1(t,Rand_gama_l,Rand_gama_r,gama);

  field <vec> b_l = gama_ess(0);
  field <vec> b_r = gama_ess(1);


  field <field<vec>> beta_ess=update_beta_param_ess1(t,Rand_beta_l,Rand_beta_r,beta);

  field <vec> nu_l=beta_ess(0);
  field <vec> nu_r=beta_ess(1);
  field <vec> nu_l_squared=beta_ess(2);
  field <vec> nu_r_squared=beta_ess(3);



  field<field<vec>> beta_ess2=update_beta_param_ess2(tau,sigma,b_l,b_r,nu_l,nu_r,nu_l_squared,nu_r_squared,
                                                     Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                                                     Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                                     penal_param);



  field <vec> nu_l_p=beta_ess2(0);
  field <vec> nu_r_p=beta_ess2(1);

  field <vec> weight_LCR=beta_ess2(2);
  field <vec> weight_LIR=beta_ess2(3);
  field <vec> weight_RCR=beta_ess2(4);
  field <vec> weight_RIR=beta_ess2(5);

  field <vec> weight_LCR_G_FS=beta_ess2(6);
  field <vec> weight_LIR_G_FS=beta_ess2(7);
  field <vec> weight_RCR_G_FS=beta_ess2(8);
  field <vec> weight_RIR_G_FS=beta_ess2(9);


  field <field<vec>> stop_param_ess= update_stop_param_ess(tau,tau_s,sigma,SSD,stop_param,b_l,b_r,nu_l,nu_r,nu_r_p,nu_l_p,
                                                           Ind_L,Ind_R,Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                                           penal_param,DEL,DEL_s,lt);




  field <vec> nu_l_s=stop_param_ess(0);
  field <vec> nu_r_s=stop_param_ess(1);
  field <vec> weight_LCR_S=stop_param_ess(2);
  field <vec> weight_LIR_S=stop_param_ess(3);
  field <vec> weight_RCR_S=stop_param_ess(4);
  field <vec> weight_RIR_S=stop_param_ess(5);



  field <field<vec>> lk_FS2=update_lk_FS2(tau,tau_s,sigma,SSD,b_l,b_r,
                                          nu_l,nu_r,nu_r_p,nu_l_p,nu_l_s,nu_r_s,
                                          Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                                          Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                          stop_param,prob_param,DEL,DEL_s,lt);



  field <vec> lk_LCR_FS=lk_FS2(0);
  field <vec> lk_LIR_FS=lk_FS2(1);

  field <vec> lk_RCR_FS=lk_FS2(2);
  field <vec> lk_RIR_FS=lk_FS2(3);


  field <vec> diff_X_LCR=lk_FS2(4);
  field <vec> diff_Y_LCR=lk_FS2(5);

  field <vec> diff_X_LIR=lk_FS2(6);
  field <vec> diff_Y_LIR=lk_FS2(7);

  field <vec> diff_X_RCR=lk_FS2(8);
  field <vec> diff_Y_RCR=lk_FS2(9);

  field <vec> diff_X_RIR=lk_FS2(10);
  field <vec> diff_Y_RIR=lk_FS2(11);


  field<field<vec>>Int_dens=update_likelihood(sigma,SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                              b_l,b_r,nu_l,nu_r,nu_l_s,nu_r_s,
                                              nu_r_p,nu_l_p,Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                              lower_bound,upper_bound,nparall,lt);



  field <vec>intgral_sum=update_integral_likelihood_sum(Int_dens);

  vec Inhib_likelihood=intgral_sum(0);



  field <field<field<vec>>>gama_grad_integral=update_gama_grad(sigma,Int_dens,
                                                               SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                                               b_l,b_r,nu_l,nu_r,
                                                               nu_l_s,nu_r_s,nu_l_p,nu_r_p,Ind_L,Ind_R,
                                                               Ind_I_L,Ind_I_R,
                                                               lower_bound,upper_bound,
                                                               nparall,lt);


  field<field<vec>>gama_LS=gama_grad_integral(0);
  field<field<vec>>gama_RS=gama_grad_integral(1);



  field <field<field<vec>>>beta_grad_integral=update_beta_grad(sigma,Int_dens,
                                                               SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                                               b_l,b_r,nu_l,nu_r,
                                                               nu_l_s,nu_r_s,nu_l_p,nu_r_p,
                                                               nu_l_squared,nu_r_squared,
                                                               Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                               lower_bound,upper_bound,
                                                               nparall,lt);




  field<field<vec>> beta_l_Stim=beta_grad_integral(0);
  field<field<vec>> beta_r_Stim=beta_grad_integral(1);



  double U_old = 0;

  unsigned N=tau.n_elem;


  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall) reduction(+:U_old)
  for (unsigned i = 0; i < N; ++i) {
    double U_old_final_result = log_lhood(Inhib_likelihood(i),
                                          sigma, tau(i), tau_s(i), SSD(i),Ind_GF(i),
                                          b_l(i), b_r(i), nu_l(i), nu_r(i), nu_l_p(i), nu_r_p(i), nu_l_s(i), nu_r_s(i),
                                          Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                                          Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                                          lk_LCR_FS(i), lk_LIR_FS(i),lk_RCR_FS(i),lk_RIR_FS(i),
                                          prob_param.col(i),T_Go(i),k(i));


    U_old += U_old_final_result;
  }


  omp_set_num_threads(6);




  double prior_main_g=mvn_prior(gama,mu_gl,sigma_gl_inv,mu_gr,sigma_gr_inv,eta_b_l,eta_b_r);

  double prior_main_b=mvn_prior(beta,mu_bl,sigma_bl_inv,mu_br,sigma_br_inv,eta_nu_l,eta_nu_r);


  U_old+=(prior_main_g+prior_main_b);

  U_old *= -1;

  double H_old = U_old+(dot(p_g,p_g)/2)+(dot(p_b,p_b)/2);



  // Gama Gradient

  mat v_old_g(size(gama), fill::zeros);

  mat v_old_b(size(beta), fill::zeros);

  // field<mat>grad_gama_result(N);


  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall)
  for (unsigned i = 0; i < N; ++i) {
    mat grad_gama_result = grad_gama(t(i), tau(i),tau_s(i), sigma, SSD(i),DEL(i),DEL_s(i),
                                     nu_l(i), nu_r(i), b_l(i), b_r(i),
                                     nu_l_p(i), nu_r_p(i), nu_l_s(i), nu_r_s(i),

                                     weight_LCR(i),weight_LIR(i),weight_RCR(i),weight_RIR(i),
                                     weight_LCR_G_FS(i),weight_LIR_G_FS(i),weight_RCR_G_FS(i),weight_RIR_G_FS(i),
                                     weight_LCR_S(i),weight_LIR_S(i),weight_RCR_S(i),weight_RIR_S(i),

                                     Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RCR(i), Ind_RIR(i),
                                     Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RCR(i), Ind_S_RIR(i),
                                     Ind_I_L(i), Ind_I_R(i),

                                     diff_X_LCR(i),diff_Y_LCR(i),diff_X_LIR(i),diff_Y_LIR(i),
                                     diff_X_RCR(i),diff_Y_RCR(i),diff_X_RIR(i),diff_Y_RIR(i),

                                     gama_LS(i), gama_RS(i));


    // Rcpp::Rcout<<"grad_gama_result"<<grad_gama_result<<endl;

    mat grad_beta_result = grad_beta(t(i),tau(i),tau_s(i),sigma,SSD(i),DEL(i),DEL_s(i),
                                     nu_l(i),nu_l_p(i),nu_r(i),nu_r_p(i),nu_l_s(i),nu_r_s(i),
                                     b_l(i),b_r(i),nu_l_squared(i),nu_r_squared(i),
                                     penal_param.col(i),Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RIR(i),Ind_RCR(i),
                                     Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RIR(i),Ind_S_RCR(i),
                                     Ind_I_L(i),Ind_I_R(i),
                                     beta_l_Stim(i),beta_r_Stim(i),
                                     weight_LCR(i),weight_LIR(i),weight_RCR(i),weight_RIR(i),
                                     weight_LCR_G_FS(i),weight_LIR_G_FS(i),weight_RCR_G_FS(i),weight_RIR_G_FS(i),
                                     weight_LCR_S(i), weight_LIR_S(i),weight_RCR_S(i), weight_RIR_S(i),
                                     diff_X_LCR(i),diff_Y_LCR(i),diff_X_LIR(i),diff_Y_LIR(i),
                                     diff_X_RCR(i),diff_Y_RCR(i),diff_X_RIR(i),diff_Y_RIR(i),lt);


#pragma omp critical
{
  v_old_g -= grad_gama_result;

  v_old_b -= grad_beta_result;

}

  }

  omp_set_num_threads(6);



  mat gama_prior=grad_gama_prior(gama,eta_b_l,eta_b_r,mu_gl,sigma_gl_inv,mu_gr,sigma_gr_inv);
  v_old_g -=gama_prior;


  mat gama_new=gama;
  mat v_new_g=v_old_g;


  // Beta_gradient

  mat beta_prior=grad_beta_prior(beta,eta_nu_l,eta_nu_r,mu_bl,sigma_bl_inv,mu_br,sigma_br_inv);
  v_old_b -=beta_prior;


  mat beta_new=beta;
  mat v_new_b=v_old_b;




  double delta= randu(distr_param(range(0),range(1)));

  int nstep=std::clamp((int) R::rpois(L),1,leapmax);


  leap_frog_main_effect(t,Rand_gama_l,Rand_gama_r,Rand_beta_l,Rand_beta_r,
                        tau,tau_s,sigma,SSD,DEL,DEL_s,
                        gama_new,beta_new,delta_param,penal_param,stop_param,prob_param,
                        Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                        Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                        Ind_I_L,Ind_I_R,
                        gama_I,beta_I,
                        eta_b_l,eta_b_r,eta_nu_l,eta_nu_r,
                        mu_gl,sigma_gl_inv,mu_gr,sigma_gr_inv,
                        mu_bl,sigma_bl_inv,mu_br,sigma_br_inv,
                        v_new_g,v_new_b, nstep, delta, p_g,p_b,
                        nparall,lower_bound,upper_bound,lt);




  field <field<vec>> gama_ess_new = update_gama_param_ess1(t,Rand_gama_l,Rand_gama_r,gama_new);


  field <vec> b_l_new = gama_ess_new(0);
  field <vec> b_r_new = gama_ess_new(1);


  field<field<vec>>  beta_ess_new=update_beta_param_ess1(t,Rand_beta_l,Rand_beta_r,beta_new);

  field <vec> nu_l_new=beta_ess_new(0);
  field <vec> nu_r_new=beta_ess_new(1);
  field <vec> nu_l_squared_new=beta_ess_new(2);
  field <vec> nu_r_squared_new=beta_ess_new(3);

  field<field<vec>> beta_ess2_new=update_beta_param_ess3(nu_l_new,nu_r_new,nu_l_squared_new,nu_r_squared_new,
                                                         Ind_L,Ind_R,penal_param);

  field <vec> nu_l_p_new=beta_ess2_new(0);
  field <vec> nu_r_p_new=beta_ess2_new(1);





  field <field<vec>> stop_param_ess_new=  update_stop_param_ess2(nu_l_new,nu_r_new,stop_param,penal_param);





  field <vec> nu_l_s_new=stop_param_ess_new(0);
  field <vec> nu_r_s_new=stop_param_ess_new(1);



  field <field<vec>> lk_FS_new= update_lk_FS(tau,tau_s, sigma,SSD,b_l_new,b_r_new,nu_l_new,nu_r_new,nu_r_p_new,nu_l_p_new,
                                             nu_l_s_new,nu_r_s_new,Ind_L,Ind_R,
                                             Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                                             Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                             stop_param,prob_param,DEL,DEL_s,lt);


  field <vec> lk_LCR_FS_new=lk_FS_new(0);
  field <vec> lk_LIR_FS_new=lk_FS_new(1);

  field <vec> lk_RCR_FS_new=lk_FS_new(2);
  field <vec> lk_RIR_FS_new=lk_FS_new(3);



  field<field<vec>>Int_dens_new=  update_likelihood(sigma,SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                                    b_l_new,b_r_new,nu_l_new,nu_r_new,nu_l_s_new,nu_r_s_new,
                                                    nu_r_p_new,nu_l_p_new,Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                    lower_bound,upper_bound,nparall,lt);


  field <vec>  intgral_sum_new=update_integral_likelihood_sum(Int_dens_new);
  vec Inhib_likelihood_new=intgral_sum_new(0);



  double U_new=0;

  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall) reduction(+:U_new)
  for (unsigned i = 0; i < N; ++i) {
    double U_old_final_result2=log_lhood(Inhib_likelihood_new(i),
                                         sigma, tau(i), tau_s(i), SSD(i),Ind_GF(i),
                                         b_l_new(i),b_r_new(i), nu_l_new(i), nu_r_new(i), nu_l_p_new(i), nu_r_p_new(i),
                                         nu_l_s_new(i), nu_r_s_new(i),
                                         Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                                         Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                                         lk_LCR_FS_new(i), lk_LIR_FS_new(i),lk_RCR_FS_new(i),lk_RIR_FS_new(i),

                                         prob_param.col(i),T_Go(i),k(i));


    U_new+=U_old_final_result2;

  }

  omp_set_num_threads(6);


  double prior_main_new_g=mvn_prior(gama_new,mu_gl,sigma_gl_inv,mu_gr,sigma_gr_inv,eta_b_l,eta_b_r);

  double prior_main_new_b=mvn_prior(beta_new,mu_bl,sigma_bl_inv,mu_br,sigma_br_inv,eta_nu_l,eta_nu_r);


  U_new+=(prior_main_new_g+prior_main_new_b);

  U_new *= -1;


  //double U_new=-U_old_final2;

  double H_new = U_new + (dot(p_g, p_g) / 2)+(dot(p_b, p_b) / 2);


  // Check for NaN or Inf and break the loop if found
  if (!std::isfinite(U_new)  ) {
    // if (std::isnan(U_new) || std::isinf(U_new)) {
    Rcpp::Rcout << "U_new is NaN or Inf. Stopping." << std::endl;
    (gama.t()).print("gama: ");
    (beta.t()).print("beta: ");
    Rcpp::stop("");
  }


  if (log(randu()) < -(H_new - H_old)) {
    gama = gama_new;
    beta = beta_new;
    ++acceptance_main_effect;
  }


}




/////////////////////////////////////////////////////////////////////////////////////// Random Effect /////////////////////////////////////////////////////////////////////////////////////////////


//Leapfrog algorithm

void leap_frog_rand_effect(const mat &Phi,const vec &b_l_main,const vec &b_r_main,
                           const vec &nu_l_main,const vec &nu_r_main,
                           const vec &S_gama_l,const vec &S_gama_r,
                           const vec &rand_param_g_l,const vec &rand_param_g_r,
                           const vec &S_beta_l,const vec &S_beta_r,
                           const vec &rand_param_b_l,const vec &rand_param_b_r,
                           const vec &tau,const vec &tau_s, const double sigma,
                           const vec &SSD,const double DEL,const double DEL_s,
                           const vec &delta_prime,const vec &penal_param,
                           const vec &stop_param,const vec &prob_param,
                           const uvec &Ind_L, const uvec &Ind_R,
                           const uvec &Ind_LCR, const uvec &Ind_LIR,
                           const uvec &Ind_RCR, const uvec &Ind_RIR,
                           const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,
                           const uvec &Ind_S_RCR,const uvec &Ind_S_RIR,

                           const uvec &Ind_I_L,const uvec &Ind_I_R,

                           mat &gama_I,mat &beta_I,

                           mat &v_old_g_I,mat &v_old_b_I, const double L, const double delta,
                           mat &p_g_I,mat &p_b_I,

                           const vec &lower_bound,const double upper_bound,const bool lt =1) {

  // Rcpp::Rcout << "AAZ" << std::endl;

  for (unsigned j = 0; j < L; ++j) {

    gama_I += delta * (p_g_I - (delta / 2) * v_old_g_I);

    beta_I+=  delta * (p_b_I-   (delta/2)  * v_old_b_I);




    vec b_l = b_l_main+ (Phi * gama_I.col(0));
    vec b_r = b_r_main+ (Phi * gama_I.col(1));


    vec nu_l = nu_l_main+ (Phi * beta_I.col(0));
    vec nu_l_squared = square(nu_l);

    vec nu_r = nu_r_main+ (Phi * beta_I.col(1));
    vec nu_r_squared = square(nu_r);


    vec nu_l_p = h(nu_l, Ind_R ) - penalty_fun(h(nu_r, Ind_R ),h(nu_r_squared, Ind_R ),penal_param);
    vec nu_r_p = h(nu_r, Ind_L ) - penalty_fun(h(nu_l, Ind_L ),h(nu_l_squared, Ind_L ),penal_param);

    vec nu_l_s=nu_l-penalty_fun_s(stop_param(1),penal_param);
    vec nu_r_s=nu_r-penalty_fun_s(stop_param(1),penal_param);



    vec weight_LCR=weight(sigma,g(tau, Ind_L, Ind_LCR ),h(nu_r_p,Ind_LCR ),g(b_r, Ind_L, Ind_LCR ));

    vec weight_LIR=weight(sigma,g(tau, Ind_L, Ind_LIR ),g(nu_l, Ind_L, Ind_LIR ),g(b_l, Ind_L, Ind_LIR ));

    vec weight_RCR=weight(sigma,g(tau, Ind_R, Ind_RCR ),h(nu_l_p, Ind_RCR ),g(b_l, Ind_R, Ind_RCR));

    vec weight_RIR=weight(sigma,g(tau, Ind_R, Ind_RIR ),g(nu_r, Ind_R, Ind_RIR ),g(b_r, Ind_R, Ind_RIR ));


    vec weight_LCR_G_FS=weight(sigma,g(tau, Ind_L, Ind_S_LCR ),h(nu_r_p, Ind_S_LCR ),g(b_r, Ind_L, Ind_S_LCR ));

    vec weight_LIR_G_FS=weight(sigma,g(tau, Ind_L, Ind_S_LIR ),g(nu_l, Ind_L, Ind_S_LIR ),g(b_l, Ind_L, Ind_S_LIR ));

    vec weight_RCR_G_FS=weight(sigma,g(tau, Ind_R, Ind_S_RCR ),h(nu_l_p, Ind_S_RCR ),g(b_l, Ind_R, Ind_S_RCR ));

    vec weight_RIR_G_FS=weight(sigma,g(tau, Ind_R, Ind_S_RIR ),g(nu_r, Ind_R, Ind_S_RIR ),g(b_r, Ind_R, Ind_S_RIR ));


    vec weight_LCR_S=weight_s(sigma,g(tau, Ind_L, Ind_S_LCR ),g(tau_s, Ind_L, Ind_S_LCR ),g(SSD, Ind_L, Ind_S_LCR ),
                              g(b_r, Ind_L, Ind_S_LCR ),h(nu_r_p, Ind_S_LCR ),
                              g(nu_r_s, Ind_L, Ind_S_LCR ),DEL,DEL_s,lt);

    vec weight_LIR_S=weight_s(sigma,g(tau, Ind_L, Ind_S_LIR ),g(tau_s, Ind_L, Ind_S_LIR ),g(SSD, Ind_L, Ind_S_LIR ),
                              g(b_l, Ind_L, Ind_S_LIR ),g(nu_l,Ind_L, Ind_S_LIR ),
                              g(nu_l_s, Ind_L, Ind_S_LIR ),DEL,DEL_s,lt);

    vec weight_RCR_S=weight_s(sigma,g(tau, Ind_R, Ind_S_RCR ),g(tau_s, Ind_R, Ind_S_RCR ),g(SSD, Ind_R, Ind_S_RCR ),
                              g(b_l, Ind_R, Ind_S_RCR ),h(nu_l_p, Ind_S_RCR ),
                              g(nu_l_s,Ind_R, Ind_S_RCR ),DEL,DEL_s,lt);

    vec weight_RIR_S=weight_s(sigma, g(tau, Ind_R, Ind_S_RIR ),g(tau_s, Ind_R, Ind_S_RIR ),g(SSD, Ind_R, Ind_S_RIR ),
                              g(b_r, Ind_R, Ind_S_RIR ),g(nu_r, Ind_R, Ind_S_RIR ),
                              g(nu_r_s,Ind_R, Ind_S_RIR ),DEL,DEL_s,lt);




    field<vec> likelihood_integral =integral_likelihood(sigma,SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                                        b_l,nu_l,nu_l_s,
                                                        b_r,nu_r_p,nu_r_s,
                                                        nu_r,nu_l_p,
                                                        Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                        lower_bound,upper_bound,lt);

    //Gama Gradient

    field<vec>gama_LS=integrate_gama_Left_Stim(likelihood_integral,sigma,
                                               SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                               b_l, nu_l,nu_l_s,
                                               b_r, nu_r_p,nu_r_s,
                                               Ind_L,Ind_I_L,
                                               lower_bound,upper_bound,lt);



    field<vec>gama_RS=integrate_gama_Right_Stim(likelihood_integral,sigma,
                                                SSD,DEL,DEL_s,stop_param,penal_param,prob_param,
                                                b_l,nu_l_p,nu_l_s,
                                                b_r,nu_r,nu_r_s,Ind_R,Ind_I_R,
                                                lower_bound,upper_bound,lt);



    //Beta Gradient


    field<vec> beta_l_Stim=  integrate_beta_l(likelihood_integral,sigma,SSD,DEL,DEL_s,
                                              stop_param,penal_param,prob_param,
                                              b_l,b_r,nu_l,nu_r,nu_l_s,nu_r_s,nu_r_p,nu_l_p,
                                              nu_l_squared,
                                              Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                              lower_bound,upper_bound,lt);


    field<vec> beta_r_Stim=  integrate_beta_r(likelihood_integral,sigma,SSD,DEL,DEL_s,
                                              stop_param,penal_param,prob_param,
                                              b_l,b_r,nu_l,nu_r,nu_l_s,nu_r_s,nu_r_p,nu_l_p,
                                              nu_r_squared,
                                              Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                              lower_bound,upper_bound,lt);



    field<vec> lk_FS=update_lk_FS4(tau,tau_s, sigma,SSD,
                                   b_l,b_r,nu_l,nu_r,
                                   nu_r_p,nu_l_p,nu_l_s,nu_r_s,
                                   Ind_L,Ind_R,Ind_LCR, Ind_LIR,Ind_RCR,Ind_RIR,
                                   Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                   stop_param,prob_param,DEL,DEL_s,lt);


    vec diff_X_LCR=lk_FS(4);
    vec diff_Y_LCR=lk_FS(5);

    vec diff_X_LIR=lk_FS(6);
    vec diff_Y_LIR=lk_FS(7);

    vec diff_X_RCR=lk_FS(8);
    vec diff_Y_RCR=lk_FS(9);

    vec diff_X_RIR=lk_FS(10);
    vec diff_Y_RIR=lk_FS(11);



    // Gama_I

    mat v_new_g_I(size(gama_I), fill::zeros);


    v_new_g_I = -grad_gama(Phi,tau,tau_s,sigma,SSD,DEL,DEL_s,
                           nu_l,nu_r,b_l,b_r,
                           nu_l_p, nu_r_p, nu_l_s,nu_r_s,
                           weight_LCR,weight_LIR,weight_RCR,weight_RIR,
                           weight_LCR_G_FS,weight_LIR_G_FS,weight_RCR_G_FS,weight_RIR_G_FS,
                           weight_LCR_S,weight_LIR_S,weight_RCR_S,weight_RIR_S,
                           Ind_L, Ind_R, Ind_LCR, Ind_LIR, Ind_RCR, Ind_RIR,
                           Ind_S_LCR, Ind_S_LIR, Ind_S_RCR, Ind_S_RIR,
                           Ind_I_L, Ind_I_R,
                           diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                           diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR,

                           gama_LS, gama_RS);


    mat gama_prior=grad_Ind_prior(gama_I,S_gama_l,S_gama_r,rand_param_g_l(0),rand_param_g_r(0));

    v_new_g_I -=gama_prior;

    // Momentum
    p_g_I -= (delta / 2) * (v_old_g_I + v_new_g_I);
    v_old_g_I = v_new_g_I;


    // Beta_I

    mat v_new_b_I(size(beta_I), fill::zeros);

    v_new_b_I =-grad_beta(Phi,tau,tau_s,sigma,SSD,DEL,DEL_s,
                          nu_l,nu_l_p,nu_r,nu_r_p,nu_l_s,nu_r_s,
                          b_l,b_r,nu_l_squared,nu_r_squared,
                          penal_param,Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RIR,Ind_RCR,
                          Ind_S_LCR,Ind_S_LIR,Ind_S_RIR,Ind_S_RCR,
                          Ind_I_L,Ind_I_R,
                          beta_l_Stim,beta_r_Stim,
                          weight_LCR,weight_LIR,weight_RCR,weight_RIR,
                          weight_LCR_G_FS,weight_LIR_G_FS,weight_RCR_G_FS,weight_RIR_G_FS,
                          weight_LCR_S, weight_LIR_S,weight_RCR_S, weight_RIR_S,
                          diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                          diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR,lt);


    mat beta_I_prior=grad_Ind_prior(beta_I,S_beta_l,S_beta_r,rand_param_b_l(0),rand_param_b_r(0));

    v_new_b_I -=beta_I_prior;


    //momentum
    p_b_I-=(delta/2)*(v_old_b_I+v_new_b_I);
    v_old_b_I=v_new_b_I;
  }
}



//update Random effect


void update_rand_effect(const field <mat> &Phi,
                        const field <vec> &b_l_main,const field <vec> &b_r_main,
                        const field <vec> &nu_l_main,const field <vec> &nu_r_main,
                        const vec &S_gama_l,const vec &S_gama_r,
                        const vec &S_beta_l,const vec &S_beta_r,
                        const vec &rand_param_g_l,const vec &rand_param_g_r,
                        const vec &rand_param_b_l,const vec &rand_param_b_r,
                        const field <vec> &tau,const field<vec> &tau_s,const double sigma,const field <vec> &SSD,const vec &DEL,const vec &DEL_s,
                        const field <vec> &Ind_GF,const mat &delta_prime,
                        const mat &penal_param,const mat &stop_param,const mat &prob_param,

                        const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                        const field <uvec> &Ind_LCR, const field <uvec> &Ind_LIR,
                        const  field <uvec> &Ind_RCR,const field <uvec> &Ind_RIR,
                        const field <uvec> &Ind_S_LCR,const field <uvec> &Ind_S_LIR,
                        const field <uvec> &Ind_S_RCR,const field <uvec> &Ind_S_RIR,
                        const field <uvec>&Ind_I_L,const field <uvec>&Ind_I_R,
                        field <mat>  &gama_I, field <mat>  &beta_I,

                        const vec &T_Go,const vec &k,
                        const vec &range,const double L,const int leapmax,vec &acceptance_rand_effect,
                        const unsigned nparall,
                        const field <vec> &lower_bound,const double upper_bound,
                        const bool lt =1){


  // Rcpp::Rcout<<"ABA"<< endl;

  unsigned N=tau.n_elem;


  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall)
  for (unsigned i = 0; i < N; ++i){

    vec b_l = b_l_main(i)+ (Phi(i) * gama_I(i).col(0));
    vec b_r = b_r_main(i)+ (Phi(i) * gama_I(i).col(1));


    vec nu_l = nu_l_main(i)+ (Phi(i) * beta_I(i).col(0));
    vec nu_l_squared = square(nu_l);

    vec nu_r = nu_r_main(i)+ (Phi(i) * beta_I(i).col(1));
    vec nu_r_squared = square(nu_r);


    vec nu_l_p = h(nu_l, Ind_R(i) ) - penalty_fun(h(nu_r, Ind_R(i) ),h(nu_r_squared, Ind_R(i) ),penal_param.col(i));
    vec nu_r_p = h(nu_r, Ind_L(i) ) - penalty_fun(h(nu_l, Ind_L(i) ),h(nu_l_squared, Ind_L(i) ),penal_param.col(i));


    vec nu_l_s=nu_l-penalty_fun_s(stop_param(1,i),penal_param.col(i));
    vec nu_r_s=nu_r-penalty_fun_s(stop_param(1,i),penal_param.col(i));


    vec weight_LCR=weight(sigma,g(tau(i), Ind_L(i), Ind_LCR(i) ),h(nu_r_p, Ind_LCR(i) ),g(b_r, Ind_L(i), Ind_LCR(i) ));

    vec weight_LIR=weight(sigma,g(tau(i), Ind_L(i), Ind_LIR(i) ),g(nu_l, Ind_L(i), Ind_LIR(i) ),
                          g(b_l, Ind_L(i), Ind_LIR(i) ));

    vec weight_RCR=weight(sigma,g(tau(i), Ind_R(i), Ind_RCR(i) ),h(nu_l_p, Ind_RCR(i) ),g(b_l, Ind_R(i), Ind_RCR(i) ));

    vec weight_RIR=weight(sigma,g(tau(i), Ind_R(i), Ind_RIR(i) ),g(nu_r, Ind_R(i), Ind_RIR(i) ),
                          g(b_r, Ind_R(i), Ind_RIR(i) ));



    vec weight_LCR_G_FS=weight(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p, Ind_S_LCR(i) ),g(b_r, Ind_L(i), Ind_S_LCR(i) ));

    vec weight_LIR_G_FS=weight(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l, Ind_L(i), Ind_S_LIR(i) ),
                               g(b_l, Ind_L(i), Ind_S_LIR(i) ));

    vec weight_RCR_G_FS=weight(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p, Ind_S_RCR(i) ),g(b_l, Ind_R(i), Ind_S_RCR(i) ));

    vec weight_RIR_G_FS=weight(sigma,g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r, Ind_R(i), Ind_S_RIR(i) ),
                               g(b_r, Ind_R(i), Ind_S_RIR(i) ));


    vec weight_LCR_S=weight_s(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),g(SSD(i), Ind_L(i), Ind_S_LCR(i) ),
                              g(b_r, Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p, Ind_S_LCR(i) ),
                              g(nu_r_s, Ind_L(i), Ind_S_LCR(i) ),DEL(i),DEL_s(i),lt);


    vec weight_LIR_S=weight_s(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),g(SSD(i), Ind_L(i), Ind_S_LIR(i) ),
                              g(b_l, Ind_L(i), Ind_S_LIR(i) ),g(nu_l,Ind_L(i), Ind_S_LIR(i) ),
                              g(nu_l_s, Ind_L(i), Ind_S_LIR(i) ),DEL(i),DEL_s(i),lt);

    vec weight_RCR_S=weight_s(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),g(SSD(i), Ind_R(i), Ind_S_RCR(i) ),
                              g(b_l, Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p, Ind_S_RCR(i) ),
                              g(nu_l_s,Ind_R(i), Ind_S_RCR(i) ),DEL(i),DEL_s(i),lt);

    vec weight_RIR_S=weight_s(sigma, g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),g(SSD(i), Ind_R(i), Ind_S_RIR(i) ),
                              g(b_r, Ind_R(i), Ind_S_RIR(i) ),g(nu_r, Ind_R(i), Ind_S_RIR(i) ),
                              g(nu_r_s,Ind_R(i), Ind_S_RIR(i) ),DEL(i),DEL_s(i),lt);




    field<vec> likelihood_integral=integral_likelihood(sigma,SSD(i),DEL(i),DEL_s(i),
                                                       stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                       b_l,nu_l,nu_l_s,b_r,nu_r_p,nu_r_s,nu_r,nu_l_p,
                                                       Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                       lower_bound(i),upper_bound,lt);


    double Inhib_likelihood=sum(likelihood_integral(0))+sum(likelihood_integral(1));

    field<vec>gama_LS= integrate_gama_Left_Stim(likelihood_integral,sigma,
                                                SSD(i),DEL(i),DEL_s(i),stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                b_l, nu_l,nu_l_s,b_r, nu_r_p,nu_r_s,
                                                Ind_L(i),Ind_I_L(i),
                                                lower_bound(i),upper_bound,lt);



    field<vec>gama_RS=integrate_gama_Right_Stim(likelihood_integral,sigma,
                                                SSD(i),DEL(i),DEL_s(i),stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                b_l,nu_l_p,nu_l_s,b_r,nu_r,nu_r_s,
                                                Ind_R(i),Ind_I_R(i),
                                                lower_bound(i),upper_bound,lt);


    field<vec> beta_l_Stim=  integrate_beta_l(likelihood_integral,sigma,SSD(i),DEL(i),DEL_s(i),
                                              stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                              b_l,b_r,nu_l,nu_r,nu_l_s,nu_r_s,nu_r_p,nu_l_p,
                                              nu_l_squared,
                                              Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                              lower_bound(i),upper_bound,lt);




    field<vec> beta_r_Stim=  integrate_beta_r(likelihood_integral,sigma,SSD(i),DEL(i),DEL_s(i),
                                              stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                              b_l,b_r,nu_l,nu_r,nu_l_s,nu_r_s,nu_r_p,nu_l_p,
                                              nu_r_squared,
                                              Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                              lower_bound(i),upper_bound,lt);


    field<vec> lk_FS=update_lk_FS4(tau(i),tau_s(i), sigma,SSD(i),
                                   b_l,b_r,nu_l,nu_r,
                                   nu_r_p,nu_l_p,nu_l_s,nu_r_s,
                                   Ind_L(i),Ind_R(i),Ind_LCR(i), Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                                   Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                                   stop_param.col(i),prob_param.col(i),DEL(i),DEL_s(i),lt);


    vec lk_LCR_FS=lk_FS(0);;
    vec lk_LIR_FS=lk_FS(1);;

    vec lk_RCR_FS=lk_FS(2);;
    vec lk_RIR_FS=lk_FS(3);;


    vec diff_X_LCR=lk_FS(4);
    vec diff_Y_LCR=lk_FS(5);

    vec diff_X_LIR=lk_FS(6);
    vec diff_Y_LIR=lk_FS(7);

    vec diff_X_RCR=lk_FS(8);
    vec diff_Y_RCR=lk_FS(9);

    vec diff_X_RIR=lk_FS(10);
    vec diff_Y_RIR=lk_FS(11);


    //Gama_I update steps

    mat p_g_I(size(gama_I(i)),fill::randn);

    mat p_b_I(size(beta_I(i)),fill::randn);


    double U_old = -log_lhood(Inhib_likelihood,
                              sigma, tau(i), tau_s(i), SSD(i),Ind_GF(i),
                              b_l, b_r, nu_l, nu_r, nu_l_p, nu_r_p, nu_l_s, nu_r_s,
                              Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                              Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                              lk_LCR_FS, lk_LIR_FS,lk_RCR_FS,lk_RIR_FS,
                              prob_param.col(i),T_Go(i),k(i));


    double prior_main_g_I=mvn_prior2(rand_param_g_l(0),rand_param_g_r(0),gama_I(i),S_gama_l,S_gama_r);

    double prior_main_b_I=mvn_prior2(rand_param_b_l(0),rand_param_b_r(0),
                                     beta_I(i),S_beta_l,S_beta_r);



    U_old-=(prior_main_g_I+prior_main_b_I);


    double H_old = U_old+(dot(p_g_I,p_g_I)/2)+(dot(p_b_I,p_b_I)/2);


    mat v_old_g_I(size(gama_I(i)), fill::zeros);

    v_old_g_I =- grad_gama(Phi(i), tau(i),tau_s(i), sigma, SSD(i),DEL(i),DEL_s(i),
                           nu_l, nu_r, b_l, b_r,
                           nu_l_p, nu_r_p, nu_l_s, nu_r_s,
                           weight_LCR,weight_LIR,weight_RCR,weight_RIR,
                           weight_LCR_G_FS,weight_LIR_G_FS,weight_RCR_G_FS,weight_RIR_G_FS,
                           weight_LCR_S,weight_LIR_S,weight_RCR_S,weight_RIR_S,
                           Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RCR(i), Ind_RIR(i),
                           Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RCR(i), Ind_S_RIR(i),
                           Ind_I_L(i), Ind_I_R(i),
                           diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                           diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR,
                           gama_LS, gama_RS);


    mat gama_I_prior=grad_Ind_prior(gama_I(i),S_gama_l,S_gama_r,rand_param_g_l(0),rand_param_g_r(0));

    v_old_g_I -=gama_I_prior;


    mat gama_I_new=gama_I(i);
    mat v_new_g_I=v_old_g_I;



    //Beta_I update steps

    mat v_old_b_I(size(beta_I(i)), fill::zeros);


    v_old_b_I =-grad_beta(Phi(i),tau(i),tau_s(i),sigma,SSD(i),DEL(i),DEL_s(i),
                          nu_l,nu_l_p,nu_r,nu_r_p,nu_l_s,nu_r_s,
                          b_l,b_r,nu_l_squared,nu_r_squared,
                          penal_param.col(i),Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RIR(i),Ind_RCR(i),
                          Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RIR(i),Ind_S_RCR(i),
                          Ind_I_L(i),Ind_I_R(i),
                          beta_l_Stim,beta_r_Stim,
                          weight_LCR,weight_LIR,weight_RCR,weight_RIR,
                          weight_LCR_G_FS,weight_LIR_G_FS,weight_RCR_G_FS,weight_RIR_G_FS,
                          weight_LCR_S, weight_LIR_S,weight_RCR_S, weight_RIR_S,
                          diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                          diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR,lt);



    mat beta_I_prior=grad_Ind_prior(beta_I(i),S_beta_l,S_beta_r,rand_param_b_l(0),rand_param_b_r(0));

    v_old_b_I -=beta_I_prior;


    mat beta_I_new=beta_I(i);
    mat v_new_b_I=v_old_b_I;



    double delta= randu(distr_param(range(0),range(1)));

    //int nstep=R::rpois(nleapfrog);       nstep=std::clamp(nstep,1,(int) leapmax);

    // int pois_draw=(unsigned) R::rpois(L);
    // int nstep =  GSL_MAX_INT(1, pois_draw);
    // nstep=GSL_MIN_INT(nstep,leapmax);
    //

    int nstep=std::clamp((int) R::rpois(L),1,leapmax);


    leap_frog_rand_effect(Phi(i),b_l_main(i),b_r_main(i),
                          nu_l_main(i),nu_r_main(i),
                          S_gama_l,S_gama_r,
                          rand_param_g_l,rand_param_g_r,
                          S_beta_l,S_beta_r,
                          rand_param_b_l,rand_param_b_r,
                          tau(i),tau_s(i),sigma,SSD(i),DEL(i),DEL_s(i),
                          delta_prime.col(i),penal_param.col(i),stop_param.col(i),prob_param.col(i),
                          Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                          Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                          Ind_I_L(i),Ind_I_R(i),gama_I_new,beta_I_new,

                          v_new_g_I,v_new_b_I,nstep,delta,p_g_I,p_b_I,
                          lower_bound(i),upper_bound,lt);



    // vec gama_I_l_new=gama_I_new.col(0);
    // vec gama_I_r_new=gama_I_new.col(1);

    // vec beta_I_l_new=beta_I_new.col(0);
    // vec beta_I_r_new=beta_I_new.col(1);


    vec b_l_new = b_l_main(i)+ (Phi(i) * gama_I_new.col(0));
    vec b_r_new = b_r_main(i)+ (Phi(i) * gama_I_new.col(1));


    vec nu_l_new = nu_l_main(i)+ (Phi(i) * beta_I_new.col(0));
    vec nu_l_squared_new = square(nu_l_new);

    vec nu_r_new = nu_r_main(i)+ (Phi(i) * beta_I_new.col(1));
    vec nu_r_squared_new = square(nu_r_new);


    vec nu_l_p_new = h(nu_l_new, Ind_R(i) ) - penalty_fun(h(nu_r_new, Ind_R(i) ),h(nu_r_squared_new, Ind_R(i) ),penal_param.col(i));
    vec nu_r_p_new = h(nu_r_new, Ind_L(i) ) - penalty_fun(h(nu_l_new, Ind_L(i) ),h(nu_l_squared_new, Ind_L(i) ),penal_param.col(i));


    vec nu_l_s_new=nu_l_new-penalty_fun_s(stop_param(1,i),penal_param.col(i));
    vec nu_r_s_new=nu_r_new-penalty_fun_s(stop_param(1,i),penal_param.col(i));



    field<vec> likelihood_integral_new=integral_likelihood(sigma,SSD(i),DEL(i),DEL_s(i),stop_param.col(i),
                                                           penal_param.col(i),prob_param.col(i),
                                                           b_l_new,nu_l_new,nu_l_s_new,
                                                           b_r_new,nu_r_p_new,nu_r_s_new,
                                                           nu_r_new,nu_l_p_new,
                                                           Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                           lower_bound(i),upper_bound,lt);


    double Inhib_likelihood_new=sum(likelihood_integral_new(0))+sum(likelihood_integral_new(1));


    field<vec> lk_FS_new= update_lk_FS3(tau(i),tau_s(i),sigma,SSD(i),
                                        b_l_new,b_r_new,nu_l_new,nu_r_new,nu_r_p_new,nu_l_p_new,
                                        nu_l_s_new,nu_r_s_new,
                                        Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                                        Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                                        stop_param.col(i),prob_param.col(i),DEL(i),DEL_s(i),lt);


    vec lk_LCR_FS_new=lk_FS_new(0);
    vec lk_LIR_FS_new=lk_FS_new(1);

    vec lk_RCR_FS_new=lk_FS_new(2);
    vec lk_RIR_FS_new=lk_FS_new(3);


    double U_new=-log_lhood(Inhib_likelihood_new,
                            sigma, tau(i), tau_s(i), SSD(i),Ind_GF(i),
                            b_l_new,b_r_new, nu_l_new, nu_r_new, nu_l_p_new, nu_r_p_new, nu_l_s_new, nu_r_s_new,
                            Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                            Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                            lk_LCR_FS_new, lk_LIR_FS_new,lk_RCR_FS_new,lk_RIR_FS_new,

                            prob_param.col(i),T_Go(i),k(i));


    double prior_main_new_g_I=mvn_prior2(rand_param_g_l(0),rand_param_g_r(0),gama_I_new,S_gama_l,S_gama_r);

    double prior_main_new_b_I=mvn_prior2(rand_param_b_l(0),rand_param_b_r(0),beta_I_new,S_beta_l,S_beta_r);

    U_new-=(prior_main_new_g_I+prior_main_new_b_I);


    double H_new = U_new + (dot(p_g_I, p_g_I) / 2)+(dot(p_b_I, p_b_I) / 2);

    if (log(randu()) < -(H_new - H_old)) {
      gama_I(i) = gama_I_new;
      beta_I(i) = beta_I_new;
#pragma omp critical
{

  ++acceptance_rand_effect[i];
}
    }


  }

  omp_set_num_threads(6);
}



/////////////////////////////////////////////////////////////////////////// penalty Paramater ////////////////////////////////////////////////////////////////////////////////////



//Leapfrog algorithm


void leap_frog_penal_param(const vec &tau,const vec &tau_s,const double sigma,
                           const vec &delta_prime,const vec &SSD,const double DEL,const double DEL_s,

                           const vec &stop_param,const vec &prob_param,

                           const vec &b_l,const vec &b_r,

                           const vec &nu_l,const vec &nu_r,const vec &nu_l_squared,const vec &nu_r_squared,
                           const uvec &Ind_L,const uvec &Ind_R,const uvec &Ind_LCR,const uvec &Ind_LIR,
                           const uvec &Ind_RIR,const uvec &Ind_RCR,
                           const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,const uvec &Ind_S_RIR,const uvec &Ind_S_RCR,

                           const uvec &Ind_I_L,const uvec &Ind_I_R,


                           vec &penal_param,const double mu_lambda_prime,const double sigma_lambda_prime,
                           const double mu_alpha_prime,const double sigma_alpha_prime,
                           vec &v_old,const double L, const double delta,mat &p,

                           const vec &lower_bound,const double upper_bound,const bool lt = true){

  // Rcpp::Rcout<<"leap_frog"<< endl;

  for (unsigned j = 0; j < L; ++j){
    penal_param+=delta*(p-(delta/2)*v_old);


    vec nu_l_p=h(nu_l, Ind_R ) - penalty_fun(h(nu_r, Ind_R),h(nu_r_squared, Ind_R ),penal_param);
    vec nu_r_p=h(nu_r, Ind_L ) - penalty_fun(h(nu_l, Ind_L),h(nu_l_squared, Ind_L ),penal_param);


    vec nu_l_s=nu_l-penalty_fun_s(stop_param(1),penal_param);
    vec nu_r_s=nu_r-penalty_fun_s(stop_param(1),penal_param);


    vec weight_LCR=weight(sigma,g(tau, Ind_L, Ind_LCR ),h(nu_r_p,Ind_LCR ),g(b_r, Ind_L, Ind_LCR ));
    vec weight_RCR=weight(sigma,g(tau, Ind_R, Ind_RCR ),h(nu_l_p, Ind_RCR ),g(b_l, Ind_R, Ind_RCR));


    vec weight_LCR_S=weight_s(sigma,g(tau, Ind_L, Ind_S_LCR ),g(tau_s, Ind_L, Ind_S_LCR ),g(SSD, Ind_L, Ind_S_LCR ),
                              g(b_r, Ind_L, Ind_S_LCR ),h(nu_r_p, Ind_S_LCR ),
                              g(nu_r_s, Ind_L, Ind_S_LCR ),DEL,DEL_s,lt);

    vec weight_LIR_S=weight_s(sigma,g(tau, Ind_L, Ind_S_LIR ),g(tau_s, Ind_L, Ind_S_LIR ),g(SSD, Ind_L, Ind_S_LIR ),
                              g(b_l, Ind_L, Ind_S_LIR ),g(nu_l,Ind_L, Ind_S_LIR ),
                              g(nu_l_s, Ind_L, Ind_S_LIR ),DEL,DEL_s,lt);

    vec weight_RCR_S=weight_s(sigma,g(tau, Ind_R, Ind_S_RCR ),g(tau_s, Ind_R, Ind_S_RCR ),g(SSD, Ind_R, Ind_S_RCR ),
                              g(b_l, Ind_R, Ind_S_RCR ),h(nu_l_p, Ind_S_RCR ),
                              g(nu_l_s,Ind_R, Ind_S_RCR ),DEL,DEL_s,lt);

    vec weight_RIR_S=weight_s(sigma, g(tau, Ind_R, Ind_S_RIR ),g(tau_s, Ind_R, Ind_S_RIR ),g(SSD, Ind_R, Ind_S_RIR ),
                              g(b_r, Ind_R, Ind_S_RIR ),g(nu_r, Ind_R, Ind_S_RIR ),
                              g(nu_r_s,Ind_R, Ind_S_RIR ),DEL,DEL_s,lt);




    vec weight_LCR_G_FS=weight(sigma,g(tau, Ind_L, Ind_S_LCR ),h(nu_r_p, Ind_S_LCR ),g(b_r, Ind_L, Ind_S_LCR ));

    vec weight_RCR_G_FS=weight(sigma,g(tau, Ind_R, Ind_S_RCR ),h(nu_l_p, Ind_S_RCR ),g(b_l, Ind_R, Ind_S_RCR ));



    field<vec> likelihood_integral=integral_likelihood(sigma,SSD,DEL,DEL_s,
                                                       stop_param,penal_param,prob_param,
                                                       b_l,nu_l,nu_l_s,
                                                       b_r,nu_r_p,nu_r_s,
                                                       nu_r,nu_l_p,
                                                       Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                       lower_bound,upper_bound,lt);

    mat Penal_LS=integrate_penalty_Left_Stim(likelihood_integral,sigma,SSD,DEL,DEL_s,
                                             penal_param,stop_param,prob_param,
                                             b_l,b_r,nu_l,nu_r,
                                             nu_l_s,nu_r_s,
                                             nu_r_p,nu_l_squared,
                                             Ind_L,Ind_I_L,
                                             lower_bound,upper_bound,lt);


    mat Penal_RS =integrate_penal_Right_Stim(likelihood_integral,sigma,SSD,DEL,DEL_s,
                                             penal_param,stop_param,prob_param,
                                             b_l,b_r,nu_l,nu_r,
                                             nu_l_s,nu_r_s,
                                             nu_l_p,nu_r_squared,
                                             Ind_R,Ind_I_R,
                                             lower_bound,upper_bound,lt);




    field<vec> lk_FS=update_lk_FS4(tau,tau_s, sigma,SSD,
                                   b_l,b_r,nu_l,nu_r,
                                   nu_r_p,nu_l_p,nu_l_s,nu_r_s,
                                   Ind_L,Ind_R,Ind_LCR, Ind_LIR,Ind_RCR,Ind_RIR,
                                   Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                   stop_param,prob_param,DEL,DEL_s,lt);



    vec diff_X_LCR = lk_FS(4);
    vec diff_Y_LCR = lk_FS(5);
    vec diff_X_LIR = lk_FS(6);
    vec diff_Y_LIR = lk_FS(7);
    vec diff_X_RCR = lk_FS(8);
    vec diff_Y_RCR = lk_FS(9);
    vec diff_X_RIR = lk_FS(10);
    vec diff_Y_RIR = lk_FS(11);




    //new gradient

    vec v_new=-grad_penal_param(tau,tau_s,sigma,SSD,DEL,DEL_s,
                                b_l,b_r,nu_l,nu_r,nu_r_p,nu_l_p,
                                nu_l_squared,nu_r_squared,nu_l_s,nu_r_s,
                                Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RIR,Ind_RCR,
                                Ind_S_LCR,Ind_S_LIR,Ind_S_RIR,Ind_S_RCR,
                                Ind_I_L,Ind_I_R,
                                Penal_LS,Penal_RS,
                                weight_LCR,weight_RCR,
                                weight_LCR_G_FS,weight_RCR_G_FS,
                                weight_LCR_S,weight_LIR_S,weight_RCR_S,weight_RIR_S,
                                penal_param,stop_param,mu_lambda_prime,sigma_lambda_prime,
                                mu_alpha_prime,sigma_alpha_prime,
                                diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                                diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR,lt);


    //momentum
    p-=(delta/2)*(v_old+v_new);
    v_old=v_new;


  }
}






//Update penalty parameter

void update_penal_param(const field <vec> &tau,const field <vec> &tau_s,const double sigma,const field <vec> &SSD,
                        const vec &DEL,const vec &DEL_s,const field <vec> &Ind_GF,
                        const mat &delta_prime,const mat &gama,const mat &beta,mat &penal_param,const mat &stop_param,
                        const field <vec> &b_l,const field <vec> &b_r,
                        const field <vec> &nu_l,const field <vec> &nu_r,
                        const field <vec> &nu_l_squared,const field <vec> &nu_r_squared,
                        const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                        const field <uvec> &Ind_LCR, const field <uvec> &Ind_LIR,
                        const field <uvec> &Ind_RCR,const field <uvec> &Ind_RIR,
                        const field <uvec> &Ind_S_LCR, const field <uvec> &Ind_S_LIR,
                        const field <uvec> &Ind_S_RCR,const field <uvec> &Ind_S_RIR,
                        const field <uvec> &Ind_I_L,const field <uvec> &Ind_I_R,
                        const double mu_lambda_prime,const double sigma_lambda_prime,
                        const double mu_alpha_prime,const double sigma_alpha_prime,

                        const mat &prob_param,
                        const vec &T_Go,const vec &k,
                        const vec &range,const double L,const int leapmax,vec &acceptance_p,
                        unsigned nparall,
                        const field <vec> &lower_bound,const double upper_bound,const bool lt = true){

  // Rcpp::Rcout<<"update"<< endl;

  unsigned N=nu_l.n_elem;

  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall)
  for (unsigned i = 0; i < N; ++i){

    // Rcpp::Rcout<< "subject: " << i <<endl;

    //Updating neceassary vectors

    vec nu_l_p_i= h(nu_l(i), Ind_R(i) ) - penalty_fun(h(nu_r(i), Ind_R(i) ),h(nu_r_squared(i), Ind_R(i) ),penal_param.col(i));
    vec nu_r_p_i= h(nu_r(i), Ind_L(i) ) - penalty_fun(h(nu_l(i), Ind_L(i) ),h(nu_l_squared(i), Ind_L(i) ),penal_param.col(i));



    vec nu_l_s_i=nu_l(i)-penalty_fun_s(stop_param(1,i),penal_param.col(i));
    vec nu_r_s_i=nu_r(i)-penalty_fun_s(stop_param(1,i),penal_param.col(i));


    vec weight_LCR=weight(sigma,g(tau(i), Ind_L(i), Ind_LCR(i) ),h(nu_r_p_i,Ind_LCR(i) ),g(b_r(i), Ind_L(i), Ind_LCR(i) ));
    vec weight_RCR=weight(sigma,g(tau(i), Ind_R(i), Ind_RCR(i) ),h(nu_l_p_i, Ind_RCR(i) ),g(b_l(i), Ind_R(i), Ind_RCR(i) ));


    vec weight_LCR_G_FS=weight(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p_i, Ind_S_LCR(i) ),g(b_r(i), Ind_L(i), Ind_S_LCR(i) ));

    vec weight_RCR_G_FS=weight(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p_i, Ind_S_RCR(i) ),g(b_l(i), Ind_R(i), Ind_S_RCR(i) ));



    vec weight_LCR_S=weight_s(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),g(SSD(i), Ind_L(i), Ind_S_LCR(i) ),
                              g(b_r(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p_i, Ind_S_LCR(i) ),
                              g(nu_r_s_i, Ind_L(i), Ind_S_LCR(i) ),DEL(i),DEL_s(i),lt);


    vec weight_LIR_S=weight_s(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),g(SSD(i), Ind_L(i), Ind_S_LIR(i) ),
                              g(b_l(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l(i),Ind_L(i), Ind_S_LIR(i) ),
                              g(nu_l_s_i, Ind_L(i), Ind_S_LIR(i) ),DEL(i),DEL_s(i),lt);

    vec weight_RCR_S=weight_s(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),g(SSD(i), Ind_R(i), Ind_S_RCR(i) ),
                              g(b_l(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p_i, Ind_S_RCR(i) ),
                              g(nu_l_s_i,Ind_R(i), Ind_S_RCR(i) ),DEL(i),DEL_s(i),lt);

    vec weight_RIR_S=weight_s(sigma, g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),g(SSD(i), Ind_R(i), Ind_S_RIR(i) ),
                              g(b_r(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RIR(i) ),
                              g(nu_r_s_i,Ind_R(i), Ind_S_RIR(i) ),DEL(i),DEL_s(i),lt);



    field<vec> likelihood_integral=integral_likelihood(sigma,SSD(i),DEL(i),DEL_s(i),
                                                       stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                       b_l(i),nu_l(i),nu_l_s_i,
                                                       b_r(i),nu_r_p_i,nu_r_s_i,
                                                       nu_r(i),nu_l_p_i,
                                                       Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                       lower_bound(i),upper_bound,lt);




    double Inhibit_likelihood=sum(likelihood_integral(0))+sum(likelihood_integral(1));


    // Rcpp::Rcout<<"Inhibit_likelihood"<<Inhibit_likelihood<<endl;


    mat Penal_LS=integrate_penalty_Left_Stim(likelihood_integral,sigma,SSD(i),DEL(i),DEL_s(i),
                                             penal_param.col(i),stop_param.col(i),prob_param.col(i),
                                             b_l(i),b_r(i),nu_l(i),nu_r(i),
                                             nu_l_s_i,nu_r_s_i,
                                             nu_r_p_i,nu_l_squared(i),
                                             Ind_L(i),Ind_I_L(i),
                                             lower_bound(i),upper_bound,lt);


    mat Penal_RS =integrate_penal_Right_Stim(likelihood_integral,sigma,SSD(i),DEL(i),DEL_s(i),
                                             penal_param.col(i),stop_param.col(i),prob_param.col(i),
                                             b_l(i),b_r(i),nu_l(i),nu_r(i),
                                             nu_l_s_i,nu_r_s_i,
                                             nu_l_p_i,nu_r_squared(i),
                                             Ind_R(i),Ind_I_R(i),
                                             lower_bound(i),upper_bound,lt);




    field<vec> lk_FS=update_lk_FS4(tau(i),tau_s(i), sigma,SSD(i),
                                   b_l(i),b_r(i),nu_l(i),nu_r(i),
                                   nu_r_p_i,nu_l_p_i,nu_l_s_i,nu_r_s_i,
                                   Ind_L(i),Ind_R(i),Ind_LCR(i), Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                                   Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                                   stop_param.col(i),prob_param.col(i),DEL(i),DEL_s(i),lt);



    vec lk_LCR_FS=lk_FS(0);
    vec lk_LIR_FS=lk_FS(1);

    vec lk_RCR_FS=lk_FS(2);
    vec lk_RIR_FS=lk_FS(3);


    vec diff_X_LCR=lk_FS(4);
    vec diff_Y_LCR=lk_FS(5);

    vec diff_X_LIR=lk_FS(6);
    vec diff_Y_LIR=lk_FS(7);

    vec diff_X_RCR=lk_FS(8);
    vec diff_Y_RCR=lk_FS(9);

    vec diff_X_RIR=lk_FS(10);
    vec diff_Y_RIR=lk_FS(11);




    //penalty parameter update steps

    vec p(size(penal_param.col(i)),fill::randn);


    double U_old=-log_lhood(Inhibit_likelihood,
                            sigma, tau(i), tau_s(i), SSD(i),Ind_GF(i),
                            b_l(i), b_r(i), nu_l(i), nu_r(i), nu_l_p_i, nu_r_p_i, nu_l_s_i, nu_r_s_i,
                            Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                            Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                            lk_LCR_FS, lk_LIR_FS,lk_RCR_FS,lk_RIR_FS,

                            prob_param.col(i),T_Go(i),k(i));


    double prior_main=log_normal_prior(penal_param.col(i),mu_lambda_prime,sigma_lambda_prime,
                                       mu_alpha_prime,sigma_alpha_prime);

    U_old-=prior_main;


    // Rcpp::Rcout<<"U_old"<<U_old<<endl;

    double H_old = U_old+(dot(p,p)/2);

    vec v_old =- grad_penal_param(tau(i),tau_s(i),sigma,SSD(i),DEL(i),DEL_s(i),
                                  b_l(i),b_r(i),nu_l(i),nu_r(i),nu_r_p_i,nu_l_p_i,
                                  nu_l_squared(i),nu_r_squared(i),nu_l_s_i,nu_r_s_i,
                                  Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RIR(i),Ind_RCR(i),
                                  Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RIR(i),Ind_S_RCR(i),
                                  Ind_I_L(i),Ind_I_R(i),
                                  Penal_LS,Penal_RS,
                                  weight_LCR,weight_RCR,
                                  weight_LCR_G_FS,weight_RCR_G_FS,
                                  weight_LCR_S,weight_LIR_S,weight_RCR_S,weight_RIR_S,
                                  penal_param.col(i),stop_param.col(i),mu_lambda_prime,sigma_lambda_prime,
                                  mu_alpha_prime,sigma_alpha_prime,
                                  diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                                  diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR,lt);


    vec penal_param_new=penal_param.col(i);
    vec v_new=v_old;

    double delta=randu(distr_param(range(0),range(1)));


    // int pois_draw=(unsigned) R::rpois(L);
    // int nstep =  GSL_MAX_INT(1, pois_draw);
    // nstep=GSL_MIN_INT(nstep, leapmax);
    //

    int nstep=std::clamp((int) R::rpois(L),1,leapmax);

    leap_frog_penal_param(tau(i),tau_s(i),sigma,delta_prime.col(i),SSD(i),DEL(i),DEL_s(i),
                          stop_param.col(i),prob_param.col(i),
                          b_l(i),b_r(i),
                          nu_l(i),nu_r(i),nu_l_squared(i),nu_r_squared(i),
                          Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RIR(i),Ind_RCR(i),
                          Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RIR(i),Ind_S_RCR(i),

                          Ind_I_L(i),Ind_I_R(i),

                          penal_param_new,mu_lambda_prime,sigma_lambda_prime,
                          mu_alpha_prime,sigma_alpha_prime,
                          v_new,nstep,delta,p,
                          lower_bound(i),upper_bound,lt);


    vec nu_l_p_i_new= h(nu_l(i), Ind_R(i) ) - penalty_fun(h(nu_r(i), Ind_R(i) ),h(nu_r_squared(i), Ind_R(i) ),penal_param_new);
    vec nu_r_p_i_new= h(nu_r(i), Ind_L(i) ) - penalty_fun(h(nu_l(i), Ind_L(i) ),h(nu_l_squared(i), Ind_L(i) ),penal_param_new);



    vec nu_l_s_i_new=nu_l(i)-penalty_fun_s(stop_param(1,i),penal_param_new);
    vec nu_r_s_i_new=nu_r(i)-penalty_fun_s(stop_param(1,i),penal_param_new);




    field<vec> likelihood_integral_new=integral_likelihood(sigma,SSD(i),DEL(i),DEL_s(i),stop_param.col(i),
                                                           penal_param_new,prob_param.col(i),
                                                           b_l(i),nu_l(i),nu_l_s_i_new,
                                                           b_r(i),nu_r_p_i_new,nu_r_s_i_new,
                                                           nu_r(i),nu_l_p_i_new,
                                                           Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                           lower_bound(i),upper_bound,lt);



    double Inhib_likelihood_new=sum(likelihood_integral_new(0))+sum(likelihood_integral_new(1));


    field<vec> lk_FS_new= update_lk_FS3(tau(i),tau_s(i),sigma,SSD(i),
                                        b_l(i),b_r(i),nu_l(i),nu_r(i),nu_r_p_i_new,nu_l_p_i_new,
                                        nu_l_s_i_new,nu_r_s_i_new,
                                        Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                                        Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                                        stop_param.col(i),prob_param.col(i),DEL(i),DEL_s(i),lt);

    vec lk_LCR_FS_new=lk_FS_new(0);
    vec lk_LIR_FS_new=lk_FS_new(1);

    vec lk_RCR_FS_new=lk_FS_new(2);
    vec lk_RIR_FS_new=lk_FS_new(3);



    double U_new=-  log_lhood(Inhib_likelihood_new,
                              sigma, tau(i), tau_s(i), SSD(i),Ind_GF(i),
                              b_l(i), b_r(i), nu_l(i), nu_r(i), nu_l_p_i_new,nu_r_p_i_new,
                              nu_l_s_i_new,nu_r_s_i_new,
                              Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                              Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                              lk_LCR_FS_new, lk_LIR_FS_new,lk_RCR_FS_new,lk_RIR_FS_new,
                              prob_param.col(i),T_Go(i),k(i));


    double prior_main_new=log_normal_prior(penal_param_new,mu_lambda_prime,sigma_lambda_prime,
                                           mu_alpha_prime,sigma_alpha_prime);

    U_new-=prior_main_new;

    // Rcpp::Rcout<<"U_new"<<U_new<<endl;

    double H_new=U_new+(dot(p,p)/2);


    // Rcpp::Rcout<<"penal_param"<<penal_param.t()<<endl;


    // bool bl= log(randu())< -(H_new-H_old);


    // if(bl ){
    if(log(randu())< -(H_new-H_old)){
#pragma omp critical
{
  penal_param.col(i)=penal_param_new;
  ++acceptance_p[i];
}
    }






  }

  omp_set_num_threads(6);
}





/////////////////////////////////////////////////////////////////////////////////////// Delta_prime /////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////// Delta_prime /////////////////////////////////////////////////////////////////////////////////////////////


//Leapfrog algorithm



void leap_frog_delta_param(const vec &tau,const vec &tau_stop,const double sigma,const vec &SSD,const double SSD_min,const double U,
                           const vec &stop_param,const vec &penal_param,const vec &prob_param,vec &delta_param,
                           const vec &b_l,const vec &b_r,
                           const vec &nu_l,const vec &nu_r,
                           const vec &nu_l_p,const vec &nu_r_p,const vec &nu_l_s, const vec &nu_r_s,
                           const uvec &Ind_L,const uvec &Ind_R,
                           const uvec &Ind_LCR,const uvec &Ind_LIR,const uvec &Ind_RIR,const uvec &Ind_RCR,
                           const uvec &Ind_S_LCR,const uvec &Ind_S_LIR,const uvec &Ind_S_RIR,const uvec &Ind_S_RCR,
                           const uvec &Ind_I_L,const uvec &Ind_I_R,
                           vec &v_old,const double L, const double delta,mat &p,
                           const vec &lower_bound_init,const double upper_bound,const bool lt=1){


  // Rcpp::Rcout<<"ABF"<< endl;

  for (unsigned j = 0; j < L; ++j){


    delta_param+=delta*(p-(delta/2)*v_old);

// --- compute DEL_s ---
double DEL_s = U * EXPITE(delta_param(0));

// --- compute DEL ---
double DEL = (SSD_min + DEL_s) * EXPITE(delta_param(1));

    
    vec tau_prime(tau.n_elem, fill::zeros);
    vec tau_s(tau_stop.n_elem, fill::zeros);

    // Apply subtraction only where tau > 0
    uvec idx_tau_pos = find(tau > 0);
    tau_prime.elem(idx_tau_pos) = tau.elem(idx_tau_pos) - DEL;

    // Apply subtraction only where tau_stop > 0
    uvec idx_tau_stop_pos = find(tau_stop > 0);
    tau_s.elem(idx_tau_stop_pos) = tau_stop.elem(idx_tau_stop_pos) - DEL_s;



    vec lower_bound=lower_bound_init+DEL_s;

    double deriv_delta=(SSD_min+DEL_s)*EXPITE_sq(delta_param(1));

    double deriv_delta_s=EXPITE(delta_param(1))*U*EXPITE_sq(delta_param(0));

    double deriv_delta_s_stop=U*EXPITE_sq(delta_param(0));


    vec weight_LCR=weight(sigma,g(tau_prime, Ind_L, Ind_LCR ),h(nu_r_p,Ind_LCR ),g(b_r, Ind_L, Ind_LCR ));
    vec weight_RCR=weight(sigma,g(tau_prime, Ind_R, Ind_RCR ),h(nu_l_p, Ind_RCR ),g(b_l, Ind_R, Ind_RCR));
    vec weight_LIR=weight(sigma,g(tau_prime, Ind_L, Ind_LIR ),g(nu_l,Ind_L,Ind_LIR ),g(b_l, Ind_L, Ind_LIR ));
    vec weight_RIR=weight(sigma,g(tau_prime, Ind_R, Ind_RIR ),g(nu_r,Ind_R, Ind_RIR ),g(b_r, Ind_R, Ind_RIR ));




    vec weight_LCR_G_FS=weight(sigma,g(tau_prime, Ind_L, Ind_S_LCR ),h(nu_r_p, Ind_S_LCR ),g(b_r, Ind_L, Ind_S_LCR ));

    vec weight_LIR_G_FS=weight(sigma,g(tau_prime, Ind_L, Ind_S_LIR ),g(nu_l, Ind_L, Ind_S_LIR ),g(b_l, Ind_L, Ind_S_LIR ));

    vec weight_RCR_G_FS=weight(sigma,g(tau_prime, Ind_R, Ind_S_RCR ),h(nu_l_p, Ind_S_RCR ),g(b_l, Ind_R, Ind_S_RCR ));

    vec weight_RIR_G_FS=weight(sigma,g(tau_prime, Ind_R, Ind_S_RIR ),g(nu_r, Ind_R, Ind_S_RIR ),g(b_r, Ind_R, Ind_S_RIR ));




    vec weight_LCR_S=weight_s(sigma,g(tau_prime, Ind_L, Ind_S_LCR ),g(tau_s, Ind_L, Ind_S_LCR ),g(SSD, Ind_L, Ind_S_LCR ),
                              g(b_r, Ind_L, Ind_S_LCR ),h(nu_r_p, Ind_S_LCR ),
                              g(nu_r_s, Ind_L, Ind_S_LCR ),DEL,DEL_s,lt);

    vec weight_LIR_S=weight_s(sigma,g(tau_prime, Ind_L, Ind_S_LIR ),g(tau_s, Ind_L, Ind_S_LIR ),g(SSD, Ind_L, Ind_S_LIR ),
                              g(b_l, Ind_L, Ind_S_LIR ),g(nu_l,Ind_L, Ind_S_LIR ),
                              g(nu_l_s, Ind_L, Ind_S_LIR ),DEL,DEL_s,lt);

    vec weight_RCR_S=weight_s(sigma,g(tau_prime, Ind_R, Ind_S_RCR ),g(tau_s, Ind_R, Ind_S_RCR ),g(SSD, Ind_R, Ind_S_RCR ),
                              g(b_l, Ind_R, Ind_S_RCR ),h(nu_l_p, Ind_S_RCR ),
                              g(nu_l_s,Ind_R, Ind_S_RCR ),DEL,DEL_s,lt);

    vec weight_RIR_S=weight_s(sigma, g(tau_prime, Ind_R, Ind_S_RIR ),g(tau_s, Ind_R, Ind_S_RIR ),g(SSD, Ind_R, Ind_S_RIR ),
                              g(b_r, Ind_R, Ind_S_RIR ),g(nu_r, Ind_R, Ind_S_RIR ),
                              g(nu_r_s,Ind_R, Ind_S_RIR ),DEL,DEL_s,lt);



    vec weight_FS_LCR=weight_stop(g(tau_s, Ind_L, Ind_S_LCR ), sigma,stop_param);

    vec weight_FS_LIR=weight_stop(g(tau_s, Ind_L, Ind_S_LIR ), sigma,stop_param);

    vec weight_FS_RCR=weight_stop(g(tau_s, Ind_R, Ind_S_RCR ), sigma,stop_param);

    vec weight_FS_RIR=weight_stop(g(tau_s, Ind_R, Ind_S_RIR ), sigma,stop_param);




    field<vec> likelihood_integral=integral_likelihood(sigma,SSD,DEL,DEL_s,
                                                       stop_param,penal_param,prob_param,
                                                       b_l,nu_l,nu_l_s,
                                                       b_r,nu_r_p,nu_r_s,
                                                       nu_r,nu_l_p,
                                                       Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                       lower_bound,upper_bound,lt);




    field<vec> delta_prime_Stim=    integrate_delta_prime(likelihood_integral,sigma,
                                                          stop_param,penal_param,prob_param,
                                                          SSD,DEL,DEL_s,b_l,b_r,nu_l,nu_r,
                                                          nu_l_s,nu_r_s,nu_r_p,nu_l_p,deriv_delta,
                                                          Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                          lower_bound,upper_bound,lt);



    field<vec> delta_s_Stim=    integrate_delta_prime_s(likelihood_integral,sigma,
                                                        stop_param,penal_param,prob_param,
                                                        SSD,DEL,DEL_s,b_l,b_r,nu_l,nu_r,
                                                        nu_l_s,nu_r_s,nu_r_p,nu_l_p,
                                                        deriv_delta_s,deriv_delta_s_stop,
                                                        Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                        lower_bound,upper_bound,lt);








    field<vec> lk_FS=update_lk_FS4(tau_prime,tau_s, sigma,SSD,
                                   b_l,b_r,nu_l,nu_r,
                                   nu_r_p,nu_l_p,nu_l_s,nu_r_s,
                                   Ind_L,Ind_R,Ind_LCR, Ind_LIR,Ind_RCR,Ind_RIR,
                                   Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                   stop_param,prob_param,DEL,DEL_s,lt);




    vec diff_X_LCR=lk_FS(4);
    vec diff_Y_LCR=lk_FS(5);

    vec diff_X_LIR=lk_FS(6);
    vec diff_Y_LIR=lk_FS(7);

    vec diff_X_RCR=lk_FS(8);
    vec diff_Y_RCR=lk_FS(9);

    vec diff_X_RIR=lk_FS(10);
    vec diff_Y_RIR=lk_FS(11);



    //new gradient
    vec v_new=-grad_delta(tau_prime,tau_s,sigma,SSD,DEL,DEL_s,delta_param,stop_param,
                          nu_l, nu_r,b_l, b_r,
                          nu_l_p,nu_r_p,nu_l_s,nu_r_s,delta_prime_Stim,delta_s_Stim,
                          weight_LIR, weight_RCR,weight_LCR,weight_RIR,
                          weight_LCR_G_FS, weight_LIR_G_FS,weight_RCR_G_FS,weight_RIR_G_FS,
                          weight_LIR_S,weight_RCR_S,weight_LCR_S,weight_RIR_S,
                          weight_FS_LCR,weight_FS_LIR,weight_FS_RCR,weight_FS_RIR,
                          diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                          diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR,
                          deriv_delta,deriv_delta_s,deriv_delta_s_stop,

                          Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                          Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,lt);



    //momentum
    p-=(delta/2)*(v_old+v_new);
    v_old=v_new;
  }


}



//Update delta_prime parameter

void update_delta_param(const field <vec> &tau,const field <vec> &tau_stop,
                        const double sigma,const field <vec> &SSD,const double SSD_min,const vec &U,const field <vec> &Ind_GF,
                        mat &delta_param,const mat &gama,const mat &beta,const mat &penal_param,const mat &stop_param,
                        const field <vec> &b_l,const field <vec> &b_r,
                        const field <vec> &nu_l,const field <vec> &nu_r,
                        const field <vec> &nu_l_p,const field <vec> &nu_r_p,
                        const field <vec> &nu_l_s,const field <vec> &nu_r_s,
                        const field <uvec> &Ind_L,const field <uvec> &Ind_R,
                        const field <uvec> &Ind_LCR, const field <uvec> &Ind_LIR,
                        const field <uvec> &Ind_RCR,const field <uvec> &Ind_RIR,
                        const field <uvec> &Ind_S_LCR, const field <uvec> &Ind_S_LIR,
                        const field <uvec> &Ind_S_RCR,const field <uvec> &Ind_S_RIR,
                        const field <uvec> &Ind_I_L,const field <uvec> &Ind_I_R,

                        const mat &prob_param,
                        const vec &T_Go,const vec &k,
                        const vec &range,const double L,const int leapmax,vec &acceptance_d,
                        unsigned nparall,
                        const field <vec> &lower_bound_init,const double upper_bound,const bool lt = 1){


  // Rcpp::Rcout<<"ABG"<< endl;



  unsigned N=nu_l.n_elem;

  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall)
  for (unsigned i = 0; i < N; ++i){

    //Updating neceassary vectors

    double DEL_s=U(i)*EXPITE(delta_param(0,i));

    double DEL=(SSD_min+DEL_s)*EXPITE(delta_param(1,i));

    vec tau_i = tau(i);
    vec tau_stop_i = tau_stop(i);


    vec tau_prime(tau_i.n_elem, fill::zeros);
    vec tau_s(tau_stop_i.n_elem, fill::zeros);

    // Apply subtraction only where tau > 0
    uvec idx_tau_pos = find(tau_i > 0);
    tau_prime.elem(idx_tau_pos) = tau_i.elem(idx_tau_pos) - DEL;

    // Apply subtraction only where tau_stop > 0
    uvec idx_tau_stop_pos = find(tau_stop_i > 0);
    tau_s.elem(idx_tau_stop_pos) = tau_stop_i.elem(idx_tau_stop_pos) - DEL_s;


    vec lower_bound=lower_bound_init(i)+DEL_s;


    double deriv_delta=(SSD_min+DEL_s)*EXPITE_sq(delta_param(1));

    double deriv_delta_s=EXPITE(delta_param(1))*U(i)*EXPITE_sq(delta_param(0));

    double deriv_delta_s_stop=U(i)*EXPITE_sq(delta_param(0));


    vec weight_LCR=weight(sigma,g(tau_prime, Ind_L(i), Ind_LCR(i) ),h(nu_r_p(i),Ind_LCR(i) ),
                          g(b_r(i), Ind_L(i), Ind_LCR(i) ));

    vec weight_LIR=weight(sigma,g(tau_prime, Ind_L(i), Ind_LIR(i) ),g(nu_l(i),Ind_L(i),Ind_LIR(i) ),
                          g(b_l(i), Ind_L(i), Ind_LIR(i) ));

    vec weight_RIR=weight(sigma,g(tau_prime, Ind_R(i), Ind_RIR(i) ),g(nu_r(i),Ind_R(i), Ind_RIR(i) ),
                          g(b_r(i), Ind_R(i), Ind_RIR(i) ));

    vec weight_RCR=weight(sigma,g(tau_prime, Ind_R(i), Ind_RCR(i) ),h(nu_l_p(i), Ind_RCR(i) ),
                          g(b_l(i), Ind_R(i), Ind_RCR(i) ));


    vec weight_LCR_G_FS=weight(sigma,g(tau_prime, Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p(i), Ind_S_LCR(i) ),g(b_r(i), Ind_L(i), Ind_S_LCR(i) ));

    vec weight_LIR_G_FS=weight(sigma,g(tau_prime, Ind_L(i), Ind_S_LIR(i) ),g(nu_l(i), Ind_L(i), Ind_S_LIR(i) ),
                               g(b_l(i), Ind_L(i), Ind_S_LIR(i) ));

    vec weight_RCR_G_FS=weight(sigma,g(tau_prime, Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p(i), Ind_S_RCR(i) ),g(b_l(i), Ind_R(i), Ind_S_RCR(i) ));

    vec weight_RIR_G_FS=weight(sigma,g(tau_prime, Ind_R(i), Ind_S_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RIR(i) ),
                               g(b_r(i), Ind_R(i), Ind_S_RIR(i) ));




    vec weight_LCR_S=weight_s(sigma,g(tau_prime, Ind_L(i), Ind_S_LCR(i) ),g(tau_s, Ind_L(i), Ind_S_LCR(i) ),g(SSD(i), Ind_L(i), Ind_S_LCR(i) ),
                              g(b_r(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p(i), Ind_S_LCR(i) ),
                              g(nu_r_s(i), Ind_L(i), Ind_S_LCR(i) ),DEL,DEL_s,lt);

    vec weight_LIR_S=weight_s(sigma,g(tau_prime, Ind_L(i), Ind_S_LIR(i) ),g(tau_s, Ind_L(i), Ind_S_LIR(i) ),g(SSD(i), Ind_L(i), Ind_S_LIR(i) ),
                              g(b_l(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l(i),Ind_L(i), Ind_S_LIR(i) ),
                              g(nu_l_s(i), Ind_L(i), Ind_S_LIR(i) ),DEL,DEL_s,lt);

    vec weight_RCR_S=weight_s(sigma,g(tau_prime, Ind_R(i), Ind_S_RCR(i) ),g(tau_s, Ind_R(i), Ind_S_RCR(i) ),g(SSD(i), Ind_R(i), Ind_S_RCR(i) ),
                              g(b_l(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p(i), Ind_S_RCR(i) ),
                              g(nu_l_s(i),Ind_R(i), Ind_S_RCR(i) ),DEL,DEL_s,lt);


    vec weight_RIR_S=weight_s(sigma, g(tau_prime, Ind_R(i), Ind_S_RIR(i) ),g(tau_s, Ind_R(i), Ind_S_RIR(i) ),g(SSD(i), Ind_R(i), Ind_S_RIR(i) ),
                              g(b_r(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RIR(i) ),
                              g(nu_r_s(i),Ind_R(i), Ind_S_RIR(i) ),DEL,DEL_s,lt);


    vec weight_FS_LIR=weight_stop(g(tau_s, Ind_L(i), Ind_S_LIR(i) ), sigma,stop_param.col(i));



    vec weight_FS_RCR=weight_stop(g(tau_s, Ind_R(i), Ind_S_RCR(i) ), sigma,stop_param.col(i));


    vec weight_FS_RIR=weight_stop(g(tau_s, Ind_R(i), Ind_S_RIR(i) ), sigma,stop_param.col(i));



    vec weight_FS_LCR=weight_stop(g(tau_s, Ind_L(i), Ind_S_LCR(i) ), sigma,stop_param.col(i));


    field<vec> likelihood_integral=integral_likelihood(sigma,SSD(i),DEL,DEL_s,
                                                       stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                       b_l(i),nu_l(i),nu_l_s(i),
                                                       b_r(i),nu_r_p(i),nu_r_s(i),
                                                       nu_r(i),nu_l_p(i),
                                                       Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                       lower_bound,upper_bound,lt);


    double Inhibit_likelihood=sum(likelihood_integral(0))+sum(likelihood_integral(1));



    field<vec> delta_prime_Stim =integrate_delta_prime(likelihood_integral,sigma,
                                                       stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                       SSD(i),DEL,DEL_s,b_l(i),b_r(i),nu_l(i),nu_r(i),
                                                       nu_l_s(i),nu_r_s(i),nu_r_p(i),nu_l_p(i),deriv_delta,
                                                       Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                       lower_bound,upper_bound,lt);

    field<vec> delta_s_Stim =integrate_delta_prime_s(likelihood_integral,sigma,
                                                     stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                     SSD(i),DEL,DEL_s,b_l(i),b_r(i),nu_l(i),nu_r(i),
                                                     nu_l_s(i),nu_r_s(i),nu_r_p(i),nu_l_p(i),
                                                     deriv_delta_s,deriv_delta_s_stop,
                                                     Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                     lower_bound,upper_bound,lt);



    field<vec> lk_FS=update_lk_FS4(tau_prime,tau_s, sigma,SSD(i),
                                   b_l(i),b_r(i),nu_l(i),nu_r(i),
                                   nu_r_p(i),nu_l_p(i),nu_l_s(i),nu_r_s(i),
                                   Ind_L(i),Ind_R(i),Ind_LCR(i), Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                                   Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                                   stop_param.col(i),prob_param.col(i),DEL,DEL_s,lt);


    vec lk_LCR_FS=lk_FS(0);
    vec lk_LIR_FS=lk_FS(1);

    vec lk_RCR_FS=lk_FS(2);
    vec lk_RIR_FS=lk_FS(3);


    vec diff_X_LCR=lk_FS(4);
    vec diff_Y_LCR=lk_FS(5);

    vec diff_X_LIR=lk_FS(6);
    vec diff_Y_LIR=lk_FS(7);

    vec diff_X_RCR=lk_FS(8);
    vec diff_Y_RCR=lk_FS(9);

    vec diff_X_RIR=lk_FS(10);
    vec diff_Y_RIR=lk_FS(11);





    //delta_prime update steps


    vec p(size(delta_param.col(i)),fill::randn);


    double U_old=-log_lhood(Inhibit_likelihood,
                            sigma, tau_prime, tau_s, SSD(i),Ind_GF(i),
                            b_l(i), b_r(i), nu_l(i), nu_r(i), nu_l_p(i), nu_r_p(i), nu_l_s(i), nu_r_s(i),
                            Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                            Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                            lk_LCR_FS, lk_LIR_FS,lk_RCR_FS,lk_RIR_FS,
                            prob_param.col(i),T_Go(i),k(i));


    double prior_main=delta_param(1,i)-(2*log(1+exp(delta_param(1,i))))+
                      delta_param(0,i)-(2*log(1+exp(delta_param(0,i))));


    U_old-=prior_main;

    // Rcpp::Rcout<<"U_old"<<U_old<<endl;

    double H_old = U_old+ (dot(p,p)/2);



    vec v_old =-    grad_delta(tau_prime,tau_s,sigma,SSD(i),DEL,DEL_s,
                               delta_param.col(i),stop_param.col(i),
                               nu_l(i), nu_r(i),b_l(i), b_r(i),
                               nu_l_p(i),nu_r_p(i),nu_l_s(i),nu_r_s(i),delta_prime_Stim,delta_s_Stim,
                               weight_LIR, weight_RCR,weight_LCR,weight_RIR,
                               weight_LCR_G_FS, weight_LIR_G_FS,weight_RCR_G_FS,weight_RIR_G_FS,
                               weight_LIR_S,weight_RCR_S,weight_LCR_S,weight_RIR_S,

                               weight_FS_LCR,weight_FS_LIR,weight_FS_RCR,weight_FS_RIR,
                               diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                               diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR,

                               deriv_delta,deriv_delta_s,deriv_delta_s_stop,

                               Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                               Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),lt);


    // Rcpp::Rcout<<"v_old"<<v_old<<endl;


    vec delta_param_new=delta_param.col(i);



    vec v_new=v_old;

    double delta=randu(distr_param(range(0),range(1)));
    // int pois_draw=(unsigned) R::rpois(L);
    // int nstep =  GSL_MAX_INT(1, pois_draw);
    // nstep=GSL_MIN_INT(nstep,leapmax);
    //

    int nstep=std::clamp((int) R::rpois(L),1,leapmax);

    leap_frog_delta_param(tau(i),tau_stop(i),sigma,SSD(i),SSD_min,U(i),
                          stop_param.col(i),penal_param.col(i),prob_param.col(i),delta_param_new,
                          b_l(i),b_r(i),
                          nu_l(i),nu_r(i),
                          nu_l_p(i),nu_r_p(i),nu_l_s(i),nu_r_s(i),
                          Ind_L(i),Ind_R(i),
                          Ind_LCR(i),Ind_LIR(i),Ind_RIR(i),Ind_RCR(i),
                          Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RIR(i),Ind_S_RCR(i),
                          Ind_I_L(i),Ind_I_R(i),
                          v_new,nstep,delta,p,lower_bound_init(i),upper_bound,lt);



    double DEL_s_new=U(i)*EXPITE(delta_param_new(0));

    double DEL_new=(SSD_min+DEL_s_new)*EXPITE(delta_param_new(1));


    vec tau_prime_new(tau_i.n_elem, fill::zeros);
    vec tau_s_new(tau_stop_i.n_elem, fill::zeros);

    tau_prime_new.elem(idx_tau_pos) = tau_i.elem(idx_tau_pos) - DEL_new;

    tau_s_new.elem(idx_tau_stop_pos) = tau_stop_i.elem(idx_tau_stop_pos) - DEL_s_new;


    vec lower_bound_new=lower_bound_init(i)+DEL_s_new;



    field<vec> likelihood_integral_new=integral_likelihood(sigma,SSD(i),DEL_new,DEL_s_new,
                                                           stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                           b_l(i),nu_l(i),nu_l_s(i),
                                                           b_r(i),nu_r_p(i),nu_r_s(i),
                                                           nu_r(i),nu_l_p(i),
                                                           Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                           lower_bound_new,upper_bound,lt);





    double Inhib_likelihood_new=sum(likelihood_integral_new(0))+sum(likelihood_integral_new(1));



    field<vec> lk_FS_new= update_lk_FS3(tau_prime_new,tau_s_new,sigma,SSD(i),
                                        b_l(i),b_r(i),nu_l(i),nu_r(i),nu_r_p(i),nu_l_p(i),
                                        nu_l_s(i),nu_r_s(i),
                                        Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                                        Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                                        stop_param.col(i),prob_param.col(i),DEL,DEL_s,lt);

    vec lk_LCR_FS_new=lk_FS_new(0);
    vec lk_LIR_FS_new=lk_FS_new(1);

    vec lk_RCR_FS_new=lk_FS_new(2);
    vec lk_RIR_FS_new=lk_FS_new(3);




    double U_new=-log_lhood(Inhib_likelihood_new,
                            sigma, tau_prime_new, tau_s_new, SSD(i),Ind_GF(i),
                            b_l(i), b_r(i), nu_l(i), nu_r(i), nu_l_p(i), nu_r_p(i), nu_l_s(i), nu_r_s(i),
                            Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                            Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                            lk_LCR_FS_new, lk_LIR_FS_new,lk_RCR_FS_new,lk_RIR_FS_new,
                            prob_param.col(i),T_Go(i),k(i));



   double prior_main_new=delta_param_new(1)-(2*log(1+exp(delta_param_new(1))))+
                         delta_param_new(0)-(2*log(1+exp(delta_param_new(0))));



    U_new-=prior_main_new;


    double H_new=U_new+ (dot(p,p)/2);


    if (log(randu()) < -(H_new - H_old)) {
#pragma omp critical
{

  delta_param.col(i) = delta_param_new;
  ++acceptance_d[i];
}
    }


    omp_set_num_threads(6);
  }

}





/////////////////////////////////////////////////////////////////////////////////////// nu_stop /////////////////////////////////////////////////////////////////////////////////////////////

//Leapfrog algorithm

void leap_frog_stop_prob_param(const vec &tau, const vec &tau_s, const double sigma,const vec &delta_prime,
                               const vec &SSD,const double DEL,const double DEL_s,
                               const vec &b_l, const vec &b_r,
                               const vec &nu_l, const vec &nu_r, const vec &nu_l_p, const vec &nu_r_p,
                               const vec &nu_l_squared, const vec &nu_r_squared,

                               const uvec &Ind_L, const uvec &Ind_R,
                               const uvec &Ind_LCR, const uvec &Ind_LIR, const uvec &Ind_RCR,const uvec &Ind_RIR,
                               const uvec &Ind_S_LCR, const uvec &Ind_S_LIR, const uvec &Ind_S_RCR,const uvec &Ind_S_RIR,
                               const uvec &Ind_I_L, const uvec &Ind_I_R,
                               vec &stop_param,const vec &penal_param,vec &prob_param,
                               const double mu_b_stop, const double sigma_b_stop, const double mu_nu_stop,const double sigma_nu_stop,

                               const vec &prob_hyp,
                               const double T_Go,const double k,const double T_Total,

                               vec &v_old_stop,vec &v_old_prob,const double L, const double delta,
                               mat &p_stop,mat &p_prob,


                               const vec &lower_bound, const double upper_bound, const bool lt = 1) {


  // Rcpp::Rcout<<"ABJ"<< endl;


  for (unsigned j = 0; j < L; ++j) {

    stop_param += delta * (p_stop - (delta / 2) * v_old_stop);

    prob_param += delta * (p_prob - (delta / 2) * v_old_prob);






    double sum_Prob = mlpack::AccuLog(prob_param);


    vec nu_l_s = nu_l - penalty_fun_s(stop_param(1), penal_param);
    vec nu_r_s = nu_r - penalty_fun_s(stop_param(1), penal_param);


    vec weight_LCR_S=weight_s(sigma,g(tau, Ind_L, Ind_S_LCR ),g(tau_s, Ind_L, Ind_S_LCR ),g(SSD, Ind_L, Ind_S_LCR ),
                              g(b_r, Ind_L, Ind_S_LCR ),h(nu_r_p, Ind_S_LCR ),
                              g(nu_r_s, Ind_L, Ind_S_LCR ),DEL,DEL_s,lt);

    vec weight_LIR_S=weight_s(sigma,g(tau, Ind_L, Ind_S_LIR ),g(tau_s, Ind_L, Ind_S_LIR ),g(SSD, Ind_L, Ind_S_LIR ),
                              g(b_l, Ind_L, Ind_S_LIR ),g(nu_l,Ind_L, Ind_S_LIR ),
                              g(nu_l_s, Ind_L, Ind_S_LIR ),DEL,DEL_s,lt);

    vec weight_RCR_S=weight_s(sigma,g(tau, Ind_R, Ind_S_RCR ),g(tau_s, Ind_R, Ind_S_RCR ),g(SSD, Ind_R, Ind_S_RCR ),
                              g(b_l, Ind_R, Ind_S_RCR ),h(nu_l_p, Ind_S_RCR ),
                              g(nu_l_s,Ind_R, Ind_S_RCR ),DEL,DEL_s,lt);

    vec weight_RIR_S=weight_s(sigma, g(tau, Ind_R, Ind_S_RIR ),g(tau_s, Ind_R, Ind_S_RIR ),g(SSD, Ind_R, Ind_S_RIR ),
                              g(b_r, Ind_R, Ind_S_RIR ),g(nu_r, Ind_R, Ind_S_RIR ),
                              g(nu_r_s,Ind_R, Ind_S_RIR ),DEL,DEL_s,lt);


    vec weight_FS_LCR=weight_stop(g(tau_s, Ind_L, Ind_S_LCR ), sigma,stop_param);

    vec weight_FS_LIR=weight_stop(g(tau_s, Ind_L, Ind_S_LIR ), sigma,stop_param);

    vec weight_FS_RCR=weight_stop(g(tau_s, Ind_R, Ind_S_RCR ), sigma,stop_param);

    vec weight_FS_RIR=weight_stop(g(tau_s, Ind_R, Ind_S_RIR ), sigma,stop_param);



    field<vec> likelihood_integral =integral_likelihood(sigma,SSD,DEL,DEL_s,
                                                        stop_param,penal_param,prob_param,
                                                        b_l,nu_l,nu_l_s,
                                                        b_r,nu_r_p,nu_r_s,
                                                        nu_r,nu_l_p,
                                                        Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                                        lower_bound,upper_bound,lt);




    field<vec> b_stop_Stim=integrate_b_stop(likelihood_integral,sigma,SSD,DEL,DEL_s,
                                            stop_param,penal_param,prob_param,
                                            b_l,b_r,nu_l,nu_r,nu_l_s,nu_r_s,nu_r_p,nu_l_p,
                                            Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                            lower_bound,upper_bound,lt);



    field<vec> nu_stop_Stim =integrate_nu_stop(likelihood_integral, sigma,SSD,DEL,DEL_s,
                                               stop_param,penal_param,prob_param,
                                               b_l,b_r,nu_l,nu_r,nu_l_s,nu_r_s,
                                               nu_r_p,nu_l_p,nu_l_squared,nu_r_squared,
                                               Ind_L,Ind_R,Ind_I_L,Ind_I_R,
                                               lower_bound,upper_bound,lt);



    field<vec> lk_FS=update_lk_FS4(tau,tau_s, sigma,SSD,
                                   b_l,b_r,nu_l,nu_r,
                                   nu_r_p,nu_l_p,nu_l_s,nu_r_s,
                                   Ind_L,Ind_R,Ind_LCR, Ind_LIR,Ind_RCR,Ind_RIR,
                                   Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                   stop_param,prob_param,DEL,DEL_s,lt);


    vec diff_X_LCR=lk_FS(4);
    vec diff_Y_LCR=lk_FS(5);

    vec diff_X_LIR=lk_FS(6);
    vec diff_Y_LIR=lk_FS(7);

    vec diff_X_RCR=lk_FS(8);
    vec diff_Y_RCR=lk_FS(9);

    vec diff_X_RIR=lk_FS(10);
    vec diff_Y_RIR=lk_FS(11);


    vec v_new_stop = -grad_stop_param(tau,tau_s,sigma,SSD,DEL,DEL_s,
                                      nu_l,nu_r,b_l,b_r,
                                      nu_l_p,nu_r_p,nu_l_s,nu_r_s,
                                      weight_LCR_S, weight_LIR_S,weight_RCR_S,weight_RIR_S,
                                      weight_FS_LCR,weight_FS_LIR,weight_FS_RCR,weight_FS_RIR,
                                      Ind_L,Ind_R,
                                      Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                                      nu_stop_Stim,b_stop_Stim,
                                      stop_param,penal_param,
                                      mu_nu_stop,sigma_nu_stop,mu_b_stop,sigma_b_stop,
                                      diff_Y_LCR,diff_Y_LIR,diff_Y_RCR,diff_Y_RIR,lt);



    vec v_new_prob= -grad_prob_param(tau_s,sigma,likelihood_integral,
                                     stop_param,prob_param,sum_Prob,prob_hyp,
                                     T_Go,k,T_Total,
                                     diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                                     diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR);


    p_stop -= (delta / 2) * (v_old_stop + v_new_stop);
    v_old_stop = v_new_stop;

    p_prob-=  (delta/2) *(v_old_prob+v_new_prob);
    v_old_prob=  v_new_prob;

  }
}



//Update nu_stop parameter


void update_stop_prob_param(const field<vec> &tau, const field<vec> &tau_s, const double sigma,const field<vec> &SSD,
                            const vec &DEL,const vec &DEL_s,
                            const field <vec> &Ind_GF,const mat &delta_prime,
                            const mat &gama, const mat &beta,const mat &penal_param, mat &stop_param,mat &prob_param,
                            const field<vec> &b_l, const field<vec> &b_r,
                            const field<vec> &nu_l, const field<vec> &nu_r,
                            const field<vec> &nu_l_p, const field<vec> &nu_r_p,
                            const field<vec> &nu_l_squared, const field<vec> &nu_r_squared,

                            const field<uvec> &Ind_L, const field<uvec> &Ind_R,
                            const field<uvec> &Ind_LCR, const field<uvec> &Ind_LIR, const field<uvec> &Ind_RCR,const field<uvec> &Ind_RIR,

                            const field<uvec> &Ind_S_LCR, const field<uvec> &Ind_S_LIR, const field<uvec> &Ind_S_RCR, const field<uvec> &Ind_S_RIR,
                            const field<uvec> &Ind_I_L, const field<uvec> &Ind_I_R,
                            const double mu_b_stop, const double sigma_b_stop,const double mu_nu_stop, const double sigma_nu_stop,

                            const vec &prob_hyp,
                            const vec &T_Go,const vec &k,const vec &T_Total,

                            const vec &range, const double L,
                            const int leapmax, vec &acceptance_stop_prob, unsigned nparall,
                            const field<vec> &lower_bound, const double upper_bound, const bool lt = true) {

  // Rcpp::Rcout<<"ABK"<< endl;

  unsigned N=nu_l.n_elem;


  omp_set_num_threads(1);
#pragma omp parallel for num_threads(nparall)
  for (unsigned i = 0; i < N; ++i) {


    double sum_prob = mlpack::AccuLog(prob_param.col(i));


    vec nu_l_s_i=nu_l(i)-penalty_fun_s(stop_param(1,i),penal_param.col(i));
    vec nu_r_s_i=nu_r(i)-penalty_fun_s(stop_param(1,i),penal_param.col(i));



    vec weight_LCR_S=weight_s(sigma,g(tau(i), Ind_L(i), Ind_S_LCR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ),g(SSD(i), Ind_L(i), Ind_S_LCR(i) ),
                              g(b_r(i), Ind_L(i), Ind_S_LCR(i) ),h(nu_r_p(i), Ind_S_LCR(i) ),
                              g(nu_r_s_i, Ind_L(i), Ind_S_LCR(i) ),DEL(i),DEL_s(i),lt);


    vec weight_LIR_S=weight_s(sigma,g(tau(i), Ind_L(i), Ind_S_LIR(i) ),g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ),g(SSD(i), Ind_L(i), Ind_S_LIR(i) ),
                              g(b_l(i), Ind_L(i), Ind_S_LIR(i) ),g(nu_l(i),Ind_L(i), Ind_S_LIR(i) ),
                              g(nu_l_s_i, Ind_L(i), Ind_S_LIR(i) ),DEL(i),DEL_s(i),lt);

    vec weight_RCR_S=weight_s(sigma,g(tau(i), Ind_R(i), Ind_S_RCR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ),g(SSD(i), Ind_R(i), Ind_S_RCR(i) ),
                              g(b_l(i), Ind_R(i), Ind_S_RCR(i) ),h(nu_l_p(i), Ind_S_RCR(i) ),
                              g(nu_l_s_i,Ind_R(i), Ind_S_RCR(i) ),DEL(i),DEL_s(i),lt);

    vec weight_RIR_S=weight_s(sigma, g(tau(i), Ind_R(i), Ind_S_RIR(i) ),g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ),g(SSD(i), Ind_R(i), Ind_S_RIR(i) ),
                              g(b_r(i), Ind_R(i), Ind_S_RIR(i) ),g(nu_r(i), Ind_R(i), Ind_S_RIR(i) ),
                              g(nu_r_s_i,Ind_R(i), Ind_S_RIR(i) ),DEL(i),DEL_s(i),lt);



    vec weight_FS_LCR=weight_stop(g(tau_s(i), Ind_L(i), Ind_S_LCR(i) ), sigma,stop_param.col(i));

    vec weight_FS_LIR=weight_stop(g(tau_s(i), Ind_L(i), Ind_S_LIR(i) ), sigma,stop_param.col(i));

    vec weight_FS_RCR=weight_stop(g(tau_s(i), Ind_R(i), Ind_S_RCR(i) ), sigma,stop_param.col(i));

    vec weight_FS_RIR=weight_stop(g(tau_s(i), Ind_R(i), Ind_S_RIR(i) ), sigma,stop_param.col(i));




    field<vec> likelihood_integral =  integral_likelihood(sigma,SSD(i),DEL(i),DEL_s(i),
                                                          stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                          b_l(i),nu_l(i),nu_l_s_i,
                                                          b_r(i),nu_r_p(i),nu_r_s_i,
                                                          nu_r(i),nu_l_p(i),
                                                          Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                          lower_bound(i),upper_bound,lt);


    double Inhib_likelihood = sum(likelihood_integral(0)) + sum(likelihood_integral(1));



    field<vec> b_stop_Stim= integrate_b_stop(likelihood_integral,sigma,SSD(i),DEL(i),DEL_s(i),
                                             stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                             b_l(i),b_r(i),nu_l(i),nu_r(i),nu_l_s_i,nu_r_s_i,
                                             nu_r_p(i),nu_l_p(i),Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                             lower_bound(i),upper_bound,lt);




    field<vec> nu_stop_Stim = integrate_nu_stop(likelihood_integral, sigma,SSD(i),DEL(i),DEL_s(i),
                                                stop_param.col(i),penal_param.col(i),prob_param.col(i),
                                                b_l(i),b_r(i),
                                                nu_l(i),nu_r(i),nu_l_s_i,nu_r_s_i,
                                                nu_r_p(i),nu_l_p(i),
                                                nu_l_squared(i),nu_r_squared(i),
                                                Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                lower_bound(i),upper_bound,lt);



    field<vec> lk_FS=update_lk_FS4(tau(i),tau_s(i), sigma,SSD(i),
                                   b_l(i),b_r(i),nu_l(i),nu_r(i),
                                   nu_r_p(i),nu_l_p(i),nu_l_s_i,nu_r_s_i,
                                   Ind_L(i),Ind_R(i),Ind_LCR(i), Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                                   Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                                   stop_param.col(i),prob_param.col(i),DEL(i),DEL_s(i),lt);



    vec lk_LCR_FS=lk_FS(0);
    vec lk_LIR_FS=lk_FS(1);

    vec lk_RCR_FS=lk_FS(2);
    vec lk_RIR_FS=lk_FS(3);


    vec diff_X_LCR=lk_FS(4);
    vec diff_Y_LCR=lk_FS(5);

    vec diff_X_LIR=lk_FS(6);
    vec diff_Y_LIR=lk_FS(7);

    vec diff_X_RCR=lk_FS(8);
    vec diff_Y_RCR=lk_FS(9);

    vec diff_X_RIR=lk_FS(10);
    vec diff_Y_RIR=lk_FS(11);



    //penalty parameter update steps

    vec p_stop(size(stop_param.col(i)),fill::randn);

    vec p_prob(size(prob_param.col(i)),fill::randn);



    // Rcpp::Rcout<<"p"<<p<<endl;


    // double U_old = -log_lhood_s(Inhib_likelihood,lk_LCR_FS, lk_LIR_FS,lk_RCR_FS, lk_RIR_FS);



    double U_old=-(log_lhood(Inhib_likelihood,
                             sigma, tau(i), tau_s(i), SSD(i),Ind_GF(i),
                             b_l(i), b_r(i), nu_l(i), nu_r(i), nu_l_p(i), nu_r_p(i), nu_l_s_i, nu_r_s_i,
                             Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                             Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                             lk_LCR_FS, lk_LIR_FS,lk_RCR_FS,lk_RIR_FS,
                             prob_param.col(i),T_Go(i),k(i))-(T_Total(i)*sum_prob));



    double prior_main_prob=log_gama_prior(prob_param.col(i),prob_hyp(0),prob_hyp(1),prob_hyp(2));


    double prior_main_stop=log_normal_prior(stop_param.col(i),mu_b_stop,sigma_b_stop,mu_nu_stop,sigma_nu_stop);

    U_old-=(prior_main_prob+prior_main_stop);


    double H_old = U_old+(dot(p_prob,p_prob)/2)+(dot(p_stop,p_stop)/2);


    vec v_old_stop = -grad_stop_param(tau(i),tau_s(i),sigma,SSD(i),DEL(i),DEL_s(i),
                                      nu_l(i),nu_r(i),b_l(i),b_r(i),
                                      nu_l_p(i),nu_r_p(i),nu_l_s_i,nu_r_s_i,
                                      weight_LCR_S, weight_LIR_S,weight_RCR_S,weight_RIR_S,
                                      weight_FS_LCR,weight_FS_LIR,weight_FS_RCR,weight_FS_RIR,
                                      Ind_L(i),Ind_R(i),

                                      Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                                      nu_stop_Stim,b_stop_Stim,
                                      stop_param.col(i),penal_param.col(i),
                                      mu_nu_stop,sigma_nu_stop,mu_b_stop,sigma_b_stop,
                                      diff_Y_LCR,diff_Y_LIR,diff_Y_RCR,diff_Y_RIR,lt);




    vec v_old_prob=-grad_prob_param(tau_s(i),sigma,likelihood_integral,
                                    stop_param.col(i),prob_param.col(i),sum_prob,
                                    prob_hyp,
                                    T_Go(i),k(i),T_Total(i),
                                    diff_X_LCR,diff_Y_LCR,diff_X_LIR,diff_Y_LIR,
                                    diff_X_RCR,diff_Y_RCR,diff_X_RIR,diff_Y_RIR);


    vec stop_param_new=stop_param.col(i);

    vec v_new_stop = v_old_stop;


    vec prob_param_new=prob_param.col(i);

    vec v_new_prob = v_old_prob;


    double delta = randu(distr_param(range(0), range(1)));
    // int pois_draw = (unsigned) R::rpois(L);
    // int nstep = GSL_MAX_INT(1, pois_draw);
    // nstep = GSL_MIN_INT(nstep, leapmax);

    int nstep=std::clamp((int) R::rpois(L),1,leapmax);


    leap_frog_stop_prob_param(tau(i), tau_s(i), sigma,delta_prime.col(i),SSD(i),DEL(i),DEL_s(i),
                              b_l(i), b_r(i),
                              nu_l(i), nu_r(i), nu_l_p(i), nu_r_p(i),
                              nu_l_squared(i), nu_r_squared(i),

                              Ind_L(i), Ind_R(i),
                              Ind_LCR(i), Ind_LIR(i), Ind_RCR(i), Ind_RIR(i),

                              Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RCR(i), Ind_S_RIR(i),
                              Ind_I_L(i), Ind_I_R(i),
                              stop_param_new, penal_param.col(i),prob_param_new,
                              mu_b_stop, sigma_b_stop, mu_nu_stop, sigma_nu_stop,
                              prob_hyp,T_Go(i),k(i),T_Total(i),

                              v_new_stop,v_new_prob,nstep,delta,p_stop,p_prob,

                              lower_bound(i), upper_bound, lt);


    vec nu_l_s_i_new = nu_l(i) - penalty_fun_s(stop_param_new(1), penal_param.col(i));
    vec nu_r_s_i_new = nu_r(i) - penalty_fun_s(stop_param_new(1), penal_param.col(i));


    field<vec> likelihood_integral_new = integral_likelihood(sigma,SSD(i),DEL(i),DEL_s(i),stop_param_new,
                                                             penal_param.col(i),prob_param_new,
                                                             b_l(i),nu_l(i),nu_l_s_i_new,
                                                             b_r(i),nu_r_p(i),nu_r_s_i_new,
                                                             nu_r(i),nu_l_p(i),
                                                             Ind_L(i),Ind_R(i),Ind_I_L(i),Ind_I_R(i),
                                                             lower_bound(i),upper_bound,lt);


    double Inhib_likelihood_new = sum(likelihood_integral_new(0)) + sum(likelihood_integral_new(1));


    field<vec> lk_FS_new= update_lk_FS3(tau(i),tau_s(i),sigma,SSD(i),
                                        b_l(i),b_r(i),nu_l(i),nu_r(i),nu_r_p(i),nu_l_p(i),
                                        nu_l_s_i_new,nu_r_s_i_new,
                                        Ind_L(i),Ind_R(i),Ind_LCR(i),Ind_LIR(i),Ind_RCR(i),Ind_RIR(i),
                                        Ind_S_LCR(i),Ind_S_LIR(i),Ind_S_RCR(i),Ind_S_RIR(i),
                                        stop_param_new,prob_param_new,DEL(i),DEL_s(i),lt);



    vec lk_LCR_FS_new=lk_FS_new(0);
    vec lk_LIR_FS_new=lk_FS_new(1);

    vec lk_RCR_FS_new=lk_FS_new(2);
    vec lk_RIR_FS_new=lk_FS_new(3);


    double sum_prob_new = mlpack::AccuLog(prob_param_new);

    double U_new=- (log_lhood(Inhib_likelihood_new,
                              sigma, tau(i), tau_s(i), SSD(i),Ind_GF(i),
                              b_l(i), b_r(i), nu_l(i), nu_r(i), nu_l_p(i), nu_r_p(i), nu_l_s_i_new, nu_r_s_i_new,
                              Ind_L(i), Ind_R(i), Ind_LCR(i), Ind_LIR(i), Ind_RIR(i), Ind_RCR(i),
                              Ind_S_LCR(i), Ind_S_LIR(i), Ind_S_RIR(i), Ind_S_RCR(i),
                              lk_LCR_FS_new, lk_LIR_FS_new,lk_RCR_FS_new,lk_RIR_FS_new,
                              prob_param_new,T_Go(i),k(i))-(T_Total(i)*sum_prob_new));


    double prior_main_prob_new=log_gama_prior(prob_param_new,prob_hyp(0),prob_hyp(1),prob_hyp(2));


    double prior_main_stop_new=log_normal_prior(stop_param_new,mu_b_stop,sigma_b_stop,mu_nu_stop,sigma_nu_stop);


    U_new-=(prior_main_prob_new+prior_main_stop_new);


    double H_new=U_new+(dot(p_stop,p_stop)/2)+(dot(p_prob,p_prob)/2);


    /*double H_new=U_new+((as_scalar(dot(p,(M_inv*p))))/2);*/

    // Rcpp::Rcout<<"new-penalty="<<-U_new<<" old-penalty="<<-U_old<< endl;


    if (log(randu()) < -(H_new - H_old)) {
#pragma omp critical
{
  stop_param.col(i) = stop_param_new;
  prob_param.col(i) = prob_param_new;
  ++acceptance_stop_prob[i];
}
    }
  }

  omp_set_num_threads(6);
}




/////////////////////////////////////////////////////////////////////////////////////// rand /////////////////////////////////////////////////////////////////////////////////////////////


//Leapfrog algorithm

void leap_frog_rand(const mat &gama_sq,
                    vec &rand_param,const double kappa,const double nu_d,const vec &lam,
                    mat &v_old,const double L,const double delta,mat &p,const unsigned N){


  // Rcpp::Rcout<<"ABB"<< endl;

  rand_param+=delta*(p-(delta/2)*v_old);

  vec S = log_spec_dens(nu_d, rand_param(1), lam, 1);
  vec grad_S = grad_lam(nu_d,rand_param(1),lam,1);


  // New gradient


  vec v_new(size(rand_param), fill::zeros);

  v_new = -grad_rand(N,gama_sq,rand_param,S,grad_S);

  vec gradient_rand_prior=grad_rand_prior(rand_param,kappa);

  v_new -=gradient_rand_prior;



  //momentum
  p-=(delta/2)*(v_old+v_new);
  v_old=v_new;

}



//Update rand

void update_rand(const mat &gama_sq,
                 vec &rand_param,const double kappa,const double nu_d,const vec &lam,
                 const vec &range,const double L,const int leapmax, unsigned &acceptance_rand,const unsigned N){


  // Rcpp::Rcout<<"update_rand"<< endl;

  vec S = log_spec_dens(nu_d, rand_param(1), lam, 1);
  vec grad_S = grad_lam(nu_d,rand_param(1),lam,1);

  unsigned m=gama_sq.n_rows;



  //Rand_param update steps

  vec p(size(rand_param),fill::randn);


  double gama_S=dot((sum(gama_sq,1)),exp(-S));



  double   U_old = -((((m*N)*rand_param(0))/2)+((N*accu(S))/2)
                       +((exp(-rand_param(0))*gama_S)/2));


  double prior_main=rand_prior(rand_param,kappa);

  U_old+=prior_main;

  U_old *= -1;


  double H_old = U_old+(dot(p,p)/2);


  vec v_old(size(rand_param), fill::zeros);

  v_old = -grad_rand(N,gama_sq,rand_param,S,grad_S);

  vec gradient_rand_prior=grad_rand_prior(rand_param,kappa);

  v_old -=gradient_rand_prior;


  vec rand_param_new=rand_param;

  vec v_new=v_old;

  double delta= randu(distr_param(range(0),range(1)));

  // int pois_draw=(unsigned) R::rpois(L);
  // int nstep =  GSL_MAX_INT(1, pois_draw);
  // nstep=GSL_MIN_INT(nstep,leapmax);

  int nstep=std::clamp((int) R::rpois(L),1,leapmax);


  leap_frog_rand(gama_sq,rand_param_new,kappa,nu_d,lam,
                 v_new,nstep,delta,p,N);


  vec S_new = log_spec_dens(nu_d, rand_param_new(1), lam, 1);


  double gama_S_new=dot((sum(gama_sq,1)),exp(-S_new));


  double  U_new=-((((m*N)*rand_param_new(0))/2)+((N*accu(S_new))/2)
                    +((exp(-rand_param_new(0))*gama_S_new)/2));


  double prior_main_new=rand_prior(rand_param_new,kappa);

  U_new+=prior_main_new;

  U_new *= -1;


  double H_new = U_new + (dot(p, p) / 2);


  if(log(randu())< -(H_new-H_old) ){
    rand_param=rand_param_new;
    ++acceptance_rand;

  }

}



////////////////////////////////////////////////////////////////////// Making Field from List ////////////////////////////////////////////////////////////////////////////////////////////

// Making Field <mat> from a List


field<mat> make_field_from_list1(const List &x) {

  /*Rcpp::Rcout<<"ABL"<< endl;*/

  unsigned n=x.length();
  field<mat> ll(n);

  for(unsigned i=0;i<n;++i){
    ll(i)=as<mat>(x[i]);
    // (ll(i)).print("(ll(i)):");
  }

  return ll;
}


// Making Field <vec> from a List


field<vec> make_field_from_list2(const List &x) {

  //Rcpp::Rcout<<"ABM"<< endl;

  unsigned n=x.length();
  field<vec> ll(n);

  for(unsigned i=0;i<n;++i){
    ll(i)=as<vec>(x[i]);
    // (ll(i)).print("(ll(i)):");
  }

  return ll;
}



// Making Field<uvec> from a List


field<uvec> make_field_from_list(const List &x) {

  //Rcpp::Rcout<<"ABN"<< endl;

  unsigned n=x.length();
  field<uvec> ll(n);

  for(unsigned i=0;i<n;++i){
    ll(i)=as<uvec>(x[i]);
    // (ll(i)).print("(ll(i)):");
  }

  return ll;
}

// Making Field <Field<uvec>> from a List



field<field<uvec>> field_of_field(const List &x) {

  //Rcpp::Rcout<<"ABO"<< endl;
  unsigned n = x.length();
  field<field<uvec>> result(n);

  for(unsigned i = 0; i < n; ++i){
    result(i) = make_field_from_list(x[i]);
  }

  return result;
}



////////////////////////////////////////////////////////////////////////// Implementing HMC //////////////////////////////////////////////////////////////////////////////////////////

// Final HMC

// [[Rcpp::export]]
List hmc(const List Time_stamp,const List Go_RT,const List Go_RT_S,const double SSD_min,const arma::vec& U,const List Ind_G,const List Stop_S_D,
           const double sigma,arma::mat delta_param,arma::mat gama,
           arma::mat beta,arma::mat stop_param,arma::mat prob_param,const List Indicator,
           const List mean_priors_main,const List var_priors_main,arma::mat penal_param,
           const arma::vec prior_penal_stop,const arma::vec T_Go,const arma::vec k,const arma::vec T_Total,
           const double a,const double b,const arma::vec prob_hyp,
           double eta_b_l,double eta_b_r,double eta_nu_l,double eta_nu_r,
           List gama_Ind,List beta_Ind,
           const double nu_d,const arma::vec lam,const List Phi_mat,
           arma::vec rand_param_g_l,arma::vec rand_param_g_r,arma::vec rand_param_b_l,arma::vec rand_param_b_r,
           const double kappa,const List ranges,
           const double L,const int leapmax,
           const double nhmc,int thin,unsigned nparall,
           const List l_bound,const double upper_bound,
           const bool update_gama_beta,const bool update_penalty,const bool update_stop_prob,const bool update_rand_eff,const bool update_delta,
           const bool lt = true){

  Rcpp::Rcout<<"ABP"<< endl;

// --- Checkpoint controls ---
  const unsigned checkpoint_every = 2000U;                    // save every 2000 iterations
  const std::string checkpoint_prefix = "checkpoint_iter_";   // filename prefix
  Function saveRDS("saveRDS");                                // R's saveRDS for checkpointing


  // Making Field from List

  field <mat> t=make_field_from_list1(Time_stamp);
  field <vec> tau=make_field_from_list2(Go_RT);
  field <vec> tau_stop=make_field_from_list2(Go_RT_S);
  field <vec> SSD=make_field_from_list2(Stop_S_D);
  field <vec> Ind_GF=make_field_from_list2(Ind_G);

  field <vec> lower_bound_init=make_field_from_list2(l_bound);

  field <mat> gama_I=make_field_from_list1(gama_Ind);
  field <mat> beta_I=make_field_from_list1(beta_Ind);
  field <mat> Phi=make_field_from_list1(Phi_mat);


  field <field<uvec>> Ind=field_of_field(Indicator);


  field <uvec> Ind_L=Ind(0);
  field <uvec> Ind_R=Ind(1);
  field <uvec> Ind_LCR=Ind(2);
  field <uvec> Ind_LIR=Ind(3);
  field <uvec> Ind_RCR=Ind(4);
  field <uvec> Ind_RIR=Ind(5);

  field <uvec> Ind_S_LCR=Ind(6);
  field <uvec> Ind_S_LIR=Ind(7);
  field <uvec> Ind_S_RCR=Ind(8);
  field <uvec> Ind_S_RIR=Ind(9);
  field <uvec> Ind_I_L=Ind(10);
  field <uvec> Ind_I_R=Ind(11);


  field <vec> mean_main=make_field_from_list2(mean_priors_main);

  vec mu_gl=mean_main(0);
  vec mu_gr=mean_main(1);
  vec mu_bl=mean_main(2);
  vec mu_br=mean_main(3);



  field <mat> var_main=make_field_from_list1(var_priors_main);

  mat sigma_gl_inv=var_main(0);
  mat sigma_gr_inv=var_main(1);
  mat sigma_bl_inv=var_main(2);
  mat sigma_br_inv=var_main(3);


  double mu_lambda_prime=prior_penal_stop(0);
  double sigma_lambda_prime=prior_penal_stop(1);
  double mu_alpha_prime=prior_penal_stop(2);
  double sigma_alpha_prime=prior_penal_stop(3);
  double mu_b_stop=prior_penal_stop(4);
  double sigma_b_stop=prior_penal_stop(5);
  double mu_nu_stop=prior_penal_stop(6);
  double sigma_nu_stop=prior_penal_stop(7);



  field <vec> range=make_field_from_list2(ranges);
  vec range_g_b=range(0);
  vec range_p=range(1);
  vec range_d=range(2);
  vec range_stop_prob=range(3);
  vec range_rand_effect=range(4);

  vec range_rand_g_l=range(5);
  vec range_rand_g_r=range(6);
  vec range_rand_b_l=range(7);
  vec range_rand_b_r=range(8);


  double P=t(0).n_cols;
  unsigned N=t.size();

  unsigned m=Phi(0).n_cols;



  // Constant vectors

  vec sigma_mu_gama_l=sigma_gl_inv*mu_gl;

  vec sigma_mu_gama_r=sigma_gr_inv*mu_gr;

  vec sigma_mu_beta_l=sigma_bl_inv*mu_bl;

  vec sigma_mu_beta_r=sigma_br_inv*mu_br;


  // Storage

  unsigned num_thinned_samples = (nhmc/thin);
  mat theta1_L(P,num_thinned_samples, fill::zeros);
  mat theta1_R(P,num_thinned_samples, fill::zeros);
  mat theta2_L(P, num_thinned_samples, fill::zeros);
  mat theta2_R(P, num_thinned_samples, fill::zeros);
  cube theta3(2, N, num_thinned_samples, fill::zeros);
  mat theta4(N,num_thinned_samples, fill::zeros);
  cube theta5(2, N, num_thinned_samples, fill::zeros);

  cube theta5_L(m,N,num_thinned_samples, fill::zeros);
  cube theta5_R(m,N,num_thinned_samples, fill::zeros);

  cube theta6_L(m,N,num_thinned_samples, fill::zeros);
  cube theta6_R(m,N,num_thinned_samples, fill::zeros);
  cube theta7(3, N, num_thinned_samples, fill::zeros);
  mat theta8(2, num_thinned_samples, fill::zeros);
  mat theta9(2, num_thinned_samples, fill::zeros);
  mat theta10(2, num_thinned_samples, fill::zeros);
  mat theta11(2, num_thinned_samples, fill::zeros);

  cube theta12(2, N, num_thinned_samples, fill::zeros);


  unsigned acceptance_main_eff=0;

  vec acceptance_p(N, fill::zeros);

  vec acceptance_d(N, fill::zeros);

  vec acceptance_stop_prob(N, fill::zeros);

  vec acceptance_rand_effect(N, fill::zeros);

  unsigned acceptance_rand_g_l=0;

  unsigned acceptance_rand_g_r=0;

  unsigned acceptance_rand_b_l=0;

  unsigned acceptance_rand_b_r=0;


  // Setting up initial gama, beta and delta_prime related matrix or fields


  vec S_gama_l = log_spec_dens(nu_d, rand_param_g_l(1), lam, 1);
  vec S_gama_r = log_spec_dens(nu_d, rand_param_g_r(1), lam, 1);

  vec S_beta_l = log_spec_dens(nu_d, rand_param_b_l(1), lam, 1);
  vec S_beta_r = log_spec_dens(nu_d, rand_param_b_r(1), lam, 1);



  field<field<vec>> gama_ess=update_gama_param_ess4(t,gama);


  field <vec> b_l_main=gama_ess(0);
  field <vec> b_r_main=gama_ess(1);


  field<field<vec>> gama_ess2=update_gama_param_ess3(b_l_main,b_r_main,Phi,gama_I);

  field <vec> b_l=gama_ess2(0);
  field <vec> b_r=gama_ess2(1);

  field <vec> Rand_gama_l=gama_ess2(2);
  field <vec> Rand_gama_r=gama_ess2(3);



  field<field<vec>> beta_ess=update_beta_param_ess5(t,beta);

  field <vec> nu_l_main=beta_ess(0);
  field <vec> nu_r_main=beta_ess(1);


  field <field<vec>> beta_ess2=update_beta_param_ess4(nu_l_main,nu_r_main,
                                                      Phi,beta_I);


  field <vec> nu_l = beta_ess2(0);
  field <vec> nu_r = beta_ess2(1);
  field <vec> nu_l_squared = beta_ess2(2);
  field <vec> nu_r_squared = beta_ess2(3);
  field <vec> Rand_beta_l = beta_ess2(4);
  field <vec> Rand_beta_r = beta_ess2(5);


  field<field<vec>> beta_ess3=update_beta_param_ess3(nu_l,nu_r,nu_l_squared,nu_r_squared,
                                                     Ind_L,Ind_R,penal_param);

  field <vec> nu_l_p=beta_ess3(0);
  field <vec> nu_r_p=beta_ess3(1);


  field <field<vec>> stop_param_ess= update_stop_param_ess2(nu_l,nu_r,stop_param,penal_param);



  field <vec> nu_l_s=stop_param_ess(0);
  field <vec> nu_r_s=stop_param_ess(1);



  vec DEL_s=U%expit(delta_param.row(0).t());

  vec DEL=(SSD_min+DEL_s)%expit(delta_param.row(1).t());



  field<field<vec>> delta_prime_ess= update_delta_prime_param_ess(tau,tau_stop,lower_bound_init,DEL,DEL_s);

  field <vec> tau_prime=delta_prime_ess(0);
  field <vec> tau_s=delta_prime_ess(1);
  field <vec> lower_bound=delta_prime_ess(2);



  // Rcpp::Rcout<<"loop started"<<endl;

  for (unsigned i = 0; i < nhmc; ++i){


    mat R_gl=sigma_gl_inv;
    mat Z_gl=R_gl*(gama.col(0)-mu_gl);
    double D_gl=eta_b_l*dot((gama.col(0)-mu_gl),Z_gl);


    // mat R_gr=chol(sigma_gr_inv);
    mat R_gr=sigma_gr_inv;
    mat Z_gr=R_gr*(gama.col(1)-mu_gr);
    double D_gr=eta_b_r*dot((gama.col(1)-mu_gr),Z_gr);


    double shape_b_l=a+((P-1)/2);
    double rate_b_l=b+(D_gl/2);

    double shape_b_r=a+((P-1)/2);
    double rate_b_r=b+(D_gr/2);

    double eta_b_l = (randg(arma::distr_param(shape_b_l, (1/rate_b_l))));
    double eta_b_r = (randg(arma::distr_param(shape_b_r, (1/rate_b_r))));


    // mat R_bl=chol(sigma_bl_inv);
    mat R_bl=sigma_bl_inv;
    mat Z_bl=R_bl*(beta.col(0)-mu_bl);
    double D_bl=eta_nu_l*dot((beta.col(0)-mu_bl),Z_bl);


    // mat R_br=chol(sigma_br_inv);
    mat R_br=sigma_br_inv;
    mat Z_br=R_br*(beta.col(1)-mu_br);
    double D_br=eta_nu_r*dot((beta.col(1)-mu_br),Z_br);

    double shape_nu_l=a+((P-1)/2);
    double rate_nu_l=b+(D_bl/2);

    double shape_nu_r=a+((P-1)/2);
    double rate_nu_r=b+(D_br/2);

    double eta_nu_l = (randg(arma::distr_param(shape_nu_l, (1/rate_nu_l))));
    double eta_nu_r = (randg(arma::distr_param(shape_nu_r, (1/rate_nu_r))));



    // Updating delta_prime Parameter

    if(update_delta==1){

      update_delta_param(tau,tau_stop,sigma,SSD,SSD_min,U,Ind_GF,delta_param,gama,beta,penal_param,stop_param,
                         b_l,b_r,nu_l,nu_r,nu_l_p,nu_r_p,nu_l_s,nu_r_s,
                         Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                         Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,Ind_I_L,Ind_I_R,
                         prob_param,T_Go,k,range_d,L,leapmax,acceptance_d,
                         nparall,lower_bound_init,upper_bound,lt);



      // Updating delta_prime parameter related matrix or fields

      vec DEL_s=U%expit(delta_param.row(0).t());

      vec DEL=(SSD_min+DEL_s)%expit(delta_param.row(1).t());


      // Rcpp::Rcout<<"DEL_s"<<DEL_s<<endl;

      // Rcpp::Rcout<<"DEL"<<DEL<<endl;


      // Extract values from update_delta_prime_param_ess
      field<field<vec>> delta_prime_ess = update_delta_prime_param_ess(tau, tau_stop, lower_bound_init, DEL, DEL_s);

      field<vec> tau_prime = delta_prime_ess(0);
      field<vec> tau_s = delta_prime_ess(1);
      field<vec> lower_bound = delta_prime_ess(2);


      uword total_neg = 0;
      for (uword i = 0; i < tau_prime.n_elem; ++i) {
        total_neg += accu(tau_prime(i) < 0);
      }

      if (total_neg > 0) {
        Rcpp::Rcout << "Total negative values in tau_prime: " << total_neg << std::endl;
      }



      // Rcpp::Rcout<<"delta_updated"<<endl;

    }



    // Updating Gama

    if(update_gama_beta==1){

    update_main_effect(t,Rand_gama_l,Rand_gama_r,Rand_beta_l,Rand_beta_r,
                       tau_prime,tau_s,sigma,SSD,DEL,DEL_s,Ind_GF,
                       gama,beta,delta_param,penal_param,stop_param,prob_param,
                       Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                       Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                       Ind_I_L,Ind_I_R,
                       eta_b_l,eta_b_r,eta_nu_l,eta_nu_r,
                       mu_gl,sigma_gl_inv,mu_gr,sigma_gr_inv,
                       mu_bl,sigma_bl_inv,mu_br,sigma_br_inv,
                       gama_I,beta_I,T_Go,k,
                       range_g_b,L,leapmax,acceptance_main_eff,
                       nparall,lower_bound,upper_bound,lt);



    // Updating gama related matrix or fields

    field<field<vec>> gama_ess=update_gama_param_ess1(t,Rand_gama_l,Rand_gama_r,gama);


    field<vec> b_l=gama_ess(0);
    field<vec> b_r=gama_ess(1);

    field<vec> b_l_main=gama_ess(2);
    field<vec> b_r_main=gama_ess(3);


    //Updating beta related matrix or fields


    field<field<vec>>  beta_ess=update_beta_param_ess1(t,Rand_beta_l,Rand_beta_r,beta);

    field<vec> nu_l=beta_ess(0);
    field<vec> nu_r=beta_ess(1);
    field<vec> nu_l_squared=beta_ess(2);
    field<vec> nu_r_squared=beta_ess(3);
    field<vec> nu_l_main=beta_ess(4);
    field<vec> nu_r_main=beta_ess(5);



    field<field<vec>> beta_ess2=update_beta_param_ess3(nu_l,nu_r,nu_l_squared,nu_r_squared,
                                                       Ind_L,Ind_R,penal_param);



    field<vec>nu_l_p=beta_ess2(0);
    field<vec>nu_r_p=beta_ess2(1);


    field <field<vec>> stop_param_ess3= update_stop_param_ess2(nu_l,nu_r,stop_param,penal_param);


    nu_l_s=stop_param_ess3(0);
    nu_r_s=stop_param_ess3(1);


    // Rcpp::Rcout<<"Main_effect"<<endl;
    }



    // Updating Penalty Parameters

    if(update_penalty==1){

      update_penal_param(tau_prime,tau_s,sigma,SSD,DEL,DEL_s,Ind_GF,
                         delta_param,gama,beta,penal_param,stop_param,
                         b_l,b_r,
                         nu_l,nu_r,nu_l_squared,nu_r_squared,
                         Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                         Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                         Ind_I_L,Ind_I_R,
                         mu_lambda_prime,sigma_lambda_prime,mu_alpha_prime,sigma_alpha_prime,
                         prob_param,T_Go,k,
                         range_p,L,leapmax,acceptance_p,nparall,
                         lower_bound,upper_bound,lt);



      // Updating penalty parameter related matrix or fields


      field<field<vec>> penal_ess=update_beta_param_ess3(nu_l,nu_r,nu_l_squared,nu_r_squared,
                                                         Ind_L,Ind_R,penal_param);


      nu_l_p=penal_ess(0);
      nu_r_p=penal_ess(1);



      field<field<vec>> penal_ess2=update_stop_param_ess2(nu_l,nu_r,stop_param,penal_param);

      nu_l_s=penal_ess2(0);
      nu_r_s=penal_ess2(1);;


    // Rcpp::Rcout<<"Panalty_updated"<<endl;

    }



    //Updating stop_param

    if(update_stop_prob==1){

      update_stop_prob_param(tau_prime,tau_s,sigma,SSD,DEL,DEL_s,Ind_GF,
                             delta_param,gama,beta,penal_param,stop_param,prob_param,
                             b_l,b_r,nu_l,nu_r,nu_l_p,nu_r_p,
                             nu_l_squared,nu_r_squared,
                             Ind_L,Ind_R,
                             Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,

                             Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                             Ind_I_L,Ind_I_R,
                             mu_b_stop,sigma_b_stop,mu_nu_stop,sigma_nu_stop,
                             prob_hyp,T_Go,k,T_Total,

                             range_stop_prob, L,leapmax,acceptance_stop_prob,nparall,
                             lower_bound,upper_bound,lt);




      // Updating stop parameter related vectors

      field <field<vec>> stop_param_ess5= update_stop_param_ess2(nu_l,nu_r,stop_param,penal_param);


      nu_l_s=stop_param_ess5(0);
      nu_r_s=stop_param_ess5(1);
    }




    // Rcpp::Rcout<<"Stop_updated"<<endl;


    //Updating Random effect coefficeints



    if(update_rand_eff==1){

    update_rand_effect(Phi,b_l_main,b_r_main,nu_l_main,nu_r_main,
                       S_gama_l,S_gama_r,S_beta_l,S_beta_r,
                       rand_param_g_l,rand_param_g_r,
                       rand_param_b_l,rand_param_b_r,
                       tau_prime,tau_s,sigma,SSD,DEL,DEL_s,Ind_GF,
                       delta_param,penal_param,stop_param,prob_param,
                       Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
                       Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,
                       Ind_I_L,Ind_I_R,
                       gama_I,beta_I,
                       T_Go,k,range_rand_effect,L,leapmax,acceptance_rand_effect,
                       nparall,lower_bound,upper_bound,lt);



    // Updating Random effect coefficient related matrix or fields

    field<field<vec>> gama_ess2=update_gama_param_ess3(b_l_main,b_r_main,Phi,gama_I);


    b_l=gama_ess2(0);
    b_r=gama_ess2(1);

    Rand_gama_l=gama_ess2(2);
    Rand_gama_r=gama_ess2(3);


    field<mat> sq_ess= update_sq(gama_I);

    mat gama_sq_l=sq_ess(0);
    mat gama_sq_r=sq_ess(1);


    field<field<vec>>  beta_ess4=update_beta_param_ess4(nu_l_main,nu_r_main,Phi,beta_I);


    nu_l=beta_ess4(0);
    nu_r=beta_ess4(1);
    nu_l_squared=beta_ess4(2);
    nu_r_squared=beta_ess4(3);

    Rand_beta_l=beta_ess4(4);
    Rand_beta_r=beta_ess4(5);



    field<field<vec>> beta_ess5=update_beta_param_ess3(nu_l,nu_r,nu_l_squared,nu_r_squared,
                                                       Ind_L,Ind_R,penal_param);

    nu_l_p=beta_ess5(0);
    nu_r_p=beta_ess5(1);


    field <field<vec>> stop_param_ess6= update_stop_param_ess2(nu_l,nu_r,stop_param,penal_param);


    nu_l_s=stop_param_ess6(0);
    nu_r_s=stop_param_ess6(1);

    field<mat> sq_ess2= update_sq(beta_I);

    mat beta_sq_l=sq_ess2(0);
    mat beta_sq_r=sq_ess2(1);

    // Rcpp::Rcout<<"Random Effect Updated"<<endl;

    //Updating rand

    update_rand(gama_sq_l,rand_param_g_l,kappa,nu_d,lam,
                range_rand_g_l, L,leapmax,acceptance_rand_g_l,N);


    update_rand(gama_sq_r,rand_param_g_r,kappa,nu_d,lam,
                range_rand_g_r, L,leapmax,acceptance_rand_g_r,N);


    update_rand(beta_sq_l,rand_param_b_l,kappa,nu_d,lam,
                range_rand_b_l, L,leapmax,acceptance_rand_b_l,N);

    update_rand(beta_sq_r,rand_param_b_r,kappa,nu_d,lam,
                range_rand_b_r, L,leapmax,acceptance_rand_b_r,N);



    vec S_gama_l = log_spec_dens(nu_d, rand_param_g_l(1), lam, 1);
    vec S_gama_r = log_spec_dens(nu_d, rand_param_g_r(1), lam, 1);

    vec S_beta_l = log_spec_dens(nu_d, rand_param_b_l(1), lam, 1);
    vec S_beta_r = log_spec_dens(nu_d, rand_param_b_r(1), lam, 1);

    // Rcpp::Rcout<<"GP Updated"<<endl;


    }


    // Thining

    if (i % thin == 0) {

      theta1_L.col(i / thin) = gama.col(0);
      theta1_R.col(i / thin) = gama.col(1);
      theta2_L.col(i / thin) = beta.col(0);
      theta2_R.col(i / thin) = beta.col(1);
      theta3.slice(i / thin) = penal_param;



      // theta4.col (i / thin)  = delta_prime;
      theta5.slice(i / thin) = stop_param;




      for (unsigned j = 0; j < N; ++j) {
        theta5_L.slice(i/thin).col(j) = gama_I(j).col(0);
        theta5_R.slice(i/thin).col(j) = gama_I(j).col(1);
        theta6_L.slice(i/thin).col(j) = beta_I(j).col(0);
        theta6_R.slice(i/thin).col(j) = beta_I(j).col(1);
      }


      theta7.slice(i / thin) = prob_param;

      theta8.col(i / thin) =rand_param_g_l;
      theta9.col(i / thin) =rand_param_g_r;
      theta10.col(i / thin) =rand_param_b_l;
      theta11.col(i / thin) =rand_param_b_r;

      theta12.slice(i / thin) = delta_param;



      Rcpp::Rcout << "Iteration: " << i
                  << " Accept_prob_main_effect = " << ((double)acceptance_main_eff) / ((double)i)
                  << " Med_accept_prob_p = " << median(acceptance_p) / ((double)i)
                  << " Med_accept_prob_d = " << median(acceptance_d) / ((double)i)
                  << " Med_accept_stop_prob = " << median(acceptance_stop_prob) / ((double)i)
                  << " Med_acceptance_rand_effect = " << median(acceptance_rand_effect) / ((double)i)
                  << " Accept_rand_g_l = " << ((double)acceptance_rand_g_l)  / ((double)i)
                  << " Accept_rand_g_r = " << ((double)acceptance_rand_g_r)  / ((double)i)
                  << " Accept_rand_b_l = " << ((double)acceptance_rand_b_l)  / ((double)i)
                  << " Accept_rand_b_r = " << ((double)acceptance_rand_b_r)  / ((double)i)

      << std::endl;


    }

      // --- Checkpoint save every 2000 iterations ---
    if ((i + 1) % checkpoint_every == 0) {
      const unsigned tcol = (i / thin);             // last filled thin index (only valid when i%thin==0 or earlier cols)
      const unsigned have_cols = std::min(tcol + 1, (unsigned)num_thinned_samples);

      // Slice only the filled portions to keep files smaller
      mat  theta1_L_part = have_cols ? theta1_L.cols(0, have_cols - 1) : mat(P, 0);
      mat  theta1_R_part = have_cols ? theta1_R.cols(0, have_cols - 1) : mat(P, 0);
      mat  theta2_L_part = have_cols ? theta2_L.cols(0, have_cols - 1) : mat(P, 0);
      mat  theta2_R_part = have_cols ? theta2_R.cols(0, have_cols - 1) : mat(P, 0);
      cube theta3_part   = have_cols ? theta3.slices(0, have_cols - 1) : cube(2, N, 0);
      cube theta5_part   = have_cols ? theta5.slices(0, have_cols - 1) : cube(2, N, 0);
      cube theta5_L_part = have_cols ? theta5_L.slices(0, have_cols - 1) : cube(m, N, 0);
      cube theta5_R_part = have_cols ? theta5_R.slices(0, have_cols - 1) : cube(m, N, 0);
      cube theta6_L_part = have_cols ? theta6_L.slices(0, have_cols - 1) : cube(m, N, 0);
      cube theta6_R_part = have_cols ? theta6_R.slices(0, have_cols - 1) : cube(m, N, 0);
      cube theta7_part   = have_cols ? theta7.slices(0, have_cols - 1) : cube(3, N, 0);
      mat  theta8_part   = have_cols ? theta8.cols(0, have_cols - 1)   : mat(2, 0);
      mat  theta9_part   = have_cols ? theta9.cols(0, have_cols - 1)   : mat(2, 0);
      mat  theta10_part  = have_cols ? theta10.cols(0, have_cols - 1)  : mat(2, 0);
      mat  theta11_part  = have_cols ? theta11.cols(0, have_cols - 1)  : mat(2, 0);
      cube theta12_part  = have_cols ? theta12.slices(0, have_cols - 1): cube(2, N, 0);

      List out_partial;
      out_partial["Gamma_L"]=theta1_L_part;
      out_partial["Gamma_R"]=theta1_R_part;
      out_partial["Beta_L"]=theta2_L_part;
      out_partial["Beta_R"]=theta2_R_part;
      out_partial["Penalty"]=exp(theta3_part);
      out_partial["Shift"]=theta12_part;
      out_partial["Stop"]=exp(theta5_part);
      out_partial["Gama_I_L"]=theta5_L_part;
      out_partial["Gama_I_R"]=theta5_R_part;
      out_partial["Beta_I_L"]=theta6_L_part;
      out_partial["Beta_I_R"]=theta6_R_part;
      out_partial["Prob_param"]=exp(theta7_part);
      out_partial["rand_param_g_l"]=theta8_part;
      out_partial["rand_param_g_r"]=theta9_part;
      out_partial["rand_param_b_l"]=theta10_part;
      out_partial["rand_param_b_r"]=theta11_part;

      // acceptance rates so far (divide by iterations done, i+1)
      double denom = (double)(i + 1);
      out_partial["Accept_main_effect"]=acceptance_main_eff/denom;
      out_partial["Accept_penal"]=acceptance_p/denom;
      out_partial["Accept_Shift"]=acceptance_d/denom;
      out_partial["Accept_Stop_Prob"]=acceptance_stop_prob/denom;
      out_partial["Accept_rand_effect"]=acceptance_rand_effect/denom;
      out_partial["Accept_rand_g_l"]=acceptance_rand_g_l/denom;
      out_partial["Accept_rand_g_r"]=acceptance_rand_g_r/denom;
      out_partial["Accept_rand_b_l"]=acceptance_rand_b_l/denom;
      out_partial["Accept_rand_b_r"]=acceptance_rand_b_r/denom;

      // Filename
      std::string fname = checkpoint_prefix + std::to_string(i + 1) + ".rds";

      // saveRDS(object, file)
      saveRDS(out_partial, fname);

      Rcpp::Rcout << "Checkpoint saved at iteration " << (i + 1)
                  << " -> " << fname << std::endl;
    }


  }

  // Storing Results

  List out;
  out["Gamma_L"]=theta1_L;
  out["Gamma_R"]=theta1_R;
  out["Beta_L"]=theta2_L;
  out["Beta_R"]=theta2_R;
  out["Penalty"]=exp(theta3);
  out["Shift"]=theta12;
  out["Stop"]=exp(theta5);
  out["Gama_I_L"]=theta5_L;
  out["Gama_I_R"]=theta5_R;
  out["Beta_I_L"]=theta6_L;
  out["Beta_I_R"]=theta6_R;
  out["Prob_param"]=exp(theta7);
  out["rand_param_g_l"]=theta8;
  out["rand_param_g_r"]=theta9;
  out["rand_param_b_l"]=theta10;
  out["rand_param_b_r"]=theta11;
  out["Accept_main_effect"]=acceptance_main_eff/nhmc;
  out["Accept_penal"]=acceptance_p/nhmc;
  out["Accept_Shift"]=acceptance_d/nhmc;
  out["Accept_Stop_Prob"]=acceptance_stop_prob/nhmc;
  out["Accept_rand_effect"]=acceptance_rand_effect/nhmc;

  out["Accept_rand_g_l"]=acceptance_rand_g_l/nhmc;
  out["Accept_rand_g_r"]=acceptance_rand_g_r/nhmc;
  out["Accept_rand_b_l"]=acceptance_rand_b_l/nhmc;
  out["Accept_rand_b_r"]=acceptance_rand_b_r/nhmc;


  return(out);

}




















