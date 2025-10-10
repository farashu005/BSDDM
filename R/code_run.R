#' Run the Bayesian SDDM model with HMC
#'
#' This function prepares a dataset, initializes priors, loads necessary C++ functions,
#' and runs a Bayesian Semi-parametric Drift Diffusion Model (SSDDM) using Hamiltonian Monte Carlo (HMC).
#'
#' @param save_location Directory to save all model outputs.
#' @param dat Input data frame with stop-signal and go-trial information.
#' @param cluster_label Cluster label used to subset the data.
#' @param Anxiety_status Group label used to subset the data.
#' @param sub_type Subtype category used to subset the data.
#' @param sampling Logical. Whether to sample subjects from the subset or use all subjects.
#' @param sample_size Number of subjects to sample from the subset (if sampling = TRUE, otherwise all subjects are used).
#' @param run Integer: 1 for first run, 2 for second run, 3 for both.
#' @param file_name File path to save the final model output as an RDS file.
#' @param prep_data_file_name File path to save the preprocessed data as an RDS file.
#' @param range_main_eff Step size range for main effects during leapfrog updates in HMC.
#' @param range_p Step size range for penalty parameters during leapfrog updates in HMC.
#' @param range_d Step size range for delta parameters during leapfrog updates in HMC.
#' @param range_stop_prob Step size range for stop and probability parameters during leapfrog updates in HMC.Prior range for  parameters.
#' @param range_rand_eff Step size range for random effect parameters during leapfrog updates in HMC.Prior
#' @param range_rand_g_l Step size range for left boundary Gaussian Process random effects during leapfrog updates.
#' @param range_rand_g_r Step size range for right boundary Gaussian Process random effects during leapfrog updates.
#' @param range_rand_b_l Step size range for left drift Gaussian Process random effects during leapfrog updates.
#' @param range_rand_b_r Step size range for right boundary Gaussian Process random effects during leapfrog updates.
##' @param penal_param.int Initial value penalty parameters.
#' @param delta_prime.int Initial value of delta prime parameters.
#' @param stop_param.int Initial value of stop parameters.
#' @param prob_param.int Initial value of probability parameters.
#' @param rand_param_g_l.int Initial value of Gaussian Proecess Parameters for left Boundary random effects.
#' @param rand_param_g_r.int Initial value of Gaussian Proecess Parameters for right Boundary random effects.
#' @param rand_param_b_l.int Initial value of Gaussian Proecess Parameters for left Drift random effects.
#' @param rand_param_b_r.int Initial value of Gaussian Process Parameters for Right Drift random effects.
#' @param nu_d Degrees of freedom for the random effect prior.
#' @param kappa Shape parameter for the inverse gamma prior.
#' @param a Hyperparameter a for the prior.
#' @param b Hyperparameter b for the prior.
#' @param CA_threshold Minimum choice accuracy required to retain a subject.
#' @param nknots Number of spline knots.
#' @param m Number of basis functions.
#' @param scale Rescaling factor applied to predictors.
#' @param SSD_min Minimum Stop Signal Delay to include.
#' @param upper_bound Upper bound for integration.
#' @param L Number of leapfrog steps in HMC.
#' @param leapmax Maximum leap size in HMC.
#' @param thin Thinning interval for storing HMC samples.
#' @param nparall Number of parallel threads to use.
#' @param nhmc Total number of HMC iterations.
#' @param intercept Logical. Whether to include intercept terms in the spline basis.
#'
#' @return Saves two RDS files: one containing the preprocessed dataset, and another containing model outputs.
#' @export


code_run<-function(save_location,dat,cluster_label,Anxiety_status,sub_type,sampling,sample_size,run,file_name,

                   prep_data_file_name,

                   range_main_eff,range_p,range_d,range_stop_prob,range_rand_eff,
                   range_rand_g_l,range_rand_g_r,range_rand_b_l,range_rand_b_r,

                   penal_param.int,delta_prime.int,stop_param.int,prob_param.int,
                   rand_param_g_l.int,rand_param_g_r.int,rand_param_b_l.int,rand_param_b_r.int,


                   nu_d, kappa, a, b,
                   CA_threshold, nknots, m, scale,
                   SSD_min, upper_bound,
                   L, leapmax, thin, nparall, nhmc,intercept){





  setwd(save_location)

  if (intercept == TRUE) {
    source(system.file("scripts", "Data_Preparation_with_intercept.R", package = "BSDDM"), echo = TRUE)
  } else {
    source(system.file("scripts", "Data_Preparation_without_intercept.R", package = "BSDDM"), echo = TRUE)
  }






  subset<-dat[dat$cluster_label==cluster_label & dat$Anxiety_status==Anxiety_status & dat$sub_type==sub_type,]


  f <- function(x) mean(x, na.rm = TRUE)

  CA_sub<-na.omit(aggregate(subset$sst_choiceacc,list(subset$subject),f))

  colnames(CA_sub)<-c("Sub","PER")



  ## Selecting Subjects
  # set.seed(1235)

  if (sampling == TRUE) {

    set.seed(5151)
    sub <- sample(CA_sub$Sub[CA_sub$PER >= CA_threshold], sample_size)

  } else {

    sub <- CA_sub$Sub[CA_sub$PER >= CA_threshold]

  }


  ## Creating Dataset with the subjects
  M_I1_Non_Anx_dat<-data_prep(subset,sub=sub,nknots,m,run,scale)


  saveRDS(M_I1_Non_Anx_dat, file = prep_data_file_name)


  ## Basic Variables
  T=M_I1_Non_Anx_dat$T
  Time=M_I1_Non_Anx_dat$Time ## Time-subject wise
  X=M_I1_Non_Anx_dat$X ##. Time-combined
  X1=M_I1_Non_Anx_dat$X1 ## Spline
  Go_RT=M_I1_Non_Anx_dat$tau
  Go_RT_S=M_I1_Non_Anx_dat$tau_s
  SP_SIM_Dat=M_I1_Non_Anx_dat$sp.x1
  Ind_G=M_I1_Non_Anx_dat$Ind_G
  Indicator=M_I1_Non_Anx_dat$Ind
  T_Go=M_I1_Non_Anx_dat$T_Go
  T_Total=M_I1_Non_Anx_dat$T_Total
  FT=M_I1_Non_Anx_dat$FT
  PGF=M_I1_Non_Anx_dat$PGF


  Stop_S_D=M_I1_Non_Anx_dat$SSD

  Phi_mat=M_I1_Non_Anx_dat$Phi

  lambda<-M_I1_Non_Anx_dat$lambda

  

  lower_bound<-M_I1_Non_Anx_dat$l_lim


  ## Initialization

  P=ncol(X1[[1]])
  N=length(Go_RT)

  #U<-M_I1_Non_Anx_dat$U

  U<-c(rep(150,N))
  #m=length(lambda)


  mu_bl<-mu_br<-mu_gl<-mu_gr<-c(rep(0,P))
  mean_priors_main<-list(mu_gl,mu_gr,mu_bl,mu_br);

  # Function to create the Du matrix using diff
  create_Du_diff <- function(K) {
    # Initialize an empty matrix for Du
    Du <- matrix(0, nrow = K-2, ncol = K)

    # Fill the Du matrix using second-order differences
    for (i in 1:(K-2)) {
      Du[i, ] <- diff(diag(K), differences = 2)[i, ]
    }

    return(Du)
  }




  # Function to compute Pu
  compute_Pu_diff <- function(K) {
    Du <- create_Du_diff(K)
    Pu <- t(Du) %*% Du
    return(Pu)
  }

  sigma_bl_inv<-sigma_br_inv<-sigma_gl_inv<-sigma_gr_inv<-compute_Pu_diff(P)

  var_priors_main<-list(sigma_gl_inv,sigma_gr_inv,sigma_bl_inv,sigma_br_inv);


  gama_Ind<-replicate(N, matrix(0, nrow = m, ncol = 2), simplify = FALSE)
  beta_Ind<-replicate(N, matrix(0, nrow = m, ncol = 2), simplify = FALSE)



  mu_lambda_prime<-0
  sigma_lambda_prime<-1
  mu_alpha_prime<-0
  sigma_alpha_prime<-1

  mu_b_stop<-0
  sigma_b_stop<-1
  mu_nu_stop<-0
  sigma_nu_stop<-1

  prior_penal_stop<-c(mu_lambda_prime,sigma_lambda_prime,mu_alpha_prime,sigma_alpha_prime,
                      mu_b_stop,sigma_b_stop,mu_nu_stop,sigma_nu_stop)





  sigma<-1



  #prob_hyp<-c(1,1,1)

  # order must be: (PGF, P00, PTF)
##prob_hyp <- 40 * c(PGF = 0.02, P00 = 0.95, PTF = 0.03)   # a = (0.8, 38, 1.2)
# w_{ik} ~ Gamma(shape = a[k], rate = 1);  P_i = w_i / sum(w_i)
prob_hyp <- 4 * c(PGF=0.02, P00=0.95, PTF=0.03)




  ranges<-list(range_main_eff,range_p,range_d,range_stop_prob,range_rand_eff,range_rand_g_l,range_rand_g_r,range_rand_b_l,range_rand_b_r)


  data<-M_I1_Non_Anx_dat$dat


  mean <- tapply(data$sst_primaryrt[data$sst_expcon == "GoTrial" | data$sst_inhibitacc == 0],
                 data$sst_primaryresp[data$sst_expcon == "GoTrial" | data$sst_inhibitacc == 0],
                 mean, na.rm = TRUE)

  mean <- mean[mean != 0]

  var <- tapply(data$sst_go_rt[data$sst_expcon == "GoTrial" | data$sst_inhibitacc == 0],
                data$sst_primaryresp[data$sst_expcon == "GoTrial" | data$sst_inhibitacc == 0],
                var, na.rm = TRUE)

  var <- var[var != 0]



  nu_hat<-sqrt(mean/var)

  b_hat<-nu_hat*mean


  SP2=Reduce(rbind,X1)

  if (intercept == TRUE) {
    source(system.file("scripts", "Method of Moment Method_with_intercept.R", package = "BSDDM"))
  } else {
    source(system.file("scripts", "Method_of_Moment_Method_without_intercept.R", package = "BSDDM"))
  }


  gama_l<-init_Gen_b(b_hat[1],SP2)
  gama_r<-init_Gen_b(b_hat[2],SP2)
  gama.int<-matrix(c(gama_l,gama_r),ncol=2,byrow = F)


  beta_l<-init_Gen_nu(nu_hat[1],SP2)
  beta_r<-init_Gen_nu(nu_hat[2],SP2)
  beta.int<-matrix(c(beta_l,beta_r),ncol=2,byrow = F)




  shape=a+((P-1)/2)




  rate_b_l=b+(t(gama.int[,1])%*%sigma_gl_inv%*%gama.int[,1])/2
  rate_b_r=b+(t(gama.int[,2])%*%sigma_gr_inv%*%gama.int[,2])/2

  rate_nu_l=b+(t(beta.int[,1])%*%sigma_bl_inv%*%beta.int[,1])/2
  rate_nu_r=b+(t(beta.int[,2])%*%sigma_br_inv%*%beta.int[,2])/2

  eta_b_l=rgamma(1,shape,1/rate_b_l)
  eta_b_r=rgamma(1,shape,1/rate_b_r)


  eta_nu_l=rgamma(1,shape,1/rate_nu_l)
  eta_nu_r=rgamma(1,shape,1/rate_nu_r)





  ## Initializing

set.seed (1451)
penal1<-log(runif(N, min = penal_param.int[1], max = penal_param.int[2]))
penal2<-log(runif(N, penal_param.int[1], max = penal_param.int[2]))

penal_param<-as.matrix(rbind(penal1,penal2))

delta_param<-matrix(c(rep(delta_prime.int,2*N)),nrow=2,ncol=N)


b_stop=rep(stop_param.int[1],N)


nu_stop=rep(stop_param.int[2],N)

stop_param<-as.matrix(rbind(b_stop,nu_stop))


GF.int<-rep(prob_param.int[1],N)

TF1.int<-rep(prob_param.int[2],N)

TF2.int<-rep(prob_param.int[3],N)

prob_param<-as.matrix(rbind(GF.int,TF1.int,TF2.int),ncol=N,nrow=3)




  library(RcppArmadillo)

  print(Sys.time()); message("Starting model...")

  # Record the start time
  start_time <- Sys.time()

  model<-hmc(X1,Go_RT=Go_RT,Go_RT_S=Go_RT_S,SSD_min=SSD_min, U=U,Ind_G=Ind_G,Stop_S_D=Stop_S_D,
             sigma,delta_param,gama=gama.int,beta=beta.int,
             stop_param,prob_param,
             Indicator=Indicator,
             mean_priors_main,var_priors_main,
             penal_param, prior_penal_stop,
             T_Go,FT,T_Total,
             a,b,prob_hyp,
             eta_b_l,eta_b_r,eta_nu_l,eta_nu_r,
             gama_Ind,beta_Ind,
             nu_d,lambda,Phi_mat,
             rand_param_g_l.int,rand_param_g_r.int,rand_param_b_l.int,rand_param_b_r.int,
             kappa,
             ranges, L,leapmax,nhmc,thin,nparall,
             lower_bound,upper_bound,
             update_gama_beta=1,update_penalty=1,update_stop_prob=1,
             update_rand_eff=1,update_delta=1,lt=1)




  # Record the end time
  end_time <- Sys.time()

  # Calculate the duration
  (runtime <- end_time - start_time)


  cat("Runtime duration:", runtime, "\n")


  saveRDS(model, file = file_name)


}
