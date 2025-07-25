
#### Plots

## model=HMC output
## sp.x1=Fitted spline
## X=timepoints
## Burn=Number of burn
## N=Number of individuals
## CI=Confidence Interval (Yes/No)
Plot_results <- function(model,sp.x1,Time,X1=X1,Time_point, Burn, N,Phi_mat,U,SSD_min, CI,convergence_plots) {

  library(ggplot2)
  library(gridExtra)
  library(dplyr)
  library(ggpubr)




  ##################################################### Convergence Diagnostics ############################################################


  if (convergence_plots==1){
    par(mfrow = c(2, 2))

    mat_gama_L<-matplot(t(model$Gamma_L[-1,-(1:Burn)]),type="l",main="matplot of Gama_L",xlab="t",ylab="Gama_L")

    mat_gama_R<-matplot(t(model$Gamma_R[-1,-(1:Burn)]),type="l",main="matplot of Gama_R",xlab="t",ylab="Gama_R")

    mat_beta_L<-matplot(t(model$Beta_L[-1,-(1:Burn)]),type="l",main="matplot of Beta_L",xlab="t",ylab="Beta_L")

    mat_beta_R<-matplot(t(model$Beta_R[-1,-(1:Burn)]),type="l",main="matplot of Beta_R",xlab="t",ylab="Beta_R")

    # Reset plotting parameters
    par(mfrow = c(1, 1))

    # Save the combined plot
    matplot_param <- recordPlot()




    # Number of penalty plots per figure
    PlotsPerFigure <- 4
    num_figures <- ceiling(N / PlotsPerFigure)  # Calculate total number of figures

    Penalty_plots <- list()  # Initialize a list to save the plots

    for (fig in 1:num_figures) {
      # Set up the plotting area for the current figure
      par(mfrow = c(2, 2))  # 2 rows and 2 columns (adjust as needed)

      # Plot the corresponding penalty plots for this figure
      for (i in ((fig - 1) * PlotsPerFigure + 1):min(fig * PlotsPerFigure, N)) {
        current_data <- t(model$Penalty[, i, -(1:Burn)])

        # Define custom labels for the parameters
        param_labels <- c("lambda", "alpha")

        # Plot the data
        matplot(current_data, type = "l", col = 1:ncol(current_data),
                xlab = "Iteration", ylab = paste("Individual", i),
                main = paste("Penalty Parameter - Individual", i))

        # Add legend with custom parameter names
        legend("topright", legend = param_labels[1:ncol(current_data)],
               col = 1:ncol(current_data), lty = 1, cex = 0.8, bty = "n")
      }

      # Save the current figure as a recorded plot
      Penalty_plots[[fig]] <- recordPlot()
    }


    ## Stop Plots

    ## Stop Plots

    Stop_plots <- list()
    for (fig in 1:num_figures) {
      # Set up the plotting area for the current figure
      par(mfrow = c(2, 2))  # 2 rows and 2 columns (adjust as needed)

      # Plot the corresponding stop plots for this figure
      for (i in ((fig - 1) * PlotsPerFigure + 1):min(fig * PlotsPerFigure, N)) {
        current_data <- t(model$Stop[, i, -(1:Burn)])

        # Define custom labels for the legend
        param_labels <- c("b_stop", "nu_stop")

        # Plot the data
        matplot(current_data, type = "l", col = 1:ncol(current_data),
                xlab = "Iteration", ylab = paste("Individual", i),
                main = paste("Stop Parameter - Individual", i))

        # Add legend with custom parameter names
        legend("topright", legend = param_labels[1:ncol(current_data)],
               col = 1:ncol(current_data), lty = 1, cex = 0.8, bty = "n")
      }

      # Save the current figure as a recorded plot
      Stop_plots[[fig]] <- recordPlot()
    }



    expit<-function(x){
      return (exp(x) / (1.0 +exp(x)));
    }

    DEL_s=U*expit(t(model$Shift[1, , -(1:Burn)]));

    DEL=(SSD_min+DEL_s)*expit(t(model$Shift[2, , -(1:Burn)]));


    # Initialize list to store plots
    Delta_plots <- list()

    for (fig in 1:num_figures) {
      # Set up the plotting area for the current figure
      par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))  # Adjust margins if needed

      # Plot the corresponding delta plots for this figure
      for (i in ((fig - 1) * PlotsPerFigure + 1):min(fig * PlotsPerFigure, N)) {
        current_data <- cbind(DEL_s[, i], DEL[, i])

        # Define custom labels for the legend
        param_labels <- c("DEL_s", "DEL")

        # Plot the data
        matplot(
          current_data, type = "l", col = 1:ncol(current_data),
          xlab = "Iteration", ylab = paste("Individual", i),
          main = paste("Delta Parameters - Individual", i)
        )

        # Add legend with custom parameter names
        legend("topright", legend = param_labels[1:ncol(current_data)],
               col = 1:ncol(current_data), lty = 1, cex = 0.8, bty = "n")
      }

      # Save the current figure as a recorded plot
      Delta_plots[[fig]] <- recordPlot()
    }





    ## Matplots of Random Effect Parameters

    ## Gama_I_L_plots

    # Create an empty list to store individual plots
    Gama_I_L_plots <- list()


    for (fig in 1:num_figures) {
      # Set up the plotting area for the current figure
      par(mfrow = c(2, 2))  # 2 rows and 2 columns (adjust as needed)

      # Plot the corresponding penalty plots for this figure
      for (i in ((fig - 1) * PlotsPerFigure + 1):min(fig * PlotsPerFigure, N)) {
        current_data <- t(model$Gama_I_L[, i, -(1:Burn)])
        matplot(current_data, type = "l", col = 1:ncol(current_data),
                xlab = "Iteration", ylab = paste("Individual", i),
                main = paste("Gama_I_L - Individual", i))
      }

      # Save the current figure as a recorded plot
      Gama_I_L_plots[[fig]] <- recordPlot()
    }





    ## Gama_I_plots

    # Create an empty list to store individual plots
    Gama_I_R_plots <- list()


    for (fig in 1:num_figures) {
      # Set up the plotting area for the current figure
      par(mfrow = c(2, 2))  # 2 rows and 2 columns (adjust as needed)

      # Plot the corresponding penalty plots for this figure
      for (i in ((fig - 1) * PlotsPerFigure + 1):min(fig * PlotsPerFigure, N)) {
        current_data <- t(model$Gama_I_R[, i, -(1:Burn)])
        matplot(current_data, type = "l", col = 1:ncol(current_data),
                xlab = "Iteration", ylab = paste("Individual", i),
                main = paste("Gama_I_R - Individual", i))
      }

      # Save the current figure as a recorded plot
      Gama_I_R_plots[[fig]] <- recordPlot()
    }




    ## Beta_I_L_plots

    # Create an empty list to store individual plots
    Beta_I_L_plots <- list()


    for (fig in 1:num_figures) {
      # Set up the plotting area for the current figure
      par(mfrow = c(2, 2))  # 2 rows and 2 columns (adjust as needed)

      # Plot the corresponding penalty plots for this figure
      for (i in ((fig - 1) * PlotsPerFigure + 1):min(fig * PlotsPerFigure, N)) {
        current_data <- t(model$Beta_I_L[, i, -(1:Burn)])
        matplot(current_data, type = "l", col = 1:ncol(current_data),
                xlab = "Iteration", ylab = paste("Individual", i),
                main = paste("Beta_I_L - Individual", i))
      }

      # Save the current figure as a recorded plot
      Beta_I_L_plots[[fig]] <- recordPlot()
    }




    ## Beta_I_R_plots

    # Create an empty list to store individual plots
    Beta_I_R_plots <- list()


    for (fig in 1:num_figures) {
      # Set up the plotting area for the current figure
      par(mfrow = c(2, 2))  # 2 rows and 2 columns (adjust as needed)

      # Plot the corresponding penalty plots for this figure
      for (i in ((fig - 1) * PlotsPerFigure + 1):min(fig * PlotsPerFigure, N)) {
        current_data <- t(model$Beta_I_R[, i, -(1:Burn)])
        matplot(current_data, type = "l", col = 1:ncol(current_data),
                xlab = "Iteration", ylab = paste("Individual", i),
                main = paste("Beta_I_R - Individual", i))
      }

      # Save the current figure as a recorded plot
      Beta_I_R_plots[[fig]] <- recordPlot()
    }


    ## Matplots of Gaussian Process Parameters

    par(mfrow = c(2, 2))

    mat_alpha_gama_L<-matplot(exp(model$rand_param_g_l[1,-(1:Burn)]),type="l",main="matplot of alpha_gama_L",xlab="t",ylab="alpha_L")
    mat_alpha_gama_R<-matplot(exp(model$rand_param_g_r[1,-(1:Burn)]),type="l",main="matplot of alpha_gama_R",xlab="t",ylab="alpha_R")

    mat_l_gama_L<-matplot(exp(model$rand_param_g_l[2,-(1:Burn)]),type="l",main="matplot of l_gama_L",xlab="t",ylab="l_L")
    mat_l_gama_R<-matplot(exp(model$rand_param_g_r[2,-(1:Burn)]),type="l",main="matplot of l_gama_R",xlab="t",ylab="l_R")



    # Reset plotting parameters
    par(mfrow = c(1, 1))

    mat_plot_param_rand<- recordPlot()


    par(mfrow = c(2, 2))


    mat_alpha_beta_L<-matplot(exp(model$rand_param_b_l[1,-(1:Burn)]),type="l",main="matplot of alpha_beta_L",xlab="t",ylab="alpha_L")
    mat_alpha_beta_R<-matplot(exp(model$rand_param_b_r[1,-(1:Burn)]),type="l",main="matplot of alpha_beta_R",xlab="t",ylab="alpha_R")




    mat_l_beta_L<-matplot(exp(model$rand_param_b_l[2,-(1:Burn)]),type="l",main="matplot of l_beta_L",xlab="t",ylab="l_L")
    mat_l_beta_R<-matplot(exp(model$rand_param_b_r[2,-(1:Burn)]),type="l",main="matplot of l_beta_R",xlab="t",ylab="l_R")



    ## Matplot of Probability Parameters

    ## PGF, P00 and P10
    nhmc<-dim(model$Penalty)[3]

    PGF <- matrix(0, nrow = N, ncol = nhmc)

    P_00 <- matrix(0, nrow = N, ncol = nhmc)

    P_10 <- matrix(0, nrow = N, ncol = nhmc)


    for (i in 1:nhmc) {
      PGF[, i] <- model$Prob_param[1, , i] /
        (model$Prob_param[1, , i] + model$Prob_param[2, , i] + model$Prob_param[3, , i])
      P_00[, i] <- model$Prob_param[2, , i] /
        (model$Prob_param[1, , i] + model$Prob_param[2, , i] + model$Prob_param[3, , i])

      P_10[, i] <- model$Prob_param[3, , i] /
        (model$Prob_param[1, , i] + model$Prob_param[2, , i] + model$Prob_param[3, , i])


    }




    # Create an empty list to store the recorded plots
    mat_plot_GF_TF <- list()

    # Loop over the figures
    for (fig in 1:num_figures) {
      # Set up a 2x2 plotting layout (adjust layout as needed)
      par(mfrow = c(2, 2))

      # Generate plots for individuals corresponding to this figure
      for (i in ((fig - 1) * PlotsPerFigure + 1):min(fig * PlotsPerFigure, N)) {

        # Plot all three probability values in one plot
        matplot(cbind(PGF[i, -(1:Burn)], P_00[i, -(1:Burn)], P_10[i, -(1:Burn)]),
                type = "l", lty = 1, col = c("blue", "red", "green"),
                main = paste("Probabilities - Individual", i),
                xlab = "t", ylab = "Probability", ylim = c(0, 1))

        # Add legend
        legend("topright", legend = c("PGF", "P_00", "P_10"),
               col = c("blue", "red", "green"), lty = 1, cex = 0.8)
      }

      # Record the plot and store it in the list
      mat_plot_GF_TF[[fig]] <- recordPlot()
    }

    # Restore the default plotting parameters
    par(mfrow = c(1, 1))


    convergence_diagnostic_plots <- list(
      matplot_param = matplot_param,
      Penalty_plots = Penalty_plots,
      Stop_plots = Stop_plots,
      Gama_I_L_plots = Gama_I_L_plots,
      Gama_I_R_plots = Gama_I_R_plots,
      Beta_I_L_plots = Beta_I_L_plots,
      Beta_I_R_plots = Beta_I_R_plots,
      matplot_param_rand = mat_plot_param_rand,
      mat_plot_GF_TF = mat_plot_GF_TF
    )

  }

  #################################################### Main Effect ##########################################################


  ## Matplots of main effect Parameters



  Gamma_L<-model$Gamma_L[,-(1:Burn)]
  Gamma_R<-model$Gamma_R[,-(1:Burn)]

  Beta_L<-model$Beta_L[,-(1:Burn)]
  Beta_R<-model$Beta_R[,-(1:Burn)]


  ### Boundary Parameter

  ## b_L
  ## Fitted Value

  b_l_temp<-exp(sp.x1%*%Gamma_L)

  Fitted_b_L<-apply(b_l_temp,1,mean)


  ## Curve fitting
  Curve_b_L<-data.frame(Time,y=Fitted_b_L,Curve="Left Stimulus")


  ## Credible interval
  Cred_b_L<-apply(b_l_temp,1,quantile,probs=c(0.025,0.975))

  Cred_b_L<-data.frame((t(Cred_b_L)))
  colnames(Cred_b_L)<-c("lower_CI","Upper_CI")

  b_L_frame<-cbind(Curve_b_L,Cred_b_L)



  ## b_R
  ## Fitted Value

  b_r_temp<-exp(sp.x1%*%Gamma_R)

  Fitted_b_R<-apply(b_r_temp,1,mean)

  ## Curve fitting
  Curve_b_R<-data.frame(Time,y=Fitted_b_R,Curve="Right Stimulus")


  ## Credible interval
  Cred_b_R<-apply(b_r_temp,1,quantile,probs=c(0.025,0.975))

  Cred_b_R<-data.frame((t(Cred_b_R)))
  colnames(Cred_b_R)<-c("lower_CI","Upper_CI")

  b_R_frame<-cbind(Curve_b_R,Cred_b_R)


  # Combine the data frames for left and right stimulus
  combined_b <- bind_rows(b_L_frame,b_R_frame)


  ## Drift Parameter

  ## nu_L
  ## Fitted Value

  nu_l_temp<-exp(sp.x1%*%Beta_L)

  Fitted_nu_L<-apply(nu_l_temp,1,mean)


  ## Curve fitting
  Curve_nu_L<-data.frame(Time,y= Fitted_nu_L,Curve="Left Stimulus")


  ## Credible interval
  Cred_nu_L<-apply(nu_l_temp,1,quantile,probs=c(0.025,0.975))

  Cred_nu_L<-data.frame((t(Cred_nu_L)))
  colnames(Cred_nu_L)<-c("lower_CI","Upper_CI")

  nu_L_frame<-cbind(Curve_nu_L,Cred_nu_L)



  ## Beta_R
  ## Fitted Value


  nu_r_temp<-exp(sp.x1%*%Beta_R)

  Fitted_nu_R<-apply(nu_r_temp,1,mean)

  ## Curve fitting
  Curve_nu_R<-data.frame(Time,y=Fitted_nu_R,Curve="Right Stimulus")



  ## Credible interval
  Cred_nu_R<-apply(nu_r_temp,1,quantile,probs=c(0.025,0.975))

  Cred_nu_R<-data.frame((t(Cred_nu_R)))
  colnames(Cred_nu_R)<-c("lower_CI","Upper_CI")

  nu_R_frame<-cbind(Curve_nu_R,Cred_nu_R)



  # Combine the data frames for left and right stimulus
  combined_nu  <- bind_rows(nu_L_frame,nu_R_frame)



  ## Mean Response time

  ## Left Stimulus

  mean_resp_temp_left_main_eff<-exp(sp.x1%*%(Gamma_L-Beta_L))

  Fitted_mean_resp_left_main_eff<-apply(mean_resp_temp_left_main_eff,1,mean)


  ## Curve fitting
  Curve_mean_resp_left_main_eff<-data.frame(Time,y=Fitted_mean_resp_left_main_eff,Curve="Left Stimulus")


  ## Credible interval
  Cred_mean_resp_left_main_eff<-apply(mean_resp_temp_left_main_eff,1,quantile,probs=c(0.025,0.975))

  Cred_mean_resp_left_main_eff<-data.frame((t(Cred_mean_resp_left_main_eff)))
  colnames(Cred_mean_resp_left_main_eff)<-c("lower_CI","Upper_CI")

  mean_resp_left_frame_main_eff<-cbind(Curve_mean_resp_left_main_eff,Cred_mean_resp_left_main_eff)




  ## Right Stimulus

  mean_resp_temp_right_main_eff<-exp(sp.x1%*%(Gamma_R-Beta_R))
  Fitted_mean_resp_right_main_eff<-apply(mean_resp_temp_right_main_eff,1,mean)


  ## Curve fitting
  Curve_mean_resp_right_main_eff<-data.frame(Time,y=Fitted_mean_resp_right_main_eff,Curve="Right Stimulus")


  ## Credible interval
  Cred_mean_resp_right_main_eff<-apply(mean_resp_temp_right_main_eff,1,quantile,probs=c(0.025,0.975))

  Cred_mean_resp_right_main_eff<-data.frame((t(Cred_mean_resp_right_main_eff)))
  colnames(Cred_mean_resp_right_main_eff)<-c("lower_CI","Upper_CI")

  mean_resp_right_frame_main_eff<-cbind(Curve_mean_resp_right_main_eff,Cred_mean_resp_right_main_eff)


  # Combine the data frames for left and right stimulus
  combined_mean_response_main_eff  <- bind_rows(mean_resp_left_frame_main_eff, mean_resp_right_frame_main_eff)



  # GGplot of Boundary parameter( with CI ribbon)
  if(CI=="YES"){

    Boundary_main_eff<-ggplot( combined_b, aes(Time/1000, y, color = Curve)) +
      geom_ribbon(aes(ymin = lower_CI, ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed")+
      geom_line() +
      xlab("t(second)") + ylab("Estimated Boundary") +
      labs(title = "Plot of Boundary Parameter for left and Right Stimulus")
  }

  else{

    # Create a single ggplot with both curves
    Boundary_main_eff<-ggplot( combined_b, aes(Time/1000, y, color = Curve)) +
      geom_line() +
      xlab("t(second)") + ylab("Estimated Boundary") +
      labs(title = "Plot of Boundary Parameter for left and Right Stimulus")
  }

  # # GGplot for Drift Parameter (withCI ribbon)
  if(CI=="YES"){

    Drift_main_eff<-ggplot(combined_nu , aes(Time/1000, y, color = Curve)) +
      geom_ribbon(aes(ymin = lower_CI,ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed") +
      geom_line() +
      xlab("t(second)") + ylab("Estimated Drift") +
      labs(title = "Plot of Drift Parameter for left and Right Stimulus")
  }


  else{
    # GGplot for Drift Parameter (without CI ribbon)

    Drift_main_eff<-ggplot(combined_nu , aes(Time/1000, y, color = Curve)) +
      # geom_ribbon(aes(ymin = lower_CI, ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed") +
      geom_line() +
      xlab("t(second)") + ylab("Estimated Drift") +
      labs(title = "Plot of Drift Parameter for left and Right Stimulus")
  }




  if(CI=="YES"){

    # Mean Response (with CI Ribbon)

    Mean_Response_main_eff<-ggplot(combined_mean_response_main_eff , aes(Time/1000, y, color = Curve)) +
      geom_ribbon(aes(ymin = lower_CI, ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed") +
      geom_line() +
      xlab("t(second)") + ylab("Estimated Mean Response (main effect)") +
      labs(title = "Plot of Mean Response (main effect) for left and Right Stimulus")
  }

  else{
    # Mean Response (without CI Ribbon)

    Mean_Response_main_eff<-ggplot(combined_mean_response_main_eff , aes(Time/1000, y, color = Curve)) +
      geom_line() +
      xlab("t(second)") + ylab("Estimated Mean Response (main effect)") +
      labs(title = "Plot of Mean Response (main effect) for left and Right Stimulus")

  }

  Parameter_graph_main_effect<-ggpubr::ggarrange(Boundary_main_eff,Drift_main_eff,Mean_Response_main_eff,nrow=3)




  #################################################### Random Effect ##########################################################

  ## Random Effect Curves

  # Rand<-model$Random_Effect_Parameter[,-(1:Burn)]

  Gama_I_L<-model$Gama_I_L[,,-(1:Burn)]
  Gama_I_R<-model$Gama_I_R[,,-(1:Burn)]

  Beta_I_L<-model$Beta_I_L[,,-(1:Burn)]
  Beta_I_R<-model$Beta_I_R[,,-(1:Burn)]



  ## Boundary


  boundary_rand_L<-sapply(1:N,function(i) as.matrix((Phi_mat[[i]])%*%as.matrix(Gama_I_L[,i,])))

  boundary_rand_R<-sapply(1:N,function(i) as.matrix((Phi_mat[[i]])%*%as.matrix(Gama_I_R[,i,])))


  Curve_boundary_rand_L<- vector("list", N)
  Curve_boundary_rand_R<- vector("list", N)

  Cred_boundary_rand_L<- vector("list", N)
  Cred_boundary_rand_R<- vector("list", N)

  Fitted_boundary_rand_L <- vector("list", N)
  Fitted_boundary_rand_R <- vector("list", N)
  boundary_rand_L_frame <- vector("list", N)
  boundary_rand_R_frame <- vector("list", N)

  combined_boundary_rand <- vector("list", N)


  for (i in seq_len(N)) {

    Cred_boundary_rand_L[[i]]<-apply(exp(boundary_rand_L[[i]]),1,quantile,probs=c(0.025,0.975))
    Cred_boundary_rand_L[[i]]<-data.frame((t(Cred_boundary_rand_L[[i]])))
    colnames(Cred_boundary_rand_L[[i]])<-c("lower_CI","Upper_CI")

    Cred_boundary_rand_R[[i]]<-apply(exp(boundary_rand_R[[i]]),1,quantile,probs=c(0.025,0.975))
    Cred_boundary_rand_R[[i]]<-data.frame((t(Cred_boundary_rand_R[[i]])))
    colnames(Cred_boundary_rand_R[[i]])<-c("lower_CI","Upper_CI")

  }




  for (i in seq_len(N)) {

    Fitted_boundary_rand_L[[i]]<-apply(exp(boundary_rand_L[[i]]),1,median)
    Fitted_boundary_rand_R[[i]]<-apply(exp(boundary_rand_R[[i]]),1,median)

    Curve_boundary_rand_L[[i]]<-data.frame(Time_point[[i]],y=Fitted_boundary_rand_L[[i]],Curve="Left Stimulus")

    boundary_rand_L_frame[[i]]<-cbind(Curve_boundary_rand_L[[i]],Cred_boundary_rand_L[[i]])

    Curve_boundary_rand_R[[i]]<-data.frame(Time_point[[i]],y=Fitted_boundary_rand_R[[i]],Curve="Right Stimulus")

    boundary_rand_R_frame[[i]]<-cbind(Curve_boundary_rand_R[[i]],Cred_boundary_rand_R[[i]])

    combined_boundary_rand[[i]] <- bind_rows(boundary_rand_L_frame[[i]],boundary_rand_R_frame[[i]])

    colnames(combined_boundary_rand[[i]])<-c("Time","Fitted_boundary_rand","Curve","lower_CI",  "Upper_CI")


  }





  ## Drift

  drift_rand_L<-sapply(1:N,function(i) as.matrix((Phi_mat[[i]])%*%as.matrix(Beta_I_L[,i,])))

  drift_rand_R<-sapply(1:N,function(i) as.matrix((Phi_mat[[i]])%*%as.matrix(Beta_I_R[,i,])))



  Curve_drift_rand_L<- vector("list", N)
  Curve_drift_rand_R<- vector("list", N)

  Cred_drift_rand_L<- vector("list", N)
  Cred_drift_rand_R<- vector("list", N)

  Fitted_drift_rand_L <- vector("list", N)
  Fitted_drift_rand_R <- vector("list", N)
  drift_rand_L_frame <- vector("list", N)
  drift_rand_R_frame <- vector("list", N)

  combined_drift_rand <- vector("list", N)


  for (i in seq_len(N)) {

    Cred_drift_rand_L[[i]]<-apply(exp(drift_rand_L[[i]]),1,quantile,probs=c(0.025,0.975))
    Cred_drift_rand_L[[i]]<-data.frame((t(Cred_drift_rand_L[[i]])))
    colnames(Cred_drift_rand_L[[i]])<-c("lower_CI","Upper_CI")

    Cred_drift_rand_R[[i]]<-apply(exp(drift_rand_R[[i]]),1,quantile,probs=c(0.025,0.975))
    Cred_drift_rand_R[[i]]<-data.frame((t(Cred_drift_rand_R[[i]])))
    colnames(Cred_drift_rand_R[[i]])<-c("lower_CI","Upper_CI")

  }




  for (i in seq_len(N)) {

    Fitted_drift_rand_L[[i]]<-apply(exp(drift_rand_L[[i]]),1,mean)
    Fitted_drift_rand_R[[i]]<-apply(exp(drift_rand_R[[i]]),1,mean)

    Curve_drift_rand_L[[i]]<-data.frame(Time_point[[i]],y=Fitted_drift_rand_L[[i]],Curve="Left Stimulus")

    drift_rand_L_frame[[i]]<-cbind(Curve_drift_rand_L[[i]],Cred_drift_rand_L[[i]])

    Curve_drift_rand_R[[i]]<-data.frame(Time_point[[i]],y=Fitted_drift_rand_R[[i]],Curve="Right Stimulus")

    drift_rand_R_frame[[i]]<-cbind(Curve_drift_rand_R[[i]],Cred_drift_rand_R[[i]])

    combined_drift_rand[[i]] <- bind_rows(drift_rand_L_frame[[i]],drift_rand_R_frame[[i]])

    colnames(combined_drift_rand[[i]])<-c("Time","Fitted_drift_rand","Curve","lower_CI",  "Upper_CI")


  }


  ## Plotting


  combined_rand <- list()

  for (i in 1:N) {

    Boundary_rand<-ggplot(combined_boundary_rand[[i]], aes(Time/1000, y=Fitted_boundary_rand, color = Curve)) +
      geom_ribbon(aes(ymin = lower_CI, ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed")+
      geom_line() +
      xlab("t(second)") + ylab("Estimated Boundary (Random Effect)") +
      labs(title = "Plot of Boundary Parameter (Random Effect) for left and Right Stimulus")

    Drift_rand<-ggplot(combined_drift_rand[[i]] , aes(Time/1000, y=Fitted_drift_rand, color = Curve)) +
      geom_ribbon(aes(ymin = lower_CI,ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed") +
      geom_line() +
      xlab("t(second)") + ylab("Estimated Drift (Random Effect)") +
      labs(title = "Plot of Drift Parameter (Random Effect) for left and Right Stimulus")



    combined_plot_rand <- ggarrange(Boundary_rand, Drift_rand,
                                    ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")

    combined_rand[[i]] <- combined_plot_rand


  }



  ########################################  Final Curves  ###################################################



  ## Boundary with main & Random effects
  ## Fitted Value

  b_l_prime <- vector("list", N)
  b_r_prime <- vector("list", N)

  Curve_b_L_prime<- vector("list", N)
  Curve_b_R_prime<- vector("list", N)

  Cred_b_L_prime<- vector("list", N)
  Cred_b_R_prime<- vector("list", N)



  for (i in seq_len(N)) {
    # Compute b_l and b_r
    b_l_prime[[i]] <- (X1[[i]] %*% Gamma_L) + boundary_rand_L[[i]]
    b_r_prime[[i]] <- (X1[[i]] %*% Gamma_R) + boundary_rand_R[[i]]

    Cred_b_L_prime[[i]]<-apply(exp(b_l_prime[[i]]),1,quantile,probs=c(0.025,0.975))
    Cred_b_L_prime[[i]]<-data.frame((t(Cred_b_L_prime[[i]])))
    colnames(Cred_b_L_prime[[i]])<-c("lower_CI","Upper_CI")

    Cred_b_R_prime[[i]]<-apply(exp(b_r_prime[[i]]),1,quantile,probs=c(0.025,0.975))
    Cred_b_R_prime[[i]]<-data.frame((t(Cred_b_R_prime[[i]])))
    colnames(Cred_b_R_prime[[i]])<-c("lower_CI","Upper_CI")

  }




  Fitted_b_L_prime <- vector("list", N)
  Fitted_b_R_prime <- vector("list", N)
  b_L_frame_prime <- vector("list", N)
  b_R_frame_prime <- vector("list", N)

  combined_b_prime <- vector("list", N)

  for (i in seq_len(N)) {

    Fitted_b_L_prime[[i]]<-apply(exp(b_l_prime[[i]]),1,median)
    Fitted_b_R_prime[[i]]<-apply(exp(b_r_prime[[i]]),1,median)

    Curve_b_L_prime[[i]]<-data.frame(Time_point[[i]],y=Fitted_b_L_prime[[i]],Curve="Left Stimulus")

    b_L_frame_prime[[i]]<-cbind(Curve_b_L_prime[[i]],Cred_b_L_prime[[i]])

    Curve_b_R_prime[[i]]<-data.frame(Time_point[[i]],y=Fitted_b_R_prime[[i]],Curve="Right Stimulus")

    b_R_frame_prime[[i]]<-cbind(Curve_b_R_prime[[i]],Cred_b_R_prime[[i]])

    combined_b_prime[[i]] <- bind_rows(b_L_frame_prime[[i]],b_R_frame_prime[[i]])

    colnames(combined_b_prime[[i]])<-c("Time","Fitted_b_prime","Curve","lower_CI",  "Upper_CI")


  }




  ## Drift with Random effects
  ## Fitted Value

  nu_l_prime <- vector("list", N)
  nu_r_prime <- vector("list", N)

  Cred_nu_L_prime <- vector("list", N)
  Cred_nu_R_prime <- vector("list", N)

  # Loop through each element
  for (i in seq_len(N)) {
    # Compute b_l and b_r
    nu_l_prime[[i]] <- (X1[[i]] %*% Beta_L) + drift_rand_L[[i]]
    nu_r_prime[[i]] <- (X1[[i]] %*% Beta_R) + drift_rand_R[[i]]

    Cred_nu_L_prime[[i]]<-apply(exp(nu_l_prime[[i]]),1,quantile,probs=c(0.025,0.975))
    Cred_nu_L_prime[[i]]<-data.frame((t(Cred_nu_L_prime[[i]])))
    colnames(Cred_nu_L_prime[[i]])<-c("lower_CI","Upper_CI")

    Cred_nu_R_prime[[i]]<-apply(exp(nu_r_prime[[i]]),1,quantile,probs=c(0.025,0.975))
    Cred_nu_R_prime[[i]]<-data.frame((t(Cred_nu_R_prime[[i]])))
    colnames(Cred_nu_R_prime[[i]])<-c("lower_CI","Upper_CI")

  }


  Fitted_nu_L_prime <- vector("list", N)
  Fitted_nu_R_prime <- vector("list", N)

  Curve_nu_L_prime <- vector("list", N)
  Curve_nu_R_prime <- vector("list", N)

  nu_L_frame_prime <- vector("list", N)
  nu_R_frame_prime <- vector("list", N)

  combined_nu_prime <- vector("list", N)


  for (i in seq_len(N)) {
    Fitted_nu_L_prime[[i]]<-apply(exp(nu_l_prime[[i]]),1,median)
    Fitted_nu_R_prime[[i]]<-apply(exp(nu_r_prime[[i]]),1,median)

    Curve_nu_L_prime[[i]]<-data.frame(Time_point[[i]],y= Fitted_nu_L_prime[[i]],Curve="Left Stimulus")
    nu_L_frame_prime[[i]]<-cbind(Curve_nu_L_prime[[i]],Cred_nu_L_prime[[i]])

    Curve_nu_R_prime[[i]]<-data.frame(Time_point[[i]],y=Fitted_nu_R_prime[[i]],Curve="Right Stimulus")

    nu_R_frame_prime[[i]]<-cbind(Curve_nu_R_prime[[i]],Cred_nu_R_prime[[i]])

    combined_nu_prime[[i]] <- bind_rows(nu_L_frame_prime[[i]],nu_R_frame_prime[[i]])
    colnames(combined_nu_prime[[i]])<-c("Time","Fitted_nu_prime","Curve","lower_CI",  "Upper_CI")


  }





  ## Mean Response time

  ## Left Stimulus


  mean_resp_temp_left <- vector("list", N)
  mean_resp_left <- vector("list", N)
  Curve_mean_resp_left<- vector("list", N)

  Cred_mean_resp_left<- vector("list", N)
  mean_resp_left_frame <- vector("list", N)


  for (i in seq_len(N)) {

    mean_resp_temp_left[[i]]<-exp(b_l_prime[[i]]-nu_l_prime[[i]]);

    mean_resp_left[[i]]<-Fitted_b_L_prime[[i]]/Fitted_nu_L_prime[[i]];

    Curve_mean_resp_left[[i]]<-data.frame(Time_point[[i]],y=mean_resp_left[[i]],Curve="Left Stimulus")


    Cred_mean_resp_left[[i]]<-apply(mean_resp_temp_left[[i]],1,quantile,probs=c(0.025,0.975))

    Cred_mean_resp_left[[i]]<-data.frame((t(Cred_mean_resp_left[[i]])))
    colnames(Cred_mean_resp_left[[i]])<-c("lower_CI","Upper_CI")

    mean_resp_left_frame[[i]]<-cbind(Curve_mean_resp_left[[i]],Cred_mean_resp_left[[i]])

    colnames(mean_resp_left_frame[[i]])<-c("Time","Fitted_mean_response","Curve","lower_CI",  "Upper_CI")

  }




  ## Right Stimulus


  mean_resp_temp_right <- vector("list", N)
  mean_resp_right <- vector("list", N)

  Curve_mean_resp_right<- vector("list", N)
  Cred_mean_resp_right<- vector("list", N)


  mean_resp_right_frame<- vector("list", N)
  combined_mean_response<- vector("list", N)


  # Loop through each element
  for (i in seq_len(N)) {

    mean_resp_temp_right[[i]]<-exp(b_r_prime[[i]]-nu_r_prime[[i]]);

    mean_resp_right[[i]]<-Fitted_b_R_prime[[i]]/Fitted_nu_R_prime[[i]];
    Curve_mean_resp_right[[i]]<-data.frame(Time_point[[i]],y=mean_resp_right[[i]],Curve="Right Stimulus")



    Cred_mean_resp_right[[i]]<-apply(mean_resp_temp_right[[i]],1,quantile,probs=c(0.025,0.975))
    Cred_mean_resp_right[[i]]<-data.frame((t(Cred_mean_resp_right[[i]])))
    colnames(Cred_mean_resp_right[[i]])<-c("lower_CI","Upper_CI")

    mean_resp_right_frame[[i]]<-cbind(Curve_mean_resp_right[[i]],Cred_mean_resp_right[[i]])

    colnames(mean_resp_right_frame[[i]])<-c("Time","Fitted_mean_response","Curve","lower_CI",  "Upper_CI")

    combined_mean_response[[i]]  <- bind_rows(mean_resp_left_frame[[i]], mean_resp_right_frame[[i]])


  }




  ## Plotting the parameters


  combined_prime <- list()

  for (i in 1:N) {

    Boundary_prime<-ggplot( combined_b_prime[[i]], aes(Time/1000, y=Fitted_b_prime, color = Curve)) +
      geom_ribbon(aes(ymin = lower_CI, ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed")+
      geom_line() +
      xlab("t(second)") + ylab("Estimated Boundary") +
      labs(title = "Plot of Boundary Parameter for left and Right Stimulus")

    Drift_prime<-ggplot(combined_nu_prime[[i]] , aes(Time/1000, y=Fitted_nu_prime, color = Curve)) +
      geom_ribbon(aes(ymin = lower_CI,ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed") +
      geom_line() +
      xlab("t(second)") + ylab("Estimated Drift") +
      labs(title = "Plot of Drift Parameter for left and Right Stimulus")



    Mean_Response_prime<-ggplot(combined_mean_response[[i]] , aes(Time/1000, y=Fitted_mean_response, color = Curve)) +
      geom_ribbon(aes(ymin = lower_CI, ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed") +
      geom_line() +
      xlab("t(second)") + ylab("Estimated Mean Response") +
      labs(title = "Plot of Mean Response for left and Right Stimulus")

    combined_plot_prime <- ggarrange(Boundary_prime, Drift_prime, Mean_Response_prime,
                                     ncol = 1, nrow = 3, common.legend = TRUE, legend = "bottom")

    combined_prime[[i]] <- combined_plot_prime


  }




  ################ Main effect adjusted for random effect



  ## Boundary

  boundary_rand_L_bar<-apply(Gama_I_L, c(1, 3), mean)

  boundary_rand_R_bar<-apply(Gama_I_R, c(1, 3), mean)

  Phi_boundary_L <- do.call(rbind, lapply(Phi_mat, function(Phi_i) Phi_i %*% boundary_rand_L_bar))
  Phi_boundary_R <- do.call(rbind, lapply(Phi_mat, function(Phi_i) Phi_i %*% boundary_rand_R_bar))



  ## Drift

  drift_rand_L_bar<-apply(Beta_I_L, c(1, 3), mean)
  drift_rand_R_bar<-apply(Beta_I_R, c(1, 3), mean)


  Phi_drift_L <- do.call(rbind, lapply(Phi_mat, function(Phi_i) Phi_i %*% drift_rand_L_bar))
  Phi_drift_R <- do.call(rbind, lapply(Phi_mat, function(Phi_i) Phi_i %*% drift_rand_R_bar))





  ### Boundary Parameter

  ## b_L
  ## Fitted Value

  b_l_temp<-exp(sp.x1%*%Gamma_L+Phi_boundary_L)

  Fitted_b_L<-apply(b_l_temp,1,median)


  ## Curve fitting
  Curve_b_L<-data.frame(Time,y=Fitted_b_L,Curve="Left Stimulus")


  ## Credible interval
  Cred_b_L<-apply(b_l_temp,1,quantile,probs=c(0.025,0.975))

  Cred_b_L<-data.frame((t(Cred_b_L)))
  colnames(Cred_b_L)<-c("lower_CI","Upper_CI")

  b_L_frame<-cbind(Curve_b_L,Cred_b_L)



  ## b_R
  ## Fitted Value

  b_r_temp<-exp(sp.x1%*%Gamma_R+Phi_boundary_R)

  Fitted_b_R<-apply(b_r_temp,1,mean)

  ## Curve fitting
  Curve_b_R<-data.frame(Time,y=Fitted_b_R,Curve="Right Stimulus")


  ## Credible interval
  Cred_b_R<-apply(b_r_temp,1,quantile,probs=c(0.025,0.975))

  Cred_b_R<-data.frame((t(Cred_b_R)))
  colnames(Cred_b_R)<-c("lower_CI","Upper_CI")

  b_R_frame<-cbind(Curve_b_R,Cred_b_R)


  # Combine the data frames for left and right stimulus
  combined_b <- bind_rows(b_L_frame,b_R_frame)



  ## Drift Parameter

  ## nu_L
  ## Fitted Value

  nu_l_temp<-exp(sp.x1%*%Beta_L+Phi_drift_L)

  Fitted_nu_L<-apply(nu_l_temp,1,median)


  ## Curve fitting
  Curve_nu_L<-data.frame(Time,y= Fitted_nu_L,Curve="Left Stimulus")


  ## Credible interval
  Cred_nu_L<-apply(nu_l_temp,1,quantile,probs=c(0.025,0.975))

  Cred_nu_L<-data.frame((t(Cred_nu_L)))
  colnames(Cred_nu_L)<-c("lower_CI","Upper_CI")

  nu_L_frame<-cbind(Curve_nu_L,Cred_nu_L)



  ## Beta_R
  ## Fitted Value


  nu_r_temp<-exp(sp.x1%*%Beta_R+Phi_drift_R)

  Fitted_nu_R<-apply(nu_r_temp,1,median)

  ## Curve fitting
  Curve_nu_R<-data.frame(Time,y=Fitted_nu_R,Curve="Right Stimulus")



  ## Credible interval
  Cred_nu_R<-apply(nu_r_temp,1,quantile,probs=c(0.025,0.975))

  Cred_nu_R<-data.frame((t(Cred_nu_R)))
  colnames(Cred_nu_R)<-c("lower_CI","Upper_CI")

  nu_R_frame<-cbind(Curve_nu_R,Cred_nu_R)



  # Combine the data frames for left and right stimulus
  combined_nu  <- bind_rows(nu_L_frame,nu_R_frame)




  ## Mean Response time

  ## Left Stimulus

  mean_resp_temp_left_main_eff<-b_l_temp/nu_l_temp

  Fitted_mean_resp_left_main_eff<-apply(mean_resp_temp_left_main_eff,1,median)


  ## Curve fitting
  Curve_mean_resp_left_main_eff<-data.frame(Time,y=Fitted_mean_resp_left_main_eff,Curve="Left Stimulus")


  ## Credible interval
  Cred_mean_resp_left_main_eff<-apply(mean_resp_temp_left_main_eff,1,quantile,probs=c(0.025,0.975))

  Cred_mean_resp_left_main_eff<-data.frame((t(Cred_mean_resp_left_main_eff)))
  colnames(Cred_mean_resp_left_main_eff)<-c("lower_CI","Upper_CI")

  mean_resp_left_frame_main_eff<-cbind(Curve_mean_resp_left_main_eff,Cred_mean_resp_left_main_eff)




  ## Right Stimulus

  mean_resp_temp_right_main_eff<-b_r_temp/nu_r_temp
  Fitted_mean_resp_right_main_eff<-apply(mean_resp_temp_right_main_eff,1,median)


  ## Curve fitting
  Curve_mean_resp_right_main_eff<-data.frame(Time,y=Fitted_mean_resp_right_main_eff,Curve="Right Stimulus")


  ## Credible interval
  Cred_mean_resp_right_main_eff<-apply(mean_resp_temp_right_main_eff,1,quantile,probs=c(0.025,0.975))

  Cred_mean_resp_right_main_eff<-data.frame((t(Cred_mean_resp_right_main_eff)))
  colnames(Cred_mean_resp_right_main_eff)<-c("lower_CI","Upper_CI")

  mean_resp_right_frame_main_eff<-cbind(Curve_mean_resp_right_main_eff,Cred_mean_resp_right_main_eff)


  # Combine the data frames for left and right stimulus
  combined_mean_response_main_eff  <- bind_rows(mean_resp_left_frame_main_eff, mean_resp_right_frame_main_eff)



  Boundary_main_eff_adj_rand<-ggplot( combined_b, aes(Time/1000, y, color = Curve)) +
    geom_ribbon(aes(ymin = lower_CI, ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed")+
    geom_line() +
    xlab("t (second)") + ylab("Estimated Boundary") +
    labs(title = "Plot of Boundary Parameter for left and Right Stimulus")+
    theme_bw() +
    theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 20),
      legend.text = element_text(size = 20),
      legend.title = element_blank()
    )




  Drift_main_eff_adj_rand<-ggplot(combined_nu , aes(Time/1000, y, color = Curve)) +
    geom_ribbon(aes(ymin = lower_CI,ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed") +
    geom_line() +
    xlab("t (second)") + ylab("Estimated Drift") +
    labs(title = "Plot of Drift Parameter for left and Right Stimulus")+
    theme_bw() +
    theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 20),
      legend.text = element_text(size = 20),
      legend.title = element_blank()
    )




  Mean_Response_main_eff_adj_rand<-ggplot(combined_mean_response_main_eff , aes(Time/1000, y, color = Curve)) +
    geom_ribbon(aes(ymin = lower_CI, ymax = Upper_CI), fill = "grey",alpha=0.6,linetype="dashed") +
    geom_line() +
    xlab("t (second)") + ylab("Estimated Mean Response (main effect)") +
    labs(title = "Plot of Mean Response (main effect) for left and Right Stimulus")+
    theme_bw() +
    theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 20),
      legend.text = element_text(size = 20),
      legend.title = element_blank()
    )




  Parameter_graph_main_effect_adj_rand_effect<-ggpubr::ggarrange(Boundary_main_eff_adj_rand,Drift_main_eff_adj_rand,Mean_Response_main_eff_adj_rand,nrow=3)








  ## Random plots of main effect parameters

  par(mfrow = c(2, 2))

  Gama_L_plot<-plot(model$Gamma_L[sample.int(nrow(model$Gamma_L),size = 1),],type="l",main="Plot of Gama_L (Random sample)",xlab="t",ylab="Gama_L")

  Gama_R_plot<-plot(model$Gamma_R[sample.int(nrow(model$Gamma_R),size = 1),],type="l",main="Plot of Gama_R (Random sample)",xlab="t",ylab="Gama_R")

  Beta_L_plot<-plot(model$Beta_L[sample.int(nrow(model$Beta_L),size = 1),],type="l",main="Plot of Beta_L (Random sample)",xlab="t",ylab="Beta_L")

  Beta_R_plot<-plot(model$Beta_R[sample.int(nrow(model$Beta_R),size = 1),],type="l",main="Plot of Beta_R (Random sample)",xlab="t",ylab="Beta_R")

  # Reset plotting parameters
  par(mfrow = c(1, 1))

  # Save the combined plot
  plot_param_rand <- recordPlot()



  ## Estimated Random curve plots of main effect parameters


  par(mfrow = c(2, 2))

  b_l_E<-matplot(t(b_l_temp[sample.int(nrow(b_l_temp),size = 5),]),type="l",main="Matplot of Estimated b_l (Random sample)",xlab="t",ylab="Boundary (Left Stimuls)")

  nu_l_E<-matplot(t(nu_l_temp[sample.int(nrow(nu_l_temp),size = 5),]),type="l",main="Matplot of Estimated nu_l (Random sample)",xlab="t",ylab="Drift (Left Stimuls)")

  b_r_E<-matplot(t(b_r_temp[sample.int(nrow(b_r_temp),size = 5),]),type="l",main="Matplot of Estimated b_r (Random sample)",xlab="t",ylab="Boundary (Rigth Stimuls)")

  nu_r_E<-matplot(t(nu_r_temp[sample.int(nrow(nu_r_temp),size = 5),]),type="l",main="Matplot of Estimated nu_r (Random sample)",xlab="t",ylab="Drift (Right Stimuls)")


  # Reset plotting parameters
  par(mfrow = c(1, 1))

  est_plot_param_rand<- recordPlot()





  # Create an empty list to store the recorded plots
  est_plot_param_with_random_effect <- list()

  for (i in 1:N) {
    # Set up a 2x2 plotting layout
    par(mfrow = c(2, 2))

    # Safeguard for sample size
    sample_size_b_l <- min(5, nrow(b_l_prime[[i]]))
    sample_size_nu_l <- min(5, nrow(nu_l_prime[[i]]))
    sample_size_b_r <- min(5, nrow(b_r_prime[[i]]))
    sample_size_nu_r <- min(5, nrow(nu_r_prime[[i]]))

    # Plot Boundary (Left Stimulus)
    matplot(
      t(b_l_prime[[i]][sample.int(nrow(b_l_prime[[i]]), size = sample_size_b_l), ]),
      type = "l", main = "Matplot of Estimated b_l_prime (Random Sample)",
      xlab = "t", ylab = "Boundary (Left Stimulus)", col = rainbow(sample_size_b_l)
    )

    # Plot Drift (Left Stimulus)
    matplot(
      t(nu_l_prime[[i]][sample.int(nrow(nu_l_prime[[i]]), size = sample_size_nu_l), ]),
      type = "l", main = "Matplot of Estimated nu_l_prime (Random Sample)",
      xlab = "t", ylab = "Drift (Left Stimulus)", col = rainbow(sample_size_nu_l)
    )

    # Plot Boundary (Right Stimulus)
    matplot(
      t(b_r_prime[[i]][sample.int(nrow(b_r_prime[[i]]), size = sample_size_b_r), ]),
      type = "l", main = "Matplot of Estimated b_r_prime (Random Sample)",
      xlab = "t", ylab = "Boundary (Right Stimulus)", col = rainbow(sample_size_b_r)
    )

    # Plot Drift (Right Stimulus)
    matplot(
      t(nu_r_prime[[i]][sample.int(nrow(nu_r_prime[[i]]), size = sample_size_nu_r), ]),
      type = "l", main = "Matplot of Estimated nu_r_prime (Random Sample)",
      xlab = "t", ylab = "Drift (Right Stimulus)", col = rainbow(sample_size_nu_r)
    )

    # Reset plotting layout to default
    par(mfrow = c(1, 1))

    # Record the plot and store it in the list
    est_plot_param_with_random_effect[[i]] <- recordPlot()
  }


  # est_plot_param_rand<-ggpubr::ggarrange(  b_l_E,  b_r_E,  nu_l_E,nu_r_E,nrow=2,ncol=2)



  Data<-list('combined_b'=combined_b,'combined_nu'=combined_nu,"combined_mean_response_main_eff"=combined_mean_response_main_eff,
             "combined_boundary_rand"=combined_boundary_rand,"combined_drift_rand"=combined_drift_rand,
             "combined_b_prime"=combined_b_prime,"combined_nu_prime"=combined_nu_prime,"combined_mean_response"=combined_mean_response)


  Plots<-list("convergence_diagnostic_plots"=convergence_diagnostic_plots,"Boundary_main_eff"=Boundary_main_eff,'Drift'=Drift_main_eff,
              "Mean_Response_main_eff"=Mean_Response_main_eff,'Parameter_graph_main_effect'=Parameter_graph_main_effect,

              'Parameter_graph_Random_effect'=combined_rand,

              "Boundary_prime"=Boundary_prime,'Drift_prime'=Drift_prime,
              "Mean_Response_prime"=Mean_Response_prime,'Parameter_graph'=combined_prime,

              "Boundary_main_eff_adj_rand"=Boundary_main_eff_adj_rand,'Drift_main_eff_adj_rand'=Drift_main_eff_adj_rand,
              "Mean_Response_main_eff_adj_rand"=Mean_Response_main_eff_adj_rand,
              'Parameter_graph_main_effect_adj_rand_effect'=Parameter_graph_main_effect_adj_rand_effect,

              'plot_param_rand'= plot_param_rand,
              'est_plot_param_with_random_effect'=est_plot_param_with_random_effect)
  Result<-list("Data"=Data,'Plots'=Plots)

  return(Result)
}


