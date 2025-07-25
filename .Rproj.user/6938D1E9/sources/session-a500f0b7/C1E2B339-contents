#' Generate Diagnostic and Posterior Graphs from SSDDM Output
#'
#' This function loads a saved model and dataset, extracts relevant posterior samples,
#' and generates convergence diagnostics and inference plots using pre-specified plotting routines.
#'
#' @param model_name Path to the RDS file containing the saved model
#' @param data Path to the RDS file containing the preprocessed data
#' @param Burn Number of burn-in iterations to discard
#' @param SSD_min Minimum SSD value used for filtering (default = 50)
#' @param CI Credible interval width (e.g., 0.95)
#' @param convergence_plots Logical. Whether to generate convergence diagnostic plots
#'
#' @return A list containing ggplot objects or summary statistics for visualization
#' @export


graph_generate<-function(model_name,data,Burn,SSD_min=50,CI,convergence_plots){


  model<-readRDS(model_name)




  ## Creating Dataset with the subjects
  data<-readRDS(data)



  Gama_I_L<-model$Gama_I_L[,,-(1:Burn)]
  Gama_I_R<-model$Gama_I_R[,,-(1:Burn)]

  Beta_I_L<-model$Beta_I_L[,,-(1:Burn)]
  Beta_I_R<-model$Beta_I_R[,,-(1:Burn)]

  Phi_mat=data$Phi_Full

  Time=data$Time_Point

  sp.x1=data$sp.x1_Full ## Spline

  X1=data$X1 ## Spline

  U<-data$U

  N=length(Time)


  if (interactive()) {
    source("inst/scripts/Graph Preparation.R", echo = TRUE)
  } else {
    source(system.file("scripts", "Graph Preparation.R", package = "BSDDM"), echo = TRUE)
  }


  result<-Plot_results(model,sp.x1,X1=sp.x1,Time_point=Time, Burn, N,Phi_mat,U,SSD_min, CI,convergence_plots)

  return(result)

}
