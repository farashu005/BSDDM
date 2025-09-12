## newdat=dataset
## sub=selected subjects


data_prep<-function(newdat,sub,nknots,m,run,scale){

  # small_dat<-newdat[newdat$subject %in% sub,]

  main_dat<-newdat[newdat$subject %in% sub,]


  main_dat$sst_stimonset<-as.numeric(main_dat$sst_stimonset)
  main_dat$sst_primaryrt<-as.numeric(main_dat$sst_primaryrt)
  main_dat$sst_go_rt<-as.numeric(main_dat$sst_go_rt)

  main_dat$sst_run<-as.numeric(main_dat$sst_run)


  main_dat$sst_inhibitacc<-as.numeric(main_dat$sst_inhibitacc)


  main_dat$sst_ssd_dur[is.na(main_dat$sst_ssd_dur)] = 0

  main_dat$sst_primaryrt[is.na(main_dat$sst_primaryrt)] = 0

  main_dat$sst_go_resp[is.na(main_dat$sst_primaryrt)] = 0


  if (run==1){

    main_dat<-main_dat[main_dat$sst_run==1,]

  }


  if (run==2){

    main_dat<-main_dat[main_dat$sst_run==2,]

  }

  else{

    main_dat=main_dat
  }

  ## Removing SSD>=700

  main_dat <- main_dat[!(main_dat$sst_ssd_dur >=700), ]


  main_dat$go_resp_diff<-main_dat$sst_primaryrt-main_dat$sst_ssd_dur


  ## If tau<SSD in Failed Stop trail, considering them as Go Trial

  main_dat$sst_expcon[main_dat$sst_expcon=="VariableStopTrial" & main_dat$sst_inhibitacc==0 & main_dat$go_resp_diff< 0]<-"GoTrial"

  ## Making tau-SSD=0 for Go Trial

  main_dat$go_resp_diff[main_dat$sst_expcon=="GoTrial" & main_dat$go_resp_diff< 0]<-0

  ## Making tau-SSD=0 for Failed Stop Trial

  main_dat$go_resp_diff[main_dat$sst_expcon=="VariableStopTrial" & main_dat$sst_inhibitacc==1 & main_dat$go_resp_diff< 0]<-0



  # main_dat$GF <- ifelse(main_dat$sst_expcon == "GoTrial" & (is.na(main_dat$sst_choiceacc) | main_dat$sst_primaryrt == 0), 1, 0)


  ## Go Failures ## Implying that primary response time <=200 is GF
  main_dat$GF <- ifelse(main_dat$sst_expcon == "GoTrial" & (is.na(main_dat$sst_choiceacc) | main_dat$sst_primaryrt <=200), 1, 0)

  ## Removing datapoints where inhibition accurcay=0 and RT <=200

  main_dat <- main_dat[!(main_dat$sst_expcon == "VariableStopTrial" &
                           main_dat$sst_inhibitacc == 0 &
                           main_dat$sst_primaryrt <= 200), ]


  ## Removing datapoints where SSD_Dur=0 and Inhibation accuracy=1

  main_dat <- main_dat[!(main_dat$sst_expcon == "VariableStopTrial" &
                           main_dat$sst_inhibitacc == 1 &
                           main_dat$sst_ssd_dur==0),]


  library(dplyr)

  main_dat <- main_dat %>%
    group_by(subject) %>%
    mutate(PGF = mean(GF, na.rm = TRUE)*100)%>%
    ungroup()



  main_dat <- main_dat[!(main_dat$PGF>=25),]

  main_dat <- main_dat %>%
    group_by(subject) %>%
    mutate(Inhib_acc = mean(sst_inhibitacc[sst_expcon == "VariableStopTrial"], na.rm = TRUE) * 100) %>%
    ungroup()


  main_dat <- main_dat[!(main_dat$Inhib_acc<=25),]


  ## Spliting dataset with respect to Subject
  B<-split(main_dat,main_dat$subject)


  ## Saving Go Failures

  W<-length(B)

  Ind_G=list()
  for ( i in 1:W){
    Ind_G[[i]]=B[[i]]$GF
  }


  T_Total<-c()
  for ( i in 1:W){
    T_Total[i]=length(B[[i]]$sst_expcon)
  }

  Ind_G=list()
  for ( i in 1:W){
    Ind_G[[i]]=B[[i]]$GF
  }


  FT<-c()
  for (i in 1:W){
    FT[i]=sum(Ind_G[[i]])
  }


  T_go<-c()

  for ( i in 1:W){
    T_go[i]=sum(B[[i]]$sst_expcon=="GoTrial")
  }


  f<-function(x){
    R=x-x[1]
    return(R)
  }





  t_full=list()
  for ( i in 1:length(B)){
    t_full[[i]]=(f(B[[i]]$sst_stimonset)/scale)
  }



  T_full<-Reduce(c,t_full)

  L_full<-max(T_full)/2




  T_full_list=list()
  for ( i in 1:length(B)){
    T_full_list[[i]]=t_full[[i]]-L_full
  }



  # Creating Lambda
  lambda_full <- c()

  for (j in 1:(m)) {
    lambda_full[j] <- (( j  * pi) / (2 * L_full))^2
  }



  ## Creating Phi
  Phi_full <- list()

  for (k in 1:length(B)) {
    # Initialize each matrix Phi[[k]] with dimensions n x (m + 1) for each k
    Phi_full[[k]] <- matrix(0, nrow = length(T_full_list[[k]]), ncol = m)

    for (i in 1:length(T_full_list[[k]])) {
      for (j in 1:m) {
        # Calculate and assign the value to Phi[[k]][i, j]
        Phi_full[[k]][i, j] <- sqrt(1 / L_full) * sin(sqrt(lambda_full[j]) * (T_full_list[[k]][i] + L_full))
      }
    }
  }




  min_val <- min(T_full, na.rm = TRUE)
  quantile_10 <- quantile(T_full, 0.10, na.rm = TRUE)
  quantile_90 <- quantile(T_full, 0.90, na.rm = TRUE)
  max_val <- max(T_full, na.rm = TRUE)

  # Define the number of points you want between the 10th and 90th percentiles
  n_points <- nknots-4  # Example: divide into  intervals between quantile_10 and quantile_90

  points <- seq(quantile_10, quantile_90, length.out = n_points + 2) # +2 to include both quantiles

  # Remove the first and last points since they are the quantile_10 and quantile_90 themselves
  points <- points[-c(1, length(points))]


  sp.x1_Full <- splines2::bSpline(
    T_full,
    # knots = seq(min(X), max(X), length.out = nknots),
    knots = c(min_val,quantile_10,points,quantile_90,max_val),
    degree = 3L,
    intercept = FALSE,
    Boundary.knots = c(min(T_full) - 0.5, max(T_full) + 0.5),
    extent = 1  # Set extent to a single value
  )

  sp.x1_Full=sp.x1_Full[,-which(colSums(sp.x1_Full)==0)]


  combine_list <- function(X, L, new_column_value = NULL) {
    result <- list()
    start_index <- 1

    for (i in seq_along(L)) {
      end_index <- start_index + L[i] - 1
      submatrix <- as.matrix(X[start_index:end_index, ])

      if (!is.null(new_column_value)) {
        new_column <- matrix(new_column_value, nrow = nrow(submatrix), ncol = 1)
        submatrix <- cbind(new_column, submatrix)
      }

      result[[i]] <- submatrix
      start_index <- end_index + 1
    }

    return(result)
  }


  tmp=list()
  for ( i in 1:W){
    tmp[[i]]=B[[i]]$sst_stimonset
  }

  Len_Full<-sapply(tmp,length)


  # Call the function with a column of 1s as the first column
  X1_Full <- combine_list(sp.x1_Full, Len_Full, new_column_value = 1)

  sp.x1_Full=cbind(1,sp.x1_Full)




  ## Removing Go Failures from the data

  main_dat<-main_dat[main_dat$GF==0,]

  main_dat$Ind_L<-ifelse(main_dat$sst_stim=="left_arrow",1,0)
  main_dat$Ind_R<-ifelse(main_dat$sst_stim=="right_arrow",1,0)


  main_dat$Ind_LCR<-ifelse(main_dat$sst_expcon=="GoTrial" & main_dat$sst_stim=="left_arrow" & main_dat$sst_primaryresp=="left_arrow" ,1,0)
  main_dat$Ind_LIR<-ifelse(main_dat$sst_expcon=="GoTrial" & main_dat$sst_stim=="left_arrow" & main_dat$sst_primaryresp=="right_arrow" ,1,0)
  main_dat$Ind_RIR<-ifelse(main_dat$sst_expcon=="GoTrial" & main_dat$sst_stim=="right_arrow" & main_dat$sst_primaryresp=="left_arrow" ,1,0)
  main_dat$Ind_RCR<-ifelse(main_dat$sst_expcon=="GoTrial" & main_dat$sst_stim=="right_arrow" & main_dat$sst_primaryresp=="right_arrow" ,1,0)


  main_dat$Ind_S_LCR<-ifelse(main_dat$sst_expcon=="VariableStopTrial" & main_dat$sst_inhibitacc==0 & main_dat$sst_stim=="left_arrow" & main_dat$sst_primaryresp=="left_arrow" ,1,0)
  main_dat$Ind_S_LIR<-ifelse(main_dat$sst_expcon=="VariableStopTrial" & main_dat$sst_inhibitacc==0 & main_dat$sst_stim=="left_arrow" & main_dat$sst_primaryresp=="right_arrow" ,1,0)
  main_dat$Ind_S_RIR<-ifelse(main_dat$sst_expcon=="VariableStopTrial" & main_dat$sst_inhibitacc==0 & main_dat$sst_stim=="right_arrow" & main_dat$sst_primaryresp=="left_arrow" ,1,0)
  main_dat$Ind_S_RCR<-ifelse(main_dat$sst_expcon=="VariableStopTrial" & main_dat$sst_inhibitacc==0 & main_dat$sst_stim=="right_arrow" & main_dat$sst_primaryresp=="right_arrow" ,1,0)


  main_dat$Ind_I_L<-ifelse(main_dat$sst_expcon=="VariableStopTrial" & main_dat$sst_inhibitacc==1 & main_dat$sst_stim=="left_arrow" ,1,0)
  main_dat$Ind_I_R<-ifelse(main_dat$sst_expcon=="VariableStopTrial" & main_dat$sst_inhibitacc==1 & main_dat$sst_stim=="right_arrow",1,0)



  ## Spliting dataset with respect to Subject
  A<-split(main_dat,main_dat$subject)

  N=length(A)



  ## Creating t

  t=list()
  for ( i in 1:N){
    t[[i]]=(f(A[[i]]$sst_stimonset)/scale)
  }


  W<-length(A)

  # tmp2=list()
  # for ( i in 1:W){
  #   tmp2[[i]]=t[[i]]
  # }
  #

  X2<-Reduce(c,t)

  L<-max(X2)/2



  T=list()
  for ( i in 1:N){
    T[[i]]=t[[i]]-L
  }



  # Creating Lambda
  lambda <- c()

  for (j in 1:(m)) {
    lambda[j] <- (( j  * pi) / (2 * L))^2
  }



  ## Creating Phi
  Phi <- list()

  for (k in 1:N) {
    # Initialize each matrix Phi[[k]] with dimensions n x (m + 1) for each k
    Phi[[k]] <- matrix(0, nrow = length(T[[k]]), ncol = m)

    for (i in 1:length(T[[k]])) {
      for (j in 1:m) {
        # Calculate and assign the value to Phi[[k]][i, j]
        Phi[[k]][i, j] <- sqrt(1 / L) * sin(sqrt(lambda[j]) * (T[[k]][i] + L))
      }
    }
  }





  ## Creating tau

  tau=list()
  for ( i in 1:N){
    tau[[i]]=A[[i]]$sst_primaryrt
  }



  ## Creating tau_s

  tau_s=list()
  for ( i in 1:N){
    tau_s[[i]]=A[[i]]$go_resp_diff
  }




  ## Creating U
  Z <- list()
  for (i in 1:N) {
    Z[[i]] <- min(
      A[[i]]$sst_primaryrt[
        A[[i]]$sst_expcon == "GoTrial" |
          (A[[i]]$sst_expcon == "VariableStopTrial" & A[[i]]$sst_inhibitacc == 0)
      ]
    )
  }


  U=unlist(Z)

  SSD=list()
  for ( i in 1:N){
    SSD[[i]]=A[[i]]$sst_ssd_dur
  }


  l_lim=list()
  for ( i in 1:N){
    l_lim[[i]]=A[[i]]$sst_ssd_dur
  }


  ## Creating Ind_L & Ind_R
  f2<-function(Ind){
    R<-which(Ind==1)
    return(R)
  }


  Indic_L=list()
  Indic_R=list()


  for (i in 1:N){
    Indic_L[[i]]=f2(A[[i]]$Ind_L)
    Indic_R[[i]]=f2(A[[i]]$Ind_R)

  }



  ## Creating Ind_LCR, Ind_LIR,Ind_RCR, Ind_RIR

  # Splitting the dataset with respect to Ind_L and Ind_R

  split_L <- list()

  split_R <- list()



  # Loop through each element in the list A
  for (i in 1:N) {
    # Split each element by the 'Category' variable
    split_L[[i]] <- split(A[[i]], A[[i]]$Ind_L)
    split_R[[i]] <- split(A[[i]], A[[i]]$Ind_R)

  }


  Ind_L_M<-lapply(split_L,function(upper_list){
    inner_list<-upper_list[[2]]
    return(inner_list)
  })

  Ind_R_M<-lapply(split_R,function(upper_list){
    inner_list<-upper_list[[2]]
    return(inner_list)
  })

  ## Creating Ind
  Indic_LCR=list()
  Indic_LIR=list()
  Indic_RCR=list()
  Indic_RIR=list()

  Indic_S_LCR=list()
  Indic_S_LIR=list()
  Indic_S_RCR=list()
  Indic_S_RIR=list()

  Indic_I_L=list()
  Indic_I_R=list()


  for (i in 1:N){
    Indic_LCR[[i]]=f2(Ind_L_M[[i]]$Ind_LCR)
    Indic_LIR[[i]]=f2(Ind_L_M[[i]]$Ind_LIR)
    Indic_RCR[[i]]=f2(Ind_R_M[[i]]$Ind_RCR)
    Indic_RIR[[i]]=f2(Ind_R_M[[i]]$Ind_RIR)

    Indic_S_LCR[[i]]=f2(Ind_L_M[[i]]$Ind_S_LCR)
    Indic_S_LIR[[i]]=f2(Ind_L_M[[i]]$Ind_S_LIR)
    Indic_S_RCR[[i]]=f2(Ind_R_M[[i]]$Ind_S_RCR)
    Indic_S_RIR[[i]]=f2(Ind_R_M[[i]]$Ind_S_RIR)

    Indic_I_L[[i]]=f2(Ind_L_M[[i]]$Ind_I_L)
    Indic_I_R[[i]]=f2(Ind_R_M[[i]]$Ind_I_R)


  }





  f3<-function(x){
    R=x-1
    return(R)
  }

  Ind_L=list()
  Ind_R=list()
  Ind_LCR=list()
  Ind_LIR=list()
  Ind_RCR=list()
  Ind_RIR=list()

  Ind_S_LCR=list()
  Ind_S_LIR=list()
  Ind_S_RCR=list()
  Ind_S_RIR=list()

  Ind_I_L=list()
  Ind_I_R=list()



  for ( i in 1:N){
    Ind_L[[i]]=f3(Indic_L[[i]])
    Ind_R[[i]]=f3(Indic_R[[i]])

    Ind_LCR[[i]]=f3(Indic_LCR[[i]])
    Ind_LIR[[i]]=f3(Indic_LIR[[i]])
    Ind_RCR[[i]]=f3(Indic_RCR[[i]])
    Ind_RIR[[i]]=f3(Indic_RIR[[i]])

    Ind_S_LCR[[i]]=f3(Indic_S_LCR[[i]])
    Ind_S_LIR[[i]]=f3(Indic_S_LIR[[i]])
    Ind_S_RCR[[i]]=f3(Indic_S_RCR[[i]])
    Ind_S_RIR[[i]]=f3(Indic_S_RIR[[i]])

    Ind_I_L[[i]]=f3(Indic_I_L[[i]])
    Ind_I_R[[i]]=f3(Indic_I_R[[i]])





  }


  Ind<-list(Ind_L,Ind_R,Ind_LCR,Ind_LIR,Ind_RCR,Ind_RIR,
            Ind_S_LCR,Ind_S_LIR,Ind_S_RCR,Ind_S_RIR,Ind_I_L,Ind_I_R)


  W<-length(A)

  # tmp=list()
  # for ( i in 1:W){
  #   tmp[[i]]=T[[i]]
  # }
  #
  Len<-sapply(T,length)

  X<-Reduce(c,T)


  library(splines2)

  min_val <- min(X, na.rm = TRUE)
  quantile_10 <- quantile(X, 0.10, na.rm = TRUE)
  quantile_90 <- quantile(X, 0.90, na.rm = TRUE)
  max_val <- max(X, na.rm = TRUE)

  # Define the number of points you want between the 10th and 90th percentiles
  n_points <- nknots-4  # Example: divide into  intervals between quantile_10 and quantile_90

  points <- seq(quantile_10, quantile_90, length.out = n_points + 2) # +2 to include both quantiles

  # Remove the first and last points since they are the quantile_10 and quantile_90 themselves
  points <- points[-c(1, length(points))]



  sp.x1 <- splines2::bSpline(
    X,
    # knots = seq(min(X), max(X), length.out = nknots),
    knots = c(min_val,quantile_10,points,quantile_90,max_val),
    degree = 3L,
    intercept = FALSE,
    Boundary.knots = c(min(X) - 0.5, max(X) + 0.5),
    extent = 1  # Set extent to a single value
  )
  sp.x1=sp.x1[,-which(colSums(sp.x1)==0)]




  # Call the function with a column of 1s as the first column
  X1 <- combine_list(sp.x1, Len, new_column_value = 1)


  # X1<-combine_list(sp.x1,Len)

  sp.x1=cbind(1,sp.x1)





  Result<-list("tau"=tau,"tau_s"=tau_s,"Ind_G"=Ind_G,"U"=U,"L"=L,"L_full"=L_full,"lambda"=lambda,"Phi"=Phi,"SSD"=SSD,"T_Go"=T_go,
               "T_Total"=T_Total,"FT"=FT,"l_lim"=l_lim,"Ind"=Ind,'sp.x1'=sp.x1,"X1"=X1,
               "sp.x1_Full"=sp.x1_Full,"X1_Full"=X1_Full,"Phi_Full"=Phi_full,"Time_Point"=T_full,"Time_Point_list"=t_full,
               "Len_Full"=Len_Full,"Len"=Len,
               "dat"=main_dat)

  return(Result)

}






