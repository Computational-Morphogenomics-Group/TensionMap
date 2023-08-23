do_gsem_regression <- function(gene, force_metric, tensionmap_res, gex_res, k_sp=300, model_fx=T, method='REML'){
  data_df <- tensionmap_res[,c('centroid_x', 'centroid_y', force_metric)]
  data_df[['force_metric']] <- log(data_df[[force_metric]])
  data_df[['gene']] <- t(gex_res[gene, rownames(data_df)])
  # Define variables for regression
  X <- data_df[['force_metric']]
  Y <- data_df[['gene']]
  
  #gSEM
  #Identify spatial patterns
  f_X_hat<-gam(force_metric~s(centroid_x,centroid_y,k=k_sp,fx=model_fx),data=data_df,method=method)$fitted.values
  r_X<-X-f_X_hat
  f_Y_hat<-gam(gene~s(centroid_x,centroid_y,k=k_sp,fx=model_fx),data=data_df,method=method)$fitted.values
  r_Y<-Y-f_Y_hat
  #Fit linear model to residuals
  mod<-lm(r_Y~r_X)
  
  return(list(mod, as.numeric(r_X), as.numeric(r_Y)))
}

parallel_gsem_regression <- function(genes, tensionmap_res, gex_res, k_sp=300, model_fx=F, method='REML', ncores=6){
  # Parallelised regression for multiple genes
  registerDoParallel(ncores)
    
  reg_res <- foreach(i=icount(length(genes)), .combine=rbind) %dopar% {
    gene <- genes[i]
    
    res <- data.frame(gene=character(0), metric=character(0), beta=numeric(0), pval=numeric(0), stat=numeric(0))
    for (force_metric in c('pressure', 'stresstensor_magnitude')){
      reg_res <- do_gsem_regression(gene, force_metric, tensionmap_res, gex_res, k_sp, model_fx, method)
    
      mod <- reg_res[[1]]
      res <- rbind(res, data.frame(gene=gene, metric=force_metric, beta=mod$coefficients[2], pval=summary(mod)$coefficients[2,4], stat=summary(mod)$coefficients[2,3]))
    }
    res
  }
  
  return(reg_res)
}


makebarplot <- function(data, title, numpoints){
  # Function for pretty bar plots
  data[['log10_pvalue']] <- -log10(data$pvalue)
  data_plot_df <- dplyr::arrange(data, desc(log10_pvalue))[1:numpoints,]
  data_plot_df$Description <- apply(str_split_fixed(data_plot_df$Description, '_', Inf), 1, paste, collapse=' ') %>% tolower()
  data_plot_df$Description <- str_wrap(data_plot_df$Description, width = 30)
  data_plot_df$Description <- factor(data_plot_df$Description, levels = rev(data_plot_df$Description))
  
  plot <- ggplot(data_plot_df, aes(x=log10_pvalue, y=Description, fill=Count)) +
    geom_bar(stat='identity') +
    xlab(bquote(~-Log[10] ~ 'p value')) +
    ylab(NULL) +
    ggtitle(title) +
    theme_bw() +
    scale_fill_gradientn(colours = jdb_palette('solar_rojos')[4:length(jdb_palette('solar_rojos'))]) + 
    theme(plot.title = element_text(hjust = 0.5, size=17, face="bold"), axis.text.y = element_text(size=14),
          panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
  return(plot)
}


do_linear_regresssion <- function(gene, force_metric, tensionmap_res, gex_res){
  data_df <- tensionmap_res[c('centroid_x', 'centroid_y', force_metric)]
  data_df[['force_metric']] <- log(data_df[[force_metric]])
  
  data_df[['gene']] <- t(gex_res[gene, rownames(data_df)])
  # Define variables for regression
  X <- data_df[['force_metric']]
  Y <- data_df[['gene']]
  
  mod<-lm(Y~X)
  return(list(mod, as.numeric(X), as.numeric(Y)))
}


parallel_linear_regression <- function(genes, tensionmap_res, gex_res, ncores=6){
  registerDoParallel(ncores)
  # Parallelised regression for multiple genes
    
  reg_res <- foreach(i=icount(length(genes)), .combine=rbind) %dopar% {
    gene <- genes[i]
    
    res <- data.frame(gene=character(0), metric=character(0), beta=numeric(0), pval=numeric(0), stat=numeric(0))
    for (force_metric in c('pressure', 'stresstensor_magnitude')){
      reg_res <- do_linear_regresssion(gene, force_metric, tensionmap_res, gex_res)
    
      mod <- reg_res[[1]]
      res <- rbind(res, data.frame(gene=gene, metric=force_metric, beta=mod$coefficients[2], pval=summary(mod)$coefficients[2,4], stat=summary(mod)$coefficients[2,3]))
    }
    res
  }
  
  return(reg_res)
}