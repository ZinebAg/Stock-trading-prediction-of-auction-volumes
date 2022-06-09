# here example following train, test, no crossvalidation is executed

# packages and needed installations


install.packages(c("readr", "data.tabl", "refund","qdapTools", "dplyr","mgcv", "visreg", "ggplot2", "tidymv")) # takes a bit less 
install.packages('R.utils')
install.packages("refund") # takes just under 10 minutes
install.packages("tidymv")
library(refund)
library(readr)
library(data.table)
library(qdapTools)
library(dplyr)
library(mgcv)
library(visreg)
library(ggplot2)
theme_set(theme_bw())
install.packages
library(tidymv)
library(R.utils)

# loading the data
target <- read_csv("/content/drive/MyDrive/Master Semester Project/data/output_training_IxKGwDV.csv")
train <- fread("/content/drive/MyDrive/Master Semester Project/data/data_RF_KNN_2.csv.gz")
train$target <- target$target

names(train) = gsub(pattern = "_", replacement = "", x = names(train))

test<-fread("/content/drive/MyDrive/Master Semester Project/data/test_RF_KNN_2.csv.gz")
names(test) = gsub(pattern = "_", replacement = "", x = names(test))
test<-cbind( test, target=0)
# double checking no data is missing
sum(is.na(train))
sum(is.na(test))


# function for writing the data into the right format
prepare_data <- function(train_DS, test_DS, Nbr_train, Nbr_test) {
  
  #train
  Sample_tr<-sample_n(train_DS, Nbr_train)
  Sample_tr$NLV<-as.data.frame(scale(Sample_tr$NLV))
  Sample_tr$LS<-as.data.frame(scale(Sample_tr$LS))
  
  #test
  Sample_te<-sample_n(test_DS, Nbr_test)
  Sample_te$NLV<-as.data.frame(scale(Sample_te$NLV))
  Sample_te$LS<-as.data.frame(scale(Sample_te$LS))
  
  #join them
  total <- rbind(Sample_tr, Sample_te)
  
  #put day and pid as factors
  total$day=factor(total$day)
  total$pid=factor(total$pid)
  return(total)  
  
}



# preparing data
n_train<-10000
n_test<-nrow(test)
data_total<-prepare_data(train, test,n_train,n_test)

index_test_start<-n_train+1
index_test_end<-n_train+n_test


colnames(data_total)
dim(data_total)

numerical_covariate_scaled<-data_total[,c(3:128)] # basically the same
dim(numerical_covariate_scaled) 
colnames(numerical_covariate_scaled)

colnames(numerical_covariate_scaled[,63:123])
relvol<-numerical_covariate_scaled[,63:123]
relvol<-as.matrix(relvol)
dim(relvol)

colnames(numerical_covariate_scaled[,2:62])
absret<-numerical_covariate_scaled[,2:62]
absret<-as.matrix(absret)

data.fit <- data.frame(pid=data_total$pid,
                       day=data_total$day,
                       NLV=data_total$NLV,
                       LS=data_total$LS,
                       target=data_total$target)
data.fit$relvol=as.matrix(relvol)
data.fit$absret=as.matrix(absret)

#taking out the test dataset
nd <- data.fit[index_test_start:index_test_end,]
s <- 1:61
s_mat <- NULL
total_rows<-n_train+n_test
s_mat <- rbind(s_mat, s)
while (nrow((s_mat)) <nrow(data.fit) ){
  s_mat <- rbind(s_mat, s_mat)
  print("*****************")
  print(nrow((s_mat)))
  print(nrow(data.fit))
}
s_mat_good<-s_mat[1:nrow(data.fit),]
nrow(s_mat_good) == nrow(data.fit) 
data.fit$s_mat <- as.matrix(s_mat_good)
nrow(data.fit)



# fitting

start_time <- Sys.time()
fit_fgam_all_0 <- fgam(target ~ 1+ s(pid,bs = 're')+s(day,bs = 're')+NLV+LS+lf(relvol)+ lf(absret),data=data.fit[1:n_train,])
end_time <- Sys.time()

par(mfrow=c(1,2))
plot(fit_fgam_all_0)
summary(fit_fgam_all_0)

# predicting
index_test_start<-n_train+1
index_test_end<-n_train+n_test
nd <- data.fit[index_test_start:index_test_end,]
predictions<-predict(fit_fgam_all_0, newdata = nd,newdata.guaranteed=TRUE)