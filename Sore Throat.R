library(gridExtra)
library(devtools)
install_github("Dasonk/knnflex")
library(knnflex)
library(mice)
library(randomForest)
library(MASS)
library(mlbench)
library(caret)
library(glmnet)
library(pROC)
library(parallel)
library(gam)
library(rpart)
library(class)
library(xtable)
library(ggplot2)
library(FactoMineR)
core.register=4


dat<-read.csv("~/Desktop/PhD/BIOS624/A1/sore_throat.csv")
attach(dat)

head(dat)


result<-result
id<-id
conf<-cbind(sex1-2, age)

colnames(conf)<-c("sex", "age")

symptom<-cbind( day1,  early,   hxf1,   nose1,   ear1, cough1,   gland1,   head1,   ache1,  
                gi1,   skin1, sinus1,   expose1)  ### what is rx1? should include in symptom or sign?
#13
colnames(symptom)<-cbind("days_sick","early","fever","runny_nose","earache","cough","gland",
                         "headache","general_ache","ginausea", "skinrash","facepain",
                         "exposure")
temperature<-ifelse(is.na(temp),0,temp)
temp_not_record<-as.factor(1-temprcrd)

sig<-cbind( temp_not_record,   temperature,   red1,   tonsil1,
            exud1,   phary1,   petech1, cerv1,   tm1,  lung1,   scarlet1)

#11

colnames(sig)<-c("temp_record","temperature", "red_throat", "swollen_tonsils","tonsillar_exudate",
                 "pharynx_exudate","palatal_petechiae", "cervaden",  "tympanic", "lung_findings", 
                 "scarlet_rash")

behavior<-cbind(swab1,dx,estimate,prescrib)
colnames(behavior)<-c("throat_swab", "diagnosis", "estimate likelihood", "prescrib")



######  missing ########

data<-cbind(conf,symptom,sig,behavior)
complete<- data[complete.cases(data), ]

###### seperate train/test ####

complete<-as.data.frame(cbind(complete,dat$result[complete.cases(data)]))
colnames(complete)[31]<-"result"

size <- floor(0.7 * nrow(complete))
## set the seed to make your partition reproducible
set.seed(108573)

## seperate training/testing data (validation done inside ML algorithm)
train_ind <- sample(seq_len(nrow(complete)), size = size)
seq1<-which(colnames(complete)%in%c('age','sex','days_sick','fever','runny_nose','cough','general_ache','exposure','result'))
train.stage1 <- complete[train_ind, seq1]
test.stage1 <- complete[-train_ind, seq1]


seq2<-which(colnames(complete)%in% c("temp_record","temperature", "red_throat", "swollen_tonsils","tonsillar_exudate",
                                     "pharynx_exudate","palatal_petechiae", "cervaden",  "tympanic", "lung_findings", 
                                     "scarlet_rash"))

train.stage2<- complete[train_ind, c(seq1, seq2)]
test.stage2 <- complete[-train_ind, c(seq1,seq2)]
  

#stage 3??
#######  glm #######

fit.glm<-glm(result~.,data=train.stage1, family="binomial")
roc.glm.train<-roc(train.stage1$result, as.numeric(predict(fit.glm, train.stage1, type="response")))
roc.glm<-roc(test.stage1$result, as.numeric(predict(fit.glm,test.stage1 , type="response")))

c(roc.glm.train[9],roc.glm[9])

table.glm<-table(test$result, ifelse(predict(fit.glm, test, type="response")>.5,1,0))
table.glm
error.glm<-(table.glm[1,2]+table.glm[2,1])/length(test.stage1$result)
error.glm




#######  glm-elastic net penalization #######

#not very meaningful to run penalization due to similar performance in train/test of logistic

a <- seq(0.1, 0.9, 0.05)
search <- foreach(i = a, .combine = rbind) %dopar% {
  i=0.1
  cv <- cv.glmnet(as.matrix(train.stage1[,-9]), train.stage1$result, family = "binomial", nfold = 10, 
                  type.measure = "deviance", paralle = TRUE, alpha = i)
  data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.1se], lambda.1se = cv$lambda.1se, alpha = i)
}
cv3 <- search[search$cvm == min(search$cvm), ]
fit.elnet <- glmnet(as.matrix(train.stage1[,-9]), train.stage1$result, 
                    family = "binomial", lambda = cv3$lambda.1se, alpha = cv3$alpha)
coefficients(fit.elnet)

roc.elnet.train<-roc(train.stage1$result, as.numeric(predict(fit.elnet, as.matrix(train.stage1[,-9]), type="response")))
roc.elnet<-roc(test.stage1$result, as.numeric(predict(fit.elnet, as.matrix(test.stage1[,-9]), type = "response")) )
c(roc.elnet.train[9], roc.elnet[9])

table.elnet<-table(test.stage1$result, ifelse(predict(fit.elnet, as.matrix(test.stage1[,-9]), type = "response")>.5,1,0))
error.elnet<-table.elnet[2,]/length(test$result)
error.elnet
table.elnet

####### glm-smooth ##########

colnames(train)
auc<-NULL
for (i in 5:15){
  for (j in 2:5){
    fit.gam<-gam(result~s(age,i)+ sex+s(days_sick,j)+early+fever+runny_nose+earache+cough
                 + gland+headache+general_ache+ginausea+skinrash+facepain+exposure, family = binomial, train)
    auc<-c(auc,roc(train$result, as.numeric(predict(fit.gam, train[,-16], type="response")))$auc)
  }
}

#grid search: days_sick df=3, age df as large as possible,(choose df=2)

fit.gam<-gam(result~s(age,3)+ sex+s(days_sick,3)+fever+runny_nose+general_ache
             +exposure, family = binomial,data= train.stage1)
roc.gam.train<-roc(train.stage1$result, as.numeric(predict(fit.gam, train.stage1[,-9], type="response")))
roc.gam<-roc(test.stage1$result,as.numeric(predict(fit.gam,test.stage1[,-9],type="response")))

c(roc.gam.train[9], roc.gam[9])

table.gam1<-table(test$result,ifelse(predict(fit.gam,test[,-16],type="response")>.5,1,0))
table.gam1
error.gam1<-(table.gam1[2,1]+table.gam1[1,2])/length(test.stage1$result)
error.gam1



####### decision tree #######
par(mfrow=c(1,1))

fit.tree <- rpart(result ~ ., control=rpart.control(minsplit=10, maxdepth=5),
                  method="class", data=train.stage1)
plot(fit.tree, uniform=TRUE)
text(fit.tree, use.n=TRUE, all=TRUE, cex=0.7)

roc.tree.train<-roc(train.stage1$result, as.numeric(predict(fit.tree, train.stage1[,-9], type="class")))

roc.tree<-roc(test.stage1$result,as.numeric(predict(fit.tree,test.stage1[,-9], type="class")))
c(roc.tree.train[9], roc.tree[9])

table.tree<-table(test$result,predict(fit.tree,test, type="class"))
table.tree
error.tree<-(table.tree[1,2]+table.tree[2,1])/length(test$result)
error.tree

###################################################
set.seed(50197)
complete_shuffle<-complete[sample(nrow(complete)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(complete_shuffle)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
auc<-NULL
for (j in 3:10){
  res<-NULL
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- complete_shuffle[testIndexes, ]
    trainData <- complete_shuffle[-testIndexes, ]
    fit.knn<-knn.predict(-testIndexes, testIndexes, complete_shuffle$result, 
                         knn.dist(complete_shuffle[,seq1[-9]]), 
                         k=j, agg.meth= "majority", ties.meth="max")
    res<-c(res,roc(testData[,9], as.numeric(fit.knn))[9])
  }
  auc<-c(auc, mean(unlist(res)))
}

## optimal k=5

fit.knn<-knn.predict(train_ind, -train_ind, complete$result, knn.dist(complete[,seq1[-9]]), k=5, agg.meth= "majority", ties.meth="max")

roc.knn.train<-"NA"

roc.knn<-roc(test.stage1$result, as.numeric(fit.knn))
roc.knn
table.knn<-table(test$result, fit.knn)
error.knn<-(table.knn[1,2]+table.knn[2,1])/length(test$result)
error.knn
###################################################

#searching for (ntree), default mtry=sqrt(p)
train_rf<-randomForest(result ~ . , data = train.stage1)                                    
plot(train_rf)
#n.tree=20?
set.seed(124090)
fit.rf<-randomForest(result ~ . , data = train.stage1, mtry=4, ntree=20)  
roc.rf.train<-roc(train.stage1$result, as.numeric(predict(fit.rf, train.stage1[,-9], type="response")))
roc.rf<-roc(test.stage1$result, as.numeric(predict(fit.rf,test.stage1[,-9],type="response")))

c(roc.rf.train[9], roc.rf[9])

table.rf1<-table(test$result,ifelse(predict(fit.rf,test[,-16],type="response")>.5,1,0)) 
error.rf1<-(table.rf1[1,2]+table.rf1[2,1])/length(test$result)
error.rf1
plot.roc(roc.rf)

#######################


roc.train<-cbind(roc.glm.train, roc.elnet.train , roc.gam.train, 
                 roc.tree.train ,roc.knn.train, roc.rf.train)

roc.test<-cbind(roc.glm, roc.elnet , roc.gam , roc.tree,roc.knn, roc.rf)
res<-rbind(roc.train[9,],roc.test[9,])
colnames(res)<-c("logistic","elastic net", "gam", "tree", "knn", "random forest")
rownames(res)<-c("training set AUC", "test set AUC")
xtable(res)

error<-cbind(error.glm,error.elnet, error.gam1, error.tree, error.knn, error.rf1)
xtable(error*100)
ggroc(list(glm=roc.glm, elnet=roc.elnet,gam=roc.gam,knn=roc.knn,
           decision.tree=roc.tree,random.forest=roc.rf))+
  theme_minimal()+
  geom_abline(slope = 1, intercept = 1,linetype='dashed', alpha=0.8)






##### stage 2, reduce dimension ####

stg2<-apply(complete[,c(seq1[-c(2,3,9)],seq2[-c(1,2)])], FUN=as.factor, MARGIN=2)
colnames(complete[,seq1])
cats = apply(stg2, 2, function(x) nlevels(as.factor(x)))

cats
mca1 = MCA(stg2, ncp=11, graph = FALSE)
mca1_vars_df = data.frame(mca1$var$coord, Variable = rep(names(cats), cats))
# data frame with observation coordinates
mca1_obs_df = data.frame(mca1$ind$coord)
ggplot(data = mca1_obs_df, aes(x = Dim.1, y = Dim.2)) +
  geom_hline(yintercept = 0, colour = "gray70") +
  geom_vline(xintercept = 0, colour = "gray70") +
  geom_point(colour = "gray50", alpha = 0.7) +
  geom_density2d(colour = "gray80") +
  geom_text(data = mca1_vars_df, 
            aes(x = Dim.1, y = Dim.2, 
                label = rownames(mca1_vars_df), colour = Variable)) +
  ggtitle("MCA plot of binary variables") +
  scale_colour_discrete(name = "Variable")


mca.dat<-mca1$ind$coord
colnames(mca.dat)<-c("dim1", "dim2","dim3","dim4","dim5","dim6",
                     "dim7","dim8","dim9","dim10","dim11")
complete2<-cbind(complete[,c(seq1[c(2,3,9)],seq2[c(1,2)])], mca.dat)
head(complete2)

size <- floor(0.7 * nrow(complete2))

## set the seed to make your partition reproducible
set.seed(112048)

## seperate training/testing data (validation done inside ML algorithm)
train_ind2 <- sample(seq_len(nrow(complete2)), size = size)

train2 <- complete2[train_ind2, ]
test2 <- complete2[-train_ind2, ]



###### choose glm, gam, and rf as candidate and proceed further #########

colnames(train2)


fit.glm2<-glm(result~.,data=train2, family="binomial")
roc.glm.train2<-roc(train2$result, as.numeric(predict(fit.glm, train2, type="response")))
roc.glm2<-roc(test2$result, as.numeric(predict(fit.glm,test2 , type="response")))

c(roc.glm.train2[9],roc.glm2[9])

table.glm2<-table(test2$result, ifelse(predict(fit.glm2, test2, type="response")>.5,1,0))
table.glm2
error.glm2<-(table.glm2[1,2]+table.glm2[2,1])/length(test2$result)
error.glm2





### choose temperature df=10
colnames(complete2)
fit.gam2<-gam( result~ s(age,3) + s(days_sick,3)+ temp_record + s(temperature,3)+dim1       
              + dim2+ dim3 +dim4 +dim5 +dim6 +dim7+ dim8+dim9+dim10+dim11 
              , family = binomial, data=train2)

roc.gam.train2<-roc(train2$result, as.numeric(predict(fit.gam2, train2, type="response")))
roc.gam2<-roc(test2$result, as.numeric(predict(fit.gam2,test2,type="response")))
c(roc.gam.train2[9], roc.gam2[9])
table.gam2<-table(test2$result2, ifelse(predict(fit.gam2,test2,type="response")>0.5,1,0))
table.gam2
error.gam2<-(table.gam2[1,2]+table.gam2[2,1])/length(test2$result)
error.gam2


######### random forest #################



set.seed(32509)
fit.rf2<-randomForest(result ~ . , data = train2, mtry=6, ntree=50, maxnodes=20)  

roc.rf2.train<-roc(train2$result, as.numeric(predict(fit.rf2, train2, type="response")))
roc.rf2.train$auc
roc.rf2<-roc(test2$result, as.numeric(predict(fit.rf2,test2,type="response")))
roc.rf2$auc

table.rf2<-table(test2$result, ifelse(predict(fit.rf2,test2,type="response")>.5,1,0))
table.rf2

(table.rf2[1,2]+table.rf2[2,1])/length(test2$result)





########  add doctor behavior to complete2 #########


complete3<-cbind(complete2, complete[,colnames(complete)%in%c("throat_swab" ,   "diagnosis", "estimate likelihood","prescrib")])
colnames(complete3)[19]<-"likelihood"
dim(complete3)

size <- floor(0.7 * nrow(complete3))

## set the seed to make your partition reproducible
set.seed(508912)

## seperate training/testing data (validation done inside ML algorithm)
train_ind3 <- sample(seq_len(nrow(complete3)), size = size)

train3 <- complete3[train_ind3, ]
test3 <- complete3[-train_ind3, ]

###### choose glm, gam, and rf as candidate and proceed further #########



fit.glm3<-glm(result~.,data=train3, family="binomial")
roc.glm.train3<-roc(train3$result, as.numeric(predict(fit.glm, train3, type="response")))
roc.glm3<-roc(test3$result, as.numeric(predict(fit.glm,test3, type="response")))

c(roc.glm.train3[9],roc.glm3[9])

table.glm3<-table(test3$result, ifelse(predict(fit.glm, test3, type="response")>.5,1,0))
table.glm3
error.glm3<-(table.glm[1,2]+table.glm[2,1])/length(test2$result)
error.glm3





### choose temperature df=3


fit.gam3<-gam( result~ s(age,3) + s(days_sick,3)+ temp_record + s(temperature,3)+dim1       
               + dim2+ dim3 +dim4 +dim5 +dim6 +dim7+ dim8+dim9+dim10+dim11 
               +  throat_swab + diagnosis +prescrib+likelihood
               , family = binomial, data=train3)

roc.gam.train3<-roc(train3$result, as.numeric(predict(fit.gam3, train3, type="response")))
roc.gam3<-roc(test3$result, as.numeric(predict(fit.gam3,test3,type="response")))
c(roc.gam.train3[9], roc.gam3[9])
table.gam3<-table(test3$result, ifelse(predict(fit.gam3,test3,type="response")>0.5,1,0))
table.gam3
error.gam3<-(table.gam3[1,2]+table.gam3[2,1])/length(test3$result)
error.gam3


######### random forest #################



set.seed(32509)
fit.rf3<-randomForest(result ~ . , data = train3, mtry=7, ntree=50, maxnodes=20)  

roc.rf.train3<-roc(train3$result, as.numeric(predict(fit.rf3, train3, type="response")))
roc.rf.train3$auc
roc.rf3<-roc(test3$result, as.numeric(predict(fit.rf3,test3,type="response")))
roc.rf3$auc

table.rf3<-table(test3$result, ifelse(predict(fit.rf3,test3,type="response")>.5,1,0))
table.rf3

error.rf3<-(table.rf3[1,2]+table.rf3[2,1])/length(test3$result)





###bayes rate##
bayes1<-sum(test.stage1$result)/length(test.stage1$result)


bayes2<-sum(test2$result)/length(test2$result)

bayes3<-sum(test3$result)/length(test3$result)

cbind(bayes1, bayes2, bayes3)



######### how accuracy is the doctor? #################
roc(dat$prescrib,dat$result)

newpres<-dat$prescrib[as.numeric(dat$prescrib)!=1]
newres<-dat$result[as.numeric(dat$prescrib)!=1]
summary(cbind(newpres-2,dat$result))
sum(complete$result)
table(complete$prescrib-2, complete$result)
roc.doctor<-roc(complete$prescrib-2, complete$result)
roc.doctor

complete$result[-train_ind]
complete$prescrib[-train_ind]-2

ifelse(as.numeric(predict(fit.glm, test.stage1, type="response"))>0.5,1,0)

overuse<-as.data.frame(cbind(complete$result[-train_ind],complete$prescrib[-train_ind]-2,
               ifelse(as.numeric(predict(fit.glm, test.stage1, type="response"))>0.5,1,0)))
colnames(overuse)<-c("infection", "doctor","model")

doc.is.wrong<-overuse[overuse$infection!=overuse$doctor,]
sum(doc.is.wrong$model==doc.is.wrong$infection)/dim(doc.is.wrong)[1]
doc.is.correct<-overuse[overuse$infection==overuse$doctor,]
sum(doc.is.correct$infection!=doc.is.correct$model)/dim(doc.is.correct)[1]



c(FP.doc,FN.doc)
ggroc(list(doctor=roc.doctor, predictive_model=roc.rf))+
  theme_minimal()+
  geom_abline(slope = 1, intercept = 1,linetype='dashed', alpha=0.8)+
  ggtitle("ROC curve of Doctor's predict and Model's predict")
## hmmm, not too good


######### prevelency matters right? ############

table(train$result) ##  I will just predict 0
roc(test$result, rep(0,length(test$result)))




######### plot compare auc ########

p0<-ggroc(list(glm.symp=roc.glm, glm.sign=roc.glm2, glm.phys=roc.glm3))+
  theme_minimal()+
  geom_abline(slope = 1, intercept = 1,linetype='dashed', alpha=0.8)+
  ggtitle("Logistic")

p1<-ggroc(list(gam.symp=roc.gam, gam.sign=roc.gam2, gam.phys=roc.gam3))+
  theme_minimal()+
  geom_abline(slope = 1, intercept = 1,linetype='dashed', alpha=0.8)+
  ggtitle("GAM")

p2<-ggroc(list(rf.symp=roc.rf, rf.sign=roc.rf2, rf.phys=roc.rf3))+
  theme_minimal()+
  geom_abline(slope = 1, intercept = 1,linetype='dashed', alpha=0.8)+
  ggtitle("Random Forest")

grid.arrange(p0,p1,p2,nrow=1)
