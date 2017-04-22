theft <- read.csv("D:\\Issaugota_20160823\\Desktop\\Irina\\MASTER STUDIES\\Saint Peters Uni\\DS-610 Big Data Analytics\\Project\\Merged333.csv", sep=",", header = TRUE)
View(theft)

set.seed(100)
trainingRowIndex <- sample(1:nrow(theft), 0.8*nrow(theft))
training_set <- theft[trainingRowIndex, ]
test_set <- theft[-trainingRowIndex, ]
#training_set <- theft[c(TRUE, FALSE),]#odd rows
#test_set <- theft[!c(TRUE, FALSE),]#even rows
View(training_set)
View(test_set)

#test for outliers in data
boxplot(training_set$Theft, data=theft, main="Boxplot of Thefts")
boxplot(training_set$Subway, data=theft, main="Boxplot of Subway entrances")
boxplot(training_set$Rest, data=theft, main="Boxplot of Restaurants")
boxplot(training_set$Graf, data=theft, main="Boxplot of Graffiti")

#checking quantitative data for normality
hist(training_set$Theft,prob=T,breaks=13, xlab="Thefts", main="Histogram of Thefts")
lines(density(training_set$Theft),col="red")
qqnorm(training_set$Theft, main="Normal Q-Q Plot of Thefts")
qqline(training_set$Theft)
TheftX <- log(training_set$Theft+1)

hist(training_set$Subway,prob=T,breaks=13, xlab="Subway entrances", main="Histogram of Subways")
lines(density(training_set$Subway),col="red")
qqnorm(training_set$Subway, main="Normal Q-Q Plot of Subways")
qqline(training_set$Subway)
SubwayX <- log(training_set$Subway+1)

hist(training_set$Rest,prob=T,breaks=13, xlab="Restaurants", main="Histogram of Restaurants")
lines(density(training_set$Rest),col="red")
qqnorm(training_set$Rest, main="Normal Q-Q Plot of Restaurants")
qqline(training_set$Rest)
RestX <- log(training_set$Rest+1)

hist(training_set$Graf,prob=T,breaks=13, xlab="Graffiti", main="Histogram of Graffiti")
lines(density(training_set$Graf),col="red")
qqnorm(training_set$Graf, main="Normal Q-Q Plot of Graffiti")
qqline(training_set$Graf)
GrafX <- log(training_set$Graf+1)

hist(training_set$rating_b,prob=T,breaks=13, xlab="Street pavement rating", main="Histogram of Pavement Rating")
lines(density(training_set$rating_b),col="red")
qqnorm(training_set$rating_b, main="Normal Q-Q Plot of Pavement Rating")
qqline(training_set$rating_b)


#Pearson describes relation in terms of linearity
cor.test(training_set$Theft,training_set$Subway, method="pearson")
plot(training_set$Theft,training_set$Subway)
abline(lm(training_set$Theft~training_set$Subway), col="red")

cor.test(training_set$Theft,training_set$Rest, method="pearson")
plot(training_set$Theft,training_set$Rest)
abline(lm(training_set$Theft~training_set$Rest), col="red")

cor.test(training_set$Theft,training_set$Graf, method="pearson")
plot(training_set$Theft,training_set$Graf)
abline(lm(training_set$Theft~training_set$Graf), col="red")

cor.test(training_set$Theft,training_set$rating_b, method="pearson")
plot(training_set$Theft,training_set$rating_b)
abline(lm(training_set$Theft~training_set$rating_b), col="red")

cor.test(training_set$Theft,training_set$width, method="pearson")
plot(training_set$Theft,training_set$width)
abline(lm(training_set$Theft~training_set$width), col="red")

cor.test(training_set$Theft,training_set$length, method="pearson")
plot(training_set$Theft,training_set$length)
abline(lm(training_set$Theft~training_set$length), col="red")

data <- training_set[c(7,8,9,10,11)] # get data
data.r <- abs(cor(data)) # get correlations
data.col <- dmat.color(data.r) # get colors
# reorder variables so those with highest correlation
# are closest to the diagonal
data.o <- order.single(data.r)
cpairs(data, data.o, panel.colors=data.col, gap=.5,#gclus package
       main="Variables Ordered and Colored by Correlation" ) 

#Multicollinearity test
cor.test(training_set$Subway,training_set$Rest,method="pearson")
cor.test(training_set$Subway,training_set$Graf,method="pearson")
cor.test(training_set$Graf,training_set$Rest,method="pearson")

#logistic regression
logit <- glm(formula = Theft ~ rating_b, data=training_set)
backward <- step(logit,direction="backward")
summary(logit)
anova(object=logit, test="Chisq")
BIC(logit) #Higher numbers of Residual deviance indicates bad fit.  Deviance is a measure of goodness of fit of a model. 
pR2(logit) # pscl Library. Higher values indicating better model fit
#r.squaredLR
res <- residuals(logit)
hist(res,prob=T,breaks=9, xlab="Residuals", main="Histogram of the residuals" )
lines(density(res),col="red")
qqnorm(res)
qqline(res)

#linear regression
modelS <- lm(formula = Theft ~ Subway, data=training_set)
abline(modelS)
summary(modelS)

modelR <- lm(formula = Theft ~ Rest, data=training_set)
abline(modelR)
summary(modelR)

modelG <- lm(formula = Theft ~ Graf, data=training_set)
plot(modelG)
abline(modelG)
summary(modelG)

modelSG <- lm(formula = Theft ~ Subway+Graf, data=training_set)
summary(modelSG)
plot(modelSG)
anova(modelSG)
confint(modelSG,conf.level=0.95)
resSG <- residuals(modelSG)
#Assumption 1 "Mean of residuals is zero": in our case assumption is met with 1.988e-15
mean(resSG)

#Assumption 2 "Homoscedasticity of residuals or equal variance" by cheking Res vs Fitted
plot(modelSG) #Assumption is met, because the red line is not flat. 
#Though, the variation is not constant here, because we see pattern on the graph (all data does not look like a cloud).

#Assumption 3 "No autocorrelation of residuals" - not met
acf(resSG)
#runs.test(theft$resSG)
dwtest(modelSG) #DW = 1.2795, p-value < 2.2e-16 > reject H0 and accept Ha: autocorrelation is greater than 0
#Residuals are autocorrelated

#Assumption 4 "The X variables and residuals are uncorrelated" - met
cor.test(resSG,training_set$Subway) #1.055004e-15 with p=0, corr is equal to 0
cor.test(resSG,training_set$Graf) #5.818904e-17 with p=1, corr is equal to 0

#Assumtpion 5 "No multicollinearity" - met
vif(modelSG) #for both Subway and Graf VIF=1. Assumption met.
#If the VIF of a variable is high, it means the information in that variable is already explained by 
#other X variables present in the given model, which means, more redundant is that variable. 
#So, lower the VIF (<2) the better.

#Assumption 6 "Normality of residuals" - not met
hist(resSG,prob=T,breaks=14, xlab="Residuals", main="Histogram of the residuals" )
qqnorm(resSG)
qqline(resSG)
gvlma(modelSG)

####MODEL
training_set <- cbind(res,training_set,TheftX,SubwayX,GrafX,RestX)
View(training_set)
model <- lm(formula = TheftX ~ SubwayX+GrafX+RestX, data=training_set)
summary(model)
plot(model)
anova(model)
confint(model,conf.level=0.95)
res <- residuals(model)
#Assumption 1 "Mean of residuals is zero": in our case assumption is met with -2.278831e-16
mean(res)

#Assumption 2 "Homoscedasticity of residuals or equal variance" by cheking Res vs Fitted
plot(model) #Assumption is met, because the red line is almost flat. 
#Though, the variation is not constant here, because we see pattern on the graph (all data does not look like a cloud).

#Assumption 3 "No autocorrelation of residuals" - not met
acf(res)
#runs.test(theft$resSG)
dwtest(model) #lmtest package. 
#DW = 2.0326, p-value < 0.9717 > accept H0: autocorrelation is not greater than 0
#Residuals are autocorrelated

#Assumption 4 "The X variables and residuals are uncorrelated" - met
cor.test(res,training_set$SubwayX) #2.440349e-16 with p=1, corr is equal to 0
cor.test(res,training_set$GrafX) #1.01497e-15 with p=1, corr is equal to 0
cor.test(res,training_set$RestX) #-2.225019e-16 with p=1, corr is equal to 0

#Assumtpion 5 "No multicollinearity" - met
vif(model) #for both Subway, Res and Graf VIF=1. Assumption met.
#If the VIF of a variable is high, it means the information in that variable is already explained by 
#other X variables present in the given model, which means, more redundant is that variable. 
#So, lower the VIF (<2) the better.

#Assumption 6 "Normality of residuals" - not met
hist(res,prob=T,breaks=13, xlab="Residuals", main="Histogram of the residuals" )
qqnorm(res)
qqline(res)
gvlma(model) #gvlma package

#checking regression model
PredictedTheft <- coef(model)[1]+coef(model)[2]*test_set$Subway+coef(model)[3]*test_set$Graf+coef(model)[4]*test_set$Rest
resTest <- test_set$Theft-PredictedTheft
mean(resTest) #0.04692426
test_set <- cbind(test_set,resTest,PredictedTheft)
#plot actual thefts vs. predicted thefts
plot(test_set$Theft,test_set$PredictedTheft,xlab ="Actual number of thefts", ylab ="Predicted number of thefts",main="Actual vs. Predicted values")
abline(lm(test_set$Theft~test_set$PredictedTheft),col="red")
cor(test_set$Theft,test_set$PredictedTheft)#0.5799567
#plot predicted thefts vs. residuals
plot(test_set$PredictedTheft,test_set$resTest)
abline(lm(test_set$PredictedTheft~test_set$resTest),col="red")
AIC(model)#1360.08 the smaller the better
min_max_accuracy <- mean(min(test_set$Theft,test_set$PredictedTheft)/max(test_set$Theft,test_set$PredictedTheft))
min_max_accuracy#0
hist(test_set$resTest, prob=TRUE)
lines(density(test_set$resTest),col="red")
hist(res, prob=TRUE)
lines(density(test_set$resTest),col="red")
