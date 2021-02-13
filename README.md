# Predictive-Analysis-of-Income-Generation-in-Ecommerce-Industry
# PROJECT OVERVIEW
 Using Linear Regression to advise a Ecommerce company that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want. The company is trying to decide whether to focus their efforts on their mobile app experience or their website. Using Linear Regression to study the correlations between the different factors and how they affect the how much the customer spends.
 
 OBJECTIVES
 
The objective of the project is that, as the company is aiming to focus their effort on taking orders of clothing from their  mobile app  or website is feasible. So the project here Predicts the Yearly amount Spent by the customers on ordering Clothing on Mobile Application or on Website using Linear Regression model. And also finding the correlation between each of  their independent variables. And finally predicting whether the company is good to go with mobile app or the website.

INTRODUCTION

Machine Learning is a sub-area of artificial intelligence, whereby the term refers to the ability of IT systems to independently find solutions to problems by recognizing patterns in databases. In other words: Machine Learning enables IT systems to recognize patterns on the basis of existing algorithms and datasets and to develop adequate solution concepts. Therefore, in Machine Learning, artificial knowledge is generated on the basis of experience. In order to enable the software to independently generate solutions, the prior action of people is necessary. For example, the required algorithms and data must be fed into the systems in advance and the respective analysis rules for the recognition of patterns in the data stock must be defined.
Linear regression is a basic and commonly used type of predictive analysis.  The overall idea of regression is to examine two things: (1) does a set of predictor variables do a good job in predicting an outcome (dependent) variable?  (2) Which variables in particular are significant predictors of the outcome variable, and in what way do they–indicated by the magnitude and sign of the beta estimates–impact the outcome variable?  These regression estimates are used to explain the relationship between one dependent variable and one or more independent variables.

THE 7 STEPS IN MACHINE LEARNING

STEP 1: GATHERING DATA

The step of gathering data is the foundation of the machine learning process. Here the quantity and quality of your data dictates how accurate your model is. The outcome of this step is generally a representation of data which we will use for training.

STEP 2: DATA PREPARATION

Once we have gathered the data for the two features, our next step would be to prepare data for further steps. A key focus of this stage is to recognize and minimize any potential biases in our data sets i.e clean the data which is not required like remove duplication, correct errors,deal with missing values etc.Another major component of data preparation is breaking down the data sets into 2 parts. The larger part (~80%) would be used for training the model while the smaller part (~20%) is used for evaluation purposes. This is important because using the same data sets for both training and evaluation would not give a fair assessment of the model’s performance in real world scenarios.

STEP 3: CHOOSING THE MODEL

The selection of the model type is our next course of action once we are done with the data-centric steps. There are various existing models developed by data scientists which can be used for different purposes. These models are designed with different goals in mind. For instance, some models are more suited to dealing with texts while another model may be better equipped to handle images.In more complex scenarios, we need to make the choice that matches our intended outcome. The options for machine learning models can be explored across 3 broad categories: Supervised Learning, Unsupervised learning and Reinforcement learning.

STEP 4: TRAINING THE MODEL

At the heart of the machine learning process is the training of the model. Bulk of the “learning” is done at this stage.Here we use the part of data set allocated for training to teach our model its work. If we view our model in mathematical terms, the inputs will have coefficients. These coefficients are called the weights of features. Initially, we pick random values for them and provide inputs. The achieved output is compared with actual output and the difference is minimized by trying different values of weights and biases. The iterations are repeated using different entries from our training data set until the model reaches the desired level of accuracy.

STEP 5: EVALUATION

With the model trained, it needs to be tested to see if it would operate well in real world situations. That is why the part of the data set created for evaluation is used to check the model’s proficiency. However, through its training, the model should be capable enough to extrapolate the information and deem whether the expected output is served. Evaluation becomes highly important when it comes to commercial applications. Evaluation allows data scientists to check whether the goals they set out to achieve were met or not. If the results are not satisfactory then the prior steps need 


to be revisited so that root cause behind the model’s under performance can be identified and subsequently, rectified.

STEP 6: HYPERPARAMETER TUNING

If the evaluation is successful, we proceed to the step of hyperparameter tuning. This step tries to improve upon the positive results achieved during the evaluation step. There are different ways we can go about improving the model. One of them is revisiting the training step and use multiple sweeps of the training data set for training the model. This could lead to greater accuracy as the longer duration of training provides more exposure and improves quality of the model. Another way to go about it is refining the initial values given to the model.  Random initial values often produce poor results as they are gradually refined by trial and error. However, if we can come up with better initial values or perhaps initiate the model using a distribution instead of a value then our results could get better.

STEP 7: PREDICTION

The final step of the machine learning process is prediction. This is the stage where we consider the model to be ready for practical applications. The model gains independence from human interference and draws its own conclusion on the basis of its data sets and training. The challenge for the model remains whether it can outperform or at least match human judgment in different relevant scenarios.

REGRESSION

Regression is analysis consists of set of machine learning methods that allow us to predict outcome variable(y) based on the value of one or multiple predictor variable(x). The goal of regression model is to build a mathematical equation that defines y as a function of the x variables.
For ex: Predicting prices of a house given the features of house like size, price etc is one of the common example of Regression and it is supervised  technique.

LINEAR REGRESSION 

Linear regression is one of the easiest and most popular Machine Learning algorithms. It is a statistical method that is used for predictive analysis. Linear regression makes predictions for continuous/real or numeric variables such as sales, salary, age, product price, etc.
 	Linear regression algorithm shows a linear relationship between a dependent (y) and one or more independent (y) variables, hence called as linear regression. Since linear regression shows the linear relationship, which means it finds how the value of the dependent variable is changing according to the value of the independent variable. The linear regression model provides a sloped straight line representing the relationship between the variables. Consider the below image:
  
  ![](https://github.com/ranjanashivarudra/Predictive-Analysis-of-Income-Generation-in-Ecommerce-Industry/blob/main/LR.png)

Mathematically, we can represent a linear regression as:
y= a0+a1x+ ε
Y= Dependent Variable (Target Variable)
X= Independent Variable (predictor Variable)
a0= intercept of the line (Gives an additional degree of freedom)
a1 = Linear regression coefficient (scale factor to each input value).
ε = random error

The values for x and y variables are training datasets for Linear Regression model representation.

Types of Linear Regression

Linear regression can be further divided into two types of the algorithm:

Simple Linear Regression:

If a single independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Simple Linear Regression.

Multiple Linear regression:

If more than one independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Multiple Linear Regression.

OUTPUT

![](https://github.com/ranjanashivarudra/Predictive-Analysis-of-Income-Generation-in-Ecommerce-Industry/blob/main/relationship%20of%20each%20col.png)

Description: The above snapshot shows the output for the relationship between each of the columns. Here each grid shows that each variable in data will be shared in Y-axis across a single row and in x-axis across a single column.

![](https://github.com/ranjanashivarudra/Predictive-Analysis-of-Income-Generation-in-Ecommerce-Industry/blob/main/heatmap%20corr.png)

Description: This snapshot shows the construction of Heatmap of pairwise correlation of all columns. Here you can observe there is a strong correlation between length of Membership and Yearly Amount Spent.

![](https://github.com/ranjanashivarudra/Predictive-Analysis-of-Income-Generation-in-Ecommerce-Industry/blob/main/yearly%20vs%20web.png)

Description: The above snapshot shows the construction of joint plot to compare the data of time on website and Yearly Amount Spent. Here by looking at the graph we can see the data are not linear they look completely scattered.

![](https://github.com/ranjanashivarudra/Predictive-Analysis-of-Income-Generation-in-Ecommerce-Industry/blob/main/yearly%20vs%20app.png)

Description: The above snapshots shows the construction of joint plot to compare the data of time on app and Yearly Amount Spent. Here by looking at the graph we can see the data are quite linear when compared to time on website.

![](https://github.com/ranjanashivarudra/Predictive-Analysis-of-Income-Generation-in-Ecommerce-Industry/blob/main/yearly%20vvs%20member.png)

Description: The above snapshots shows the Linear model plot of Yearly Amount Spent vs Length of Membership. Here by looking at the graph we can see as data of Length of membership increases the data of Yearly Amount Spent increases that exactly what linear regression has to be. And also we have quite good linear regression graph. 

![](https://github.com/ranjanashivarudra/Predictive-Analysis-of-Income-Generation-in-Ecommerce-Industry/blob/main/predictive%20value%20vs%20real%20test%20value.png)

Description:  The above snapshots shows the construction of scatter plot  between Predicted value and Real test value. Here the graph looks almost linear and hence the real test values are almost close to predicted values.

![](https://github.com/ranjanashivarudra/Predictive-Analysis-of-Income-Generation-in-Ecommerce-Industry/blob/main/coefficient.png)

Description: The above snapshots shows the Coefficients for each of the independent variables. Here you can see among Time on App and Time on Website, Time on App has greater coefficient than Time on Website. Hence Time on App is what the Company has to choose for the betterment of their company.

CONCLUSION

As the project is completely based on whether the company needs to focus their effort on marketing their clothing on company’s App or website, by looking at their financial graph which among App or website will increase their company’s financial growth. From Snapshot coeeficient.png we can say that, holding all other features fixed, a 1 unit increase in, Avg. Session Length will lead to an increase in $25.981550 in Yearly Amount Spent, and similarly holding all other features fixed, a 1 unit increase in, Time on App will lead to an increase in $38.590159 in Yearly Amount Spent and holding all other features fixed, a 1 unit increase in, Time on Website will lead to an increase in $0.190405 in Yearly Amount Spent.
So as Time on App is a much more significant factor than Time on Website, the company has a choice: they could either focus all the attention into the App as that is what is bringing the most money in, or they could focus on the Website as it is performing so poorly!

