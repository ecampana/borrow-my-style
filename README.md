# Borrow My Style

---
## Contents

[**1. Introduction**](#introduction)

[**2. Challenges with the Data**](#data)

[**3. Fun with Feature Engineering**](#feature_engineering)

[**4. Modeling Rentability**](#modeling_rentability)

[**5. Model Performance**](#model_performance)

[**6. The Future Looks Fashionable**](#summary)

**Page under construction. Please have a look through the git repository. The Jupyter notebooks are finalized and describe the analysis in greater detail.**



# <a name="introduction">Introduction</a>

Insight Data Science is an intensive postdoctoral training fellowship that bridges the gap between academia and a career in data science. As part of the program, I had the wonderful opportunity to consult with _Borrow My Style_*, a fashion e-commerce startup. My client company provides a peer-2-peer rental community where they wish to enable people to either rent or sell fashion items such as dresses, handbags, as well as shoes and accessories. Currently, the company has no recommendation system in place for lenders so adding one will greatly benefit them. The purpose of this blog post is to detail the models that were built to evaluate inventory performance.

<div style="text-align:center">
<img src ="images/computer.jpg" width="438" height="275" /><img src ="images/online closet.jpg" width="240" height="275" />
</div>

I was tasked with creating a recommendation system to help garment lenders know which of their items will get rented most frequently so that they can offer more popular items and the company can enhance their revenue. I trained a logistic regression model that predicts how likely garments will be rented based exclusively on information provided by all lenders, namely brand name, item type, apparel size, and rental price.

*For the purposes of anonymity, _Borrow My Style_ is a pseudonym for the consulting client.


# <a name="data">Challenges with the Data</a>

[Data cleanining pipeline (notebook)](https://nbviewer.jupyter.org/github/ecampana/borrow-my-style/blob/master/data-cleaning-pipeline.ipynb)


The company is a young start up with a small technical team. They are immensely interested in finding ways to explore the data and extract fashion trends that could help lenders. Their inventory data is stored on a Heroku PostgreSQL database that contained about six thousand apparel item samples collect during a three year period. The data contained a small fraction of samples that needed to be removed simply because they did not provide enough information to be of any use. With that in mind, the rest of the data had a wealth of information. The data exploration revealed that much of the inventory is under utilized which really eliminated the possibility of using standard forecasting models to predict inventory trends.


<div style="text-align:center"><img src ="images/data preperation.jpg" width="630" height="320" /></div>

Brand names were curated to remove any variability in their spellings. This reduced the list of brand names by 30%. More advance techniques, such as Natural Language Processing (NLP), to determine text similarity between brand names could have been used but our main concern regarded brand names that are related but appear with completely different spellings. For example, lenders could list their item as "marciano" which is a brand of Guess under the more formal labeling Guess by Marciano. Domain knowledge was instrumental to guarantee that items were associated with the appropriate brand name, and in this instance with "guess".

Moving forward we will only consider apparel (i.e. "tops", "skirts", "pants", "outerwear", "rompers", "shirts", "dresses", and "bottoms") in our modeling while handbags, shoes, and accessories can be modeled independently. 

# <a name="feature_engineering">Fun with Feature Engineering</a>

[Feature engineering (notebook)](https://nbviewer.jupyter.org/github/ecampana/borrow-my-style/blob/master/feature-engineering.ipynb)

We focus on engineering new features that will advance us towards a predictive model for inventory trends.


## Apparel Sizes

Apparel sizes can be numerical, ranging from zero and upwards, but in some instances they may be categorical, for example, "XS", "S", etc.. Most sizes in the data are reported as a number and, therefore, we will choose to transform the few categorical labels that exist into a numerical value. Had the converse been true, we would have converted the numerical values into categorical labels. Individual ranges for "XS", "S", and "M" may be found online. For simplicity, we did not take into account the vanity sizes of the diverse brands and leave this as an underlying assumption of our modeling.

A minority of samples have their apparel sizes missing. For these cases, we replaced the missing value by the most frequent size in their respective item type, for example, the most common dress size was 4. This choice made the most sense when taking a look at the distribution of dress sizes.


## Standardizing Features

The apparel size and rental price are standardized for all models even though some may not strictly need this transformation. A standardized feature is a variable that has been rescaled to have a mean of zero and a standard deviation of one. A benefit of this, for instance, is that results of models such as logistic regression can tell us how the predictive value of item size compares to the predictive value of rental price.


## Choosing Rentability as the Predicted Measure

It is insufficient to merely predict whether an item will be rented since a lender will not be aware under what circumstances their item is predicted to be rented. The model is implicitly assuming the apparel will be available for at least a certain amount of time because this is what it observed in the training data for similar items. This situation is not ideal so taking inventory lifetime into account in some manner will go a long way in resolving the dilemma. 

A suitable quantity to track inventory trends is rentability, which we define as the average number rentals per week (i.e. rental count/lifetime). The lifetime is calculated by taking the difference between the date the item was last listed and the date it was first listed. The result are given in units of days and for this reason we divide by seven so that we may report the lifetime in number of weeks.

We choose the size and number of bins in which to break up the observed rentability rates, as shown in Figure 1. 

<div style="text-align:center"><img src ="images/rentability.jpg" width="494" height="379" />
<figcaption>
<font size="2">
Figure 1: Rentability rates in Log(count) vs Log(average rental per week) scale <br>
to magnify any interesting features the data may have.
</font>
</figcaption>
</div>

Items that have never rented at any point in time are classified as "Low" performing inventory. We next select a rentability threshold that will allow the top 50% of inventory with large rentability rates to be labeled as "High" performing while the rest will be classified as "Moderate" performing. The motivation behind choosing 50% was to ensure that each rentability classification will have enough statistics for our modeling. We have now framed the problem as a multi-class classification.

<div style="text-align:center"><img src ="images/rentability classification.jpg" width="620" height="280" /></div>


# <a name="modeling_rentability">Modeling Rentability</a>

[Modeling rentability (notebook)](https://nbviewer.jupyter.org/github/ecampana/borrow-my-style/blob/master/modeling-rentability.ipynb)


We are now ready to model inventory trends for our client company. The main focus here will be to explore different machine learning algorithms to predict item performance based on brand name, item type, apparel size, and rental price. The reason we are restricting ourselves to these particular features is that lenders will be able to provide this information for us.


## Modeling of Inventory Performance

We explore several machine learning models that are inherently multi-class classifiers. Models that are interpretable are preferred for implementation, while less interpretable models are used as benchmarks.

Looking at the distribution of rentability rates in Figure 1, we can see that there are far more apparels that fall in the low rentability bin than in the moderate or high rentability bins. Classification models perform best when every bin contains a similar number of samples. Therefore, we artificially increase the number of samples in the moderate and high rentability bins through a process known as bootstrapping (i.e. oversampling with replacement of the minority class). The hyper-parameters will be optimized and cross-validated using the Logarithmic Loss function (i.e. log loss). Log loss heavily penalizes any strongly mis-classified prediction, and for this reason it was chosen.


### Dummy Classification

In order to quantify the minimum acceptable performance for a classification model, we employ a **dummy classifier**. The classifier makes random predictions on the test dataset based on what it found the class composition to be in the training sample. If the training data had 60% low-performing inventory, 30% moderate-performing inventory, and 10% high-performing inventory then it will make predictions based on these proportions on the test dataset irrespective of the samples' actual features.


As expected, the dummy classifier performs worse on the test dataset than on the training dataset. Its precision and recall values were about 8% for both moderate and high performing inventory. Therefore, we would like to know if other machine learning algorithms can perform better than this baseline model.


### Logistic Regression

Our first attempt is with a linear model like **Multinomial logistic regression with Ridge regularization**. We investigate different regularization parameters and use the one that performs the best. The final logistic regression model had a precision of 12% and recall of 39% for moderate-performing items. For high-performing inventory the precision and recall were 19% and 52%, respectively. Undoubtedly, we have found a model that performs better than random guessing. 

### Non-linear or Non-probabilistic Classifiers

I used non-linear models (**Gradient boosting** and **Random forest**) to explore whether the data contained any variables whose interaction terms had predictive value. Additionally, non-probabilisitic models (**K-neighbors** and **Support vector machine**) were chosen to cross check our preferred models. The findings of these studies are detailed in the Jupyter notebook under "[Modeling of Inventory Performance](https://nbviewer.jupyter.org/github/ecampana/borrow-my-style/blob/master/modeling-rentability.ipynb)".


## Learning Curve

In the plot below, the negative log loss is shown as a function of training sample size. We notice that no sample size optimizes the training score curve since it never peaks or plateaus. This indicates that having more data will enhance the model performance. As the company continues to rent or sell more inventory, garment lenders will receive more accurate predictions of how frequently their apparels will be rented. 

<div style="text-align:center">
<img src ="images/learning curve logistic regression.png" width="459" height="348" />
</div>
                                                                                                                                                                                                                                                                                                                                                                                    

# <a name="model_performance">Model Performance</a>

[Model performance (notebook)](https://nbviewer.jupyter.org/github/ecampana/borrow-my-style/blob/master/model-performance.ipynb)

My primary focus was to choose a model that fulfills the company's goal of helping lenders understand which fashion items they should make available to other people. To compare the performance of different model configurations, two metrics tracked were **precision** and **recall**. The higher the precision the fewer the number of false positives (i.e. classes of no interest which were predicted to be of interest) while the higher the recall the smaller the chances an item of interest is predicted to be of no interest.

For this project, having a high recall value for the moderate and high rentability bins took priority because we want the model to find as many of those types of items as possible. These will be the fashion items that have the potential to bring in greater revenue for both the client company and lender. Unfortunately, the higher the recall score the lower the precision will be. In our case, we do not necessarily need precision to be especially large. Although, as a consequence there will be a greater number of low-performing items on the website than what is ideal but overall the fashion catalog should decidedly improve with apparel that renters demand.


## Precision vs Recall

Precision and recall are used for the model selection and evaluation and, in addition, they are cross-validated for robustness.

### Low Performing Inventory

In the plot below, which compares how precisely different models classified low-performing inventory, random forest and gradient boosted decision trees perform the best, but we do not care to model the low-performing inventory as best as possible. It is more important to chose a machine learning algorithm that performs better for high and moderate-performing inventory than for low performing inventory.

<div style="text-align:center">
<img align="center" width="451" height="356" src="images/p vs r low performing inventory.png" hspace="40" vspace="40">
</div>


### Moderate Performing Inventory

In the plot below, both random forest and logistic regression perform the best for moderately performing inventory. For now these are our best candidates.

<div style="text-align:center">
<img align="center" width="451" height="356" src="images/p vs r moderate performing inventory.png" hspace="40" vspace="40">
</div>


### High Performing Inventory

In this last plot, we can see that logistic regression out performs random forest. For this reason we select logistic regression as our final model. It has the best recall without sacrificing precision too much.

<div style="text-align:center">
<img align="center" width="451" height="356" src="images/p vs r high performing inventory.png" hspace="40" vspace="40">
</div>


All models had relatively low precision but we should not be overly concerned about this issue since unintentionally allowing lenders to share items that may not perform as well as they expect will not cause the client company to incur unnecessary monetary loss. 


## Feature Importance

Now that we have settled on Multinomial logistic regression with Ridge regularization as our model to evaluate inventory performance we can use it to understand our data. What data insights can we extract from our model? Are there some brands more popular than others? Does rental price have an effect on rentability? Is there a mismatch between apparel sizes offered by lenders and those sizes demanded by renters? We will focus our efforts on answering these questions.
​

We can use the regression coefficients of our logistic model as a way to rank the relative predictive importance of each feature. The coefficients of logistic regression are interpretable. For example, for one unit of increase in the rental price of an item, we expect to see an increase or decrease in the odds of being a high-performing item over a low-performing item, given by the expression,

<div style="text-align:left">
<img src="images/change in odds.png" width="337" height="27" />
</div>

where <img src="images/regression coefficient.png" width="97" height="27" horizontal="5" /> is the coefficient of the rental price for class 1, and similarly for class 0.


### What makes an item moderate performing?


The model indicates that the rental price, apparel size, and item type do not strongly impact whether a garment will be of a moderate rentability rate when compared to low-performing inventory. In fact, it was brand name that most influenced whether a garment was of moderate rentability. The figure below illustrates the regression coefficients.

<div style="text-align:center">
<img src="images/lr coefficients moderate relative to low.png" width="510" height="322" />
<img src="images/lr coefficient size and rental price moderate relative to low.jpg" width="202" height="141" />
</div>


### What makes an item high performing?

With respect to high-performing inventory, the model indicates the rental price has an effect on rentability. It suggests, by having a positive regression coefficient, that the higher the rental price is the better it will perform. This may be counterintuitive at first as we would expect that as the rental price increases the item will be less likely to rent. One explanation for this is that the items are perceived as having greater value because of the higher price tag. Another possible explanation is that since there is a suggested rental price of about 15% of the retail price lenders tend to set the price higher for more well-known brands. This causes the rental price to be correlated with expensive brand names. Had this not been the case then the change in odds would have most likely reflected our intuition.

<div style="text-align:center">
<img src="images/lr coefficients high relative to low.png" width="510" height="322" />
<img src="images/lr coefficient size and rental price high relative to low.jpg" width="202" height="141" />
</div>

We are also starting to see a trend as to which brand names are the most popular in the moderate and high performing inventory category.


### Any differences between high and moderate performing?

Comparing model results for high-performing items versus moderate-performing ones uncovers some interesting subtleties.

<div style="text-align:center">
<img src="images/lr coefficients high relative to moderate.png" width="510" height="322" />
<img src="images/lr coefficient size and rental price high relative to moderate.jpg" width="202" height="141" />
</div>

One last observation is that, for low-performing garments, lenders tend to offer larger sizes than the sizes most sought by renters. The magnitude of the regression coefficient for apparel size was comparable to the coefficient of other features. If lenders offered smaller sizes the chances of an item renting will improve in comparison to low-performing inventory. This is not as true between high and moderate-performing inventory where item size does not seem to make much of a difference, as suggested by the very small regression coefficient.


# <a name="summary">The Future Looks Fashionable</a>

We have explored a few years worth of inventory data and attempted to model their rentability in order to help lenders understand which items to make available to other people. Multinomial logistic regression had the best performance for identifying high and moderate-performing items by exhibiting a high recall value without the need to greatly sacrifice precision. What does this mean for _Borrow My Style_? Going forward they can use the model to construct a recommendation system for lenders to guide them in sharing potentially better performing fashion items. And, renters will have a better selection of apparel available to them. The company expects to measure success based on whether a person lends an item recommended by the system. In the future, we could investigate other features, for example, ascertaining whether the color of fashion items has any impact on rentability.


### <a name="about_me">About the Author</a>


<img align="right" width="290" height="289" src="images/IMG_2431_circular.png" hspace="25">

My name is Emmanuel Contreras-Campana. I received a Ph.D. in experimental high energy physics searching for anomalous production of multilepton events in proton-proton collision at the LHC complex collected by CMS collaboration. I am seeking opportunities in big data analytics in the financial, technology, or health industries. My passion is working with complicated datasets that require rigorous transformations and cleaning in the interest of extracting useful insights that have substantive business value. Last summer, I completed a data science internship at TripleLift. They are in the marketing and online advertising industry. I had the opportunity to worked on predicting viewability of digital ads to improve advertiser spending.

