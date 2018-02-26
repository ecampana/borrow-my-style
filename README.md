# Borrow My Style

---
## Contents

[**1. Introduction**](#introduction)

[**2. Challenges with the Data**](#data)

[**3. Fun with Feature Engineering**](#feature_engineering)

[**4. Modeling Rentability**](#modeling_rentability)

[**5. Model Performance**](#model_performance)

[**6. The Future Looks Colorful**](#summary)

**Page under construction. Please have a look through the git repository. The jupyter notebooks are finalized and describe the analysis in greater detail.**



# <a name="introduction">Introduction</a>

Insight Data Science is an intensive postdoctoral training fellowship that bridges the gap between academia and a career in data science. As part of the program, I had the wonderful opportunity to consult with _Borrow My Style_*, a fashion e-commerce startup. My client company provides a peer-2-peer rental community where they wish to enable people to either rent or sell fashion items such as dresses, handbags, as well as shoes and accessories. The purpose of this blog post is to detail the models that were built to evaluate inventory performance and provide a recommendation system for lenders.

<div style="text-align:center">
<img src ="images/computer.jpg" width="438" height="275" /><img src ="images/online closet.jpg" width="240" height="275" />
</div>

*For the purposes of anonymity, _Borrow My Style_ is a pseudonym for the consulting client.



# <a name="data">Challenges with the Data</a>

[Data cleanining pipeline (notebook)](https://nbviewer.jupyter.org/github/ecampana/borrow-my-style/blob/master/data-cleaning-pipeline.ipynb)


The company is a young start up with a small technical team. They are immensely interested in finding ways to explore the data and extract fashion trends that could help lenders. Their inventory data is stored on a Heroku PostgreSQL database that contained about six thousand sample collect during a three year period. The data contained a small fraction of samples that needed to be removed simply because they did not provide enough information to be of any use. With that in mind, the rest of the data had a wealth of information. The data exploration revealed that much of the inventory is under utilized which really eliminated the possibility of using standard forecasting models to predict inventory trends.


<div style="text-align:center"><img src ="images/data preperation.jpg" width="630" height="320" /></div>

Brand names were curated to remove any variablity in their spellings. This reduced the list of brand names by 30%. More advance techniques, such as Natural Language Processing (NLP), to determine text similarity between brand names could have been used but our main concern regarded brand names that are related but appear with completely different spellings. For example, lenders could list their item as "marciano" which is a brand of Guess under the more formal labeling Guess by Marciano. Domain knowledge was instrumental to guarantee that items were associated with the appropriate brand name, and in this instance with "guess".

Moving forward we will only consider apparel (i.e. "tops", "skirts", "pants", "outerwear", "rompers", "shirts", "dresses", and "bottoms") in our modeling while handbags, shoes, and accesories can be modeled independently. 

# <a name="feature_engineering">Fun with Feature Engineering</a>

[Feature engineering (notebook)](https://nbviewer.jupyter.org/github/ecampana/borrow-my-style/blob/master/feature-engineering.ipynb)

We focus on engineering new features that will advance us towards a predictive model for inventory trends.


## Apparel Sizes

Apparel sizes can be numerical, ranging from zero and upwards, but in some instances they may be categorical, for example, "XS", "S", etc.. Most sizes in the data are reported as a number and, therefore, we will choose to transform the few categorical labels that exist into a numerical value. Had the converse been true, we would have converted the numerical values into categorical labels. Individual ranges for "XS", "S", and "M" may be found online. For simplicity, we did not take into account the vanity sizes of the diverse brands and leave this as an underlying assumption of our modeling.

A minority of samples have their apparel sizes missing. For these cases, we replaced the missing value by the most frequent size in their respective item type, for example, the most common dress size was 4. This choice made the most sense when taking a look at the distribution of dress sizes.


## Standardizing Features

The apparel size and rental price are standardized for all models even though some may not strictly need this transformation. One benefit of this, for example, is that the regression coefficients for logistic models may be compared to each other to gain additional insight into the data that may prove to be useful.


## Classifying Rentability

It is insufficient to merely predict whether an item will be rented since a lender will not be aware under what circumstances their item is predicted to be rented. The model is implicitly assuming the apparel will be available for at least a certain amount of time because this is what it observed in the training data for similar items. This situation is not ideal so taking inventory lifetime into account in some manner will go a long way in resolving the dilemma. 

A suitable quantity to track inventory trends is rentability, which we define as the average number rentals per week (i.e. rental count/lifetime). The lifetime is calculated by taking the difference between the date the item was last listed and the date it was first listed. The result are given in units of days and for this reason we divide by 7 so that we may report the lifetime in number of weeks.

We study the rentability distribution of items to see if they fall into separate groups, which will serve as our target value for prediction. 

<div style="text-align:center"><img src ="images/rentability.jpg" width="494" height="379" />
<figcaption>Rentability is plotted in Log(count) vs Log(average rental per week) to magnify any interesting features the data may have.</figcaption>
</div>

Items that have never rented at any point in time are classified as "Low" performing inventory. We next select a rentability threshold that will allow the top 50% of inventory with large rentability to be labeled as "High" performing while the rest will be classified as "Moderate" performing. The motivation behind chosing 50% was to ensure that each rentability classification will have enough statistics for our modeling. We have now framed the problem as a Multi-class classification.

<div style="text-align:center"><img src ="images/rentability classification.jpg" width="620" height="280" /></div>


# <a name="modeling_rentability">Modeling Rentability</a>

[Modeling rentability (notebook)](https://nbviewer.jupyter.org/github/ecampana/borrow-my-style/blob/master/modeling-rentability.ipynb)


We are now ready to model inventory trends for our client company. The main focus here will be to explore different machine learning algorithms to predict item performance based on brand name, item type, apparel size, and rental price. The reason we are restricting ourselves to these particular features is that lenders will be able to provide this information for us.


## Modeling of Inventory Performance

We explore several machine learning models that are inherently multi-class classifiers. Models that are interpretible are preferred and use the others as a sanity check.

The high and moderate performing inventory samples are highly imblanced with respect to the low preforming inventory so care must be taken. This is accomplished by oversampling the minority class to match that of the majority class. In essence, we will be using bootstrap sampling. The hyper-paramerters will be optimized and cross-validated using the Logarithmic Loss function (i.e. log loss). Log loss heavily penalizes any strongly mis-classified prediction and for this reason it was chosen. Precision and Recall are used for the model selection and evaluation. Those values are also cross-validated for robustness.


### Dummy Classification

Dummy classification will serve as our baseline model. The classifier will make random predictions on the test dataset based on what it found the class composition to be in the training sample. If the training data had 60% low performing inventory, 30% moderate performing inventory, and 10% high performing inventory then it will make predictions based on these porportions on the test dataset irrepective of what the actual features are of the samples.

The dummy classifier does not generalize well to the test dataset. Its precsion (i.e. about 8%) and recall (i.e. 8%) values are very low and it has a large log loss value.

Therefore, we would like to know if other machine learning algorithms can peform better than the baseline model.


### Logistic Regression

Our first attempt is with a linear model like Logistic Regression. We investigate different regularization parameters and use the one that performs the best.


These values may be compared to the previous cross-validated precision and recall values and serves to check that everything is staying consistent.


### Gradient Boosting Classifier

For our second attempt, we choose a non-linear model that could in essence capture any interaction terms between features the data may have.


### Random Forest Classifier

Another non-linear model we can use is random forrest which differs from gradient boosted decesion trees in that the former produces trees that are statistically independent.


## Overfitting plots

Once the models have been trained and hyper-parameters optimized, we can explore how well they model the data. For example, is the probability distribution for high performing inventory items modeled well between test and training data in the case of the signal and background samples seperately.

In the plots above, the solid blue histograms represent the probability distribution for the background training data while the solid red histograms are for the signal training data. The blue and red dots are for the test data. In the first plot, the signal is the low performing inventory while the moderate and high performing inventory are considered background. In the second plot, the signal is the moderate performing inventory and in the last the signal is the high performing inventory.

### Logistic Regression Classifier


We observed that there that the model is not being overtrained on the training data as the background and signal distributions are well model, respectively. There also appears to be a discrimination between the signal and background distributions which is a good indication that the model will perform well but not necessarily. Further investigation will be needed to be sure of this.


<img src ="images/logistic regression low.png" width="400" height="306" /><img src ="images/logistic regression moderate.png" width="400" height="306" />

<div style="text-align:center"><img src ="images/logistic regression high.png" width="400" height="306" /></div>

Having used several machine learning algorithms to model the data in order to predict inventory performace we must systematically evaluate each one to select the best one. We accomplish this in the next step.


## Learning Curve

The plot shows that given the amount of data that is currently available our models are under trained. In the future as the company continues to rent or sell more inventory the model should show improvement in its predictions.


<div style="text-align:center">
<img src ="images/learning curve logistic regression.png" width="510" height="387" />
</div>
                                                                                                                                                                                                                                                                                                                                                                                    

# <a name="model_performance">Model Performance</a>

[Model performance (notebook)](https://nbviewer.jupyter.org/github/ecampana/borrow-my-style/blob/master/model-performance.ipynb)


In this notebook, we center our attention to evaluating our various models in a more systematic fashion. An important idea to keep in mind is choosing a model that furthers the companies objective. We would like to help lenders understand which fashion items they should make available to others.

Two metrics worth considering are Precision and Recall. The higher the precision the lower the number of false positives (i.e. classes of no interest but were predicted to be of interest). While the higher the recall the smaller the chances an item of interest is predicted to be of no interest.

We prefer the recall value to be as high as possible for the high and moderate performing inventory because we would like to find as many of them as possible. Those are the fashion items that have the potential to bring in more revenue for both the lender and client company. Unfortunately, the higher the recall the lower the precision will be. For our case, we do not necessarily need precision to be all that high for the high and moderate performing inventory. The consequences of this will mean having more unwanted fashion items in the peer-2-peer rental community but the company will not actuallly incur any monetary loss only potential loss.


## Precision vs Recall

Our cross-validated precision and recall values appear as dots on the figures. We can see that the probabilty thresholds chosen by the algorithm are quite good and do not need to be modified. We must keep in mind that the curves themselves are not cross-validated so they will have statistical fluctations making the cross-validated points not lie completely on the curve.


### Low Performing Inventory

<img align="right" width="451" height="356" src="images/p vs r low performing inventory.png" hspace="40" vspace="40">

In the plot to the right, random forest and gradient boosting decision trees peform the best but we do not care to model the low performing inventory as best as we can. It is more important to chose a machine learning algorithm that performs better for high and moderate performing inventory than for low performing inventory.


### Moderate Performing Inventory

<img align="right" width="451" height="356" src="images/p vs r moderate performing inventory.png" hspace="40" vspace="40">

In the above plot, both random forest and logistic regression perform the best for moderately performing inventory. For now these are our best candidates.


### High Performing Inventory

<img align="right" width="451" height="356" src="images/p vs r high performing inventory.png" hspace="40" vspace="40">

In this last plot, we can see that logistic regression out performs random forest. For this reason we select logistic regression as our final model. It has the best recall without sacrificing precision too much.



## Feature Importance

Now that we have settled on Logistic Regression with Ridge regularization as our model to evaluate inventory performance we can use it to extract insight about our data. Are there some brands more popular than others? Does rental price have an effect on rentability? Is there a mismatch between apparel sizes offered by lenders and those sizes demendad by renters? We will focus on answering these questions in this section.
​
The coefficients of logistic regression are intrepretable. For example, for one unit of increase in the rental price of an time, we expect to see increase in the odds of being a high performing item over a low performing item, given by the expression,

Δodds = eβrental priceclass 1−βrental priceclass 0
Δodds = eβclass 1rental price−βclass 0rental price,

where βrental priceclass 1βclass 1rental price is the coefficient of the rental price for class 1, and similarly for class 0.


### Moderate Performing Inventory against Low Performing Inventory

The increase in odds of being a moderate performing item for a unit of increase in rental price appears counterintuitive at first. We would expect that as the rental price increases that the item will be less likely to rent. But this is not the case. The reason being is that there is a suggested rental price between 15-20% of the retail price for newer items and 10-15% for older items and so lenders tend to set the price higher for more well known brands. This leads to an artificial dependency of the rental price on the brand name. Had this not been the case then the change in odds would have most likely reflected our intuition.

<img align="right" width="446" height="287" src="images/lr coefficients moderate relative to low.png" hspace="40" vspace="40">


### High Performing Inventory against Low Performing Inventory

The model is again trying to suggest that the lender increase the rental price so that it has a higher chance of being a high peforming inventory item than a low performing inventory item, wich goes against intuition. Basically, rental price is not being as powerful of a predictor for rentability as we would have hoped.

<img align="right" width="446" height="287" src="images/lr coefficients high relative to low.png" hspace="40" vspace="40">

We are also starting to see a trend of which brand names are popular in the moderate and high performing inventory category.


### High Performing Inventory against Moderately Performing Inventory

Another interesting option to consider is what the model has to say about high performing items against moderately peforming items. This will allow us to understand slight subtlities in their differences.

<img align="right" width="446" height="287" src="images/lr coefficients high relative to moderate.png" hspace="40" vspace="40">

We notice that the regression coefficient for rental price is still positive. This indicates that the model continues to predict fashion items to rent better if we were to increase the rental price. We expect that with higher volume of rentals this counter intuitive result to reverse.

One last observations to make is that there isn't a significant mismatch between apparal sizes offered by lenders and those sizes renters are interested in. In all is cases the magnitude of the regression coefficient for apparel size was rather small indicating that it does not offer serious predicitve power but on the bright size this means that there is a good renter experience.




# <a name="summary">The Future Looks Colorful</a>

We have explored a few years worth of inventory data and attempted to model their rentability in order to help lenders understand which items to make available to other people. Logistic regression had the best performance for identifying high and moderate performing items by possessing a high recall value without the need to sacrafice precision too much. All models had relatively low precision but we should not be overly concerned about this issue since unintentially allowing lenders to share items that may not perform as well as they expect will not cause the client company to incur unnecessary monetary loss. Although, as a consequence there will be a greater number of low performing items than what is ideal but overall the fashion catalog should decidely improve with apparel that renters desire. Going forward the fashion company can use the model to construct a recommendation system for lenders to guide them to share better peforming fasshion items. In the future, we could explre whether the color of fashion items has any impact on rentability.


### <a name="about_me">About the Author</a>

My name is Emmanuel Contreras-Campana. I received a Ph.D. in experimental high energy physics searching for anomalous production of multilepton events in proton-proton collision at the LHC complex collected by CMS collaboration. I am seeking opportunities in big data analytics in the financial, technology, or health industries. My passion is working with complicated datasets that require rigorous transformations and cleaning in the interest of extracting useful insights that have substantive business value. Last summer, I completed a data science internship at TripleLift. They are in the marketing and advertising industry. I had the opportunity to worked on predicting viewability of digital ads to improve advertiser spending.

