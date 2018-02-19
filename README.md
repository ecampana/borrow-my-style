# Borrow My Style

---
## Contents

[**1. Introduction**](#introduction)

[**2. Challenges with the Data**](#data)

[**3. Feature Engineering**](#feature_engineering)

[**4. Modeling Rentability**](#modeling_rentability)

[**5. Model Performance**](#model_performance)

[**5. Summary**](#summary)

Page under construction. Please have a look through the git repository. The jupyter notebooks are finalized and describe the analysis in greater detail.

# <a name="introduction">Introduction</a>
Insight Data Science is an intensive postdoctoral training fellowship that bridges the gap between academia and a career in data science. As part of the program, I had the wonderful opportunity to consult with _Borrow My Style_*, a fashion e-commerce startup. My client company provides a peer-2-peer rental community where they wish to enable people to either rent or sell fashion items such as dresses, handbags, shoes, and accessories as well. The purpose of this blog post is to detail the models that were produced to evaluate inventory performance and provide a recommendation system for lenders.

*For the purposes of anonymity, _Borrow My Style_ is a pseudonym for consulting client.

<div style="text-align:center"><img src ="images/computer.jpg" width="317" height="210" /></div>

<div style="text-align:center"><img src ="images/online closet.jpg" width="317" height="383" /></div>

# <a name="data">Challenges with the Data</a>
[data cleanining pipeline](https://github.com/ecampana/borrow-my-style/blob/master/data-cleaning-pipeline.ipynb)

This notebook outlines the data cleaning and transformations that were performed on the client company's data for the purposes of analyzing it.

The data is stored on a Heroku PostgreSQL database which we ingest into this notebook and proceed to explore it.

We drop all columns that are not relevant for identifying inventory trends.

We join the items table to the rental items table to have a more meaningful table which can be used to explore trends of the fashion items ordered.

From the orders table we can determine the rental count of each item. This information can be used, for example, to learn how often an item is rented during its listing lifetime.

An interesting quantity to have is the rental revenue of each item which may be used at a later point to evaluate the improvements in company profits when compared to the previous model.


The data contains several samples that need to be removed either because there is not enough information for it to be useful or they need to be corrected in one way or another.

In order to determine the lifetime of items we need to know when they were delisted. But most items continue to be rented so in those cases we chose the current date as the date that they were delisted.

Brand names are curated to remove any variablity in their spellings. This reduced the list of brand names by 30%.

The final stage of the data cleaning pipeline is storing the results into cvs files for later use.


<div style="text-align:center"><img src ="images/data preperation.jpg" width="741" height="376" /></div>

# <a name="feature_engineering">Feature Engineering</a>
[feature engineering](https://github.com/ecampana/borrow-my-style/blob/master/feature-engineering.ipynb)

# <a name="modeling_rentability">Modeling Rentability</a>
[modeling rentability](https://github.com/ecampana/borrow-my-style/blob/master/modeling-rentability.ipynb)
<div style="text-align:center"><img src ="images/rentability classification.jpg" width="388" height="175" /></div>

<div style="text-align:center"><img src ="images/rentability.jpg" width="317" height="210" /></div>

# <a name="model_performance">Model Performance</a>
[model performance](https://github.com/ecampana/borrow-my-style/blob/master/model-performance.ipynb)


# <a name="summary">Summary</a>

We have explored a few years worth of inventory data and attempted to model their rentability in order to help lenders understand what items to make available to other people. Logistic regression had the best performance for identifying high and moderate performing items by having a high recall value without needing to sacrafice precision too much. All models had relatively low precision but we should not be overly concerned about this since unintentially allowing lenders to share items that may not perform as well as they hope will not cause the client company to incur any monetary loss. The consequences are that there may be more lower performing items than what is ideal but overall the fashion catalog should decidely improve with those items that renters desire. Going forward the fashion company can use the model to construct a recommendation system for lenders to guide them to share better peforming apparel.


### <a name="about_me">About the Author</a>

My name is Emmanuel Contreras-Campana. I received a Ph.D. in experimental high energy physics searching for anomalous production of multilepton events in proton-proton collision at the LHC complex collected by CMS collaboration. I am seeking opportunities in big data analytics in the financial, technology, or health industries. My passion is working with complicated datasets that require rigorous transformations and cleaning in the interest of extracting useful insights that have substantive business value. Last summer, I completed a data science internship at TripleLift. They are in the marketing and advertising industry. I had the opportunity to worked on predicting viewability of digital ads to improve advertiser spending.

