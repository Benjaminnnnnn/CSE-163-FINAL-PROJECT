---
author: 'Claire Luo, Samantha Shimogawa, Benjamin Zhuang'
date: May 2020
title: 'An Analysis of World Happiness'
---

**Research Questions**

1.  What is the distribution of the happiness scores?

    By understanding the distribution of a dataset, we can identify some
    basic properties of the world happiness dataset. In addition, we can
    identify the outliers of the dataset. And based on this
    distribution, we can determine how to normalize our data.

2.  Which five countries have the most happiness score? Which five
    countries have the least happiness score? What are the geolocations
    of these countries?

    There are over at least 100 countries in the dataset. It's unlikely
    to gain meaningful insights looking at 100 countries at one time.
    Instead, by looking at the top and bottom of happy countries, we can
    get a better sense of the distribution. We can also start thinking
    about the geolocation factors in contributing to the happiness of a
    country.

3.  What are the trends of happiness among different countries? Are they
    increasing, decreasing, or fluctuating?

    If we can see how happiness changes over time, we can see if all
    countries are typically sharing the same trends, or maybe the
    happier countries are all trending upward, while the less happy
    countries are all trending downward, or vice versa. We can see if
    there is sort of a happiness gap across countries, or if countries
    are just trading places in the happiness rank.

4.  What are some measures that contribute to a higher happiness score?
    And what are some measures that contribute to a lower happiness
    score?

    It's useful to understand what contributes most to a country's
    happiness, i.e., what's most important for people to feel happy
    overall? And similarly, to understand what contributes most to a
    country's unhappiness. It helps us to understand what seems to be
    essential for human contentment and happiness.

5.  What is the future happiness score for the five most happy countries
    and the five least happy countries?

    We want to see if we can try to predict future happiness scores for
    these countries based on the information we have in this dataset.

**Motivations**

Why understand the happiest country and the least happiest country? Why
learn the trend of the happiness level of a country? Why extrapolate the
relationship between the six key factors (i.e., economy, health, family,
etc.,) and the level of happiness?\
This world happiness dataset originates from a landmark survey of the
state of global happiness. The first report was published in 2012. In
2017, the report was released at the United Nations. This report
continues to gain attention from governmental entities, global
institutions, different societies, and many individuals, in exploring
how personal and national invariants affect the state of being happy.\
Governments and organizations use happiness indicators to inform
policy-making decisions.\
We care about these problems because we want to understand what
contributes to a well-organized nation, and what makes a "poor nation".
We also care about whether the current progress of a society is making a
person's life happier or not because all the people would want to pursue
a more pleasant life. By observing the happiness ranks of the countries,
we can assess the progress of the nations based on the measurements.

**Datasets**

[World
Happiness](https://www.kaggle.com/unsdsn/world-happiness)(Click on
dataset name to get URL)

The world happiness data consists the happiness score of 155 countries
from 2015 to 2019. The happiness score use data from the Gallup World
Poll, and the score is determined by various aspects of a country, such
as economy, health, freedom, trust between government and people, etc.

-   [^1] Country Name of the country.

-   Region

-   [^2] Happiness Rank Rank of the country based on the Happiness
    Score.

-   Happiness Score Range from 0-10.

-   Standard Error The standard error of the happiness score.

-   Economy (GPD per Capita)

-   Family The extent to which Family contributes to the calculation of
    the Happiness Score.

-   Health (Life Expectancy) The extent to which Life expectancy
    contributed to the calculation of the Happiness Score.

-   Freedom The extent to which Freedom contributed to the calculation
    of the Happiness Score.

-   Trust (Government Corruption) The extent to which Perception of
    Corruption contributes to Happiness Score.

-   Generosity The extent to which Generosity contributed to the
    calculation of the Happiness Score.

-   Dystopia Residual The extent to which Dystopia Residual contributed
    to the calculation of the Happiness Score.

[Country
Ploygon](https://datahub.io/core/geo-countries#pandas)(Click on
dataset name to get URL)

GeoJSON formatted data, describing each country as either a polygon or a
multipolygon

1.  New libraries

    -   requests (loading GitHub repository data)
    -   plotly (visualization animation)

    -   SciPy (data normalization)

    -   folium (interactive map visualization)

    -   mpl_toolkits (2D base map data)

    -   descartes (matplotlib GeoJSON data patch)

    We will use to normalize our data to minimize the cost of our
    machine learning model. We will use to outline the basic world map.
    Lastly, we will use to create interactive, animated visualizations
    and use to interactive 2D map graph. See footnote for .\
    The dataset that we selected has standard country name, which then
    can be joined with our country GeoJSON data. Therefore, we can map
    our data to 2D graph based on different aggregations.

2.  Machine Learning

    -   Whether the country is happy (classification)

    -   Country future happiness score prediction (regression)

    -   Find what are the most important factors for a happy country

    As stated above, we will be exploring different ML models, such as
    Lasso, RandomForest, KNN, to predict the future of world happiness
    score, and classify whether the country is happy or not. The dataset
    has dense features. For regression, we can easily normalize some
    features and train a model for future happiness score prediction.
    For classification, we can convert the continuous variables to
    discrete data by setting up partition intervals, and then train the
    model. We can also use the model determine which features are most
    important for predicting the happiness score, thereby determining
    which features most influence happiness.

**Methodology**

1.  Distribution
    We can make a box plot or histogram to see what the distribution of
    the happiness scores look like. Analyses we can make include the
    most frequent happiness score and the spread of the scores.

2.  5 happiest and least happy countries and their geolocation
    We can use a pandas function to quickly find the top five and bottom
    five countries in the dataset. We can plot these countries onto a
    map, coloring the happy countries one color and the least happy
    countries another color. Then, we can see if location plays a role
    in affecting their happiness scores.

3.  Happiness over time
    We can create a scatterplot of happiness scores over time (i.e.,
    create a plot where Time (in years) is on the x-axis and happiness
    score is on the y-axis, with different countries in different colors
    so we can see their trends over time).

4.  Factors that contribute to a high/low happiness score
    We can use a decision tree to see which factors are important for
    determining the happiness score of a country. We can perform this
    analysis on different countries and find out the variations in the
    factors that they value the most.

5.  Future happiness scores for the 5 happiest and least happy countries

    We can create a ML model (decision tree, logistic regression, random
    forest,or KNN), train it using the existing data from our dataset,
    and then use the model with the best performance to try to predict
    the future happiness scores for the five happiest and least happy
    countries, as found in research question \#2.

**Work Plan**
For the writing portion (Part 0 and Part 1), we work on the report
together through video calling to discuss what to write about for each
part. For the coding portion, we made a GitHub repository where we share
our code. We can assign the simpler tasks for each of us to work
individually. For more challenging or group tasks, everyone can
contribute through video calling.
Estimated Time on Each Problem:

1.  Research Question 1: 1 hour.

2.  Research Question 2: 1 hour.

3.  Research Question 3: 1 hour.

4.  Research Question 4 and 5, possibly using the same ML model: 3-5
    hours.

5.  Testing code & And Adjustments: 3 hours.

6.  General brainstorming, discussion, determining how to divide tasks,
    etc.: 5-10 hours overall from start to finish of the project.

 **Response to Part0 Feedback**
We deleted using to get GitHub repository data part.

[^1]: string

[^2]: real
