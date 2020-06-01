'''
Claire Luo, Samantha Shimogawa, Benjamin Zhuang
CSE 163 Section AD

This program analyzes the world happiness
data from 2015 to 2019. It explores the data
and makes some plots in order to answer few
research question toward the world happiness.
'''
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import graphviz

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

import processing

sns.set()
warnings.filterwarnings('ignore')


# Question 1
# What is the distribution of the happiness scores?
def q1(happiness_data):
    '''
    q1 takes a pandas dataframe as a parameter, then
    makes three plots, a histogram of the distribution of
    happiness score for each year, a heatmap represents the
    correlation in each column of the data, and a 3D choropleth
    map shows the actual distribution of happiness score in the
    world.
    '''
    # histogram
    # make subplots for each year's distribution
    fig = make_subplots(rows=1, cols=5, shared_yaxes=True,
                        vertical_spacing=0.04)
    # getting colors for plotting
    color_histogram = px.colors.qualitative.Set1
    color_discrete_sequence = px.colors.qualitative.D3

    for i, color in list(zip(range(2015, 2020), color_discrete_sequence)):
        data = happiness_data[happiness_data['year'] == i]['happiness score']
        mean_score = data.mean()

        # add a histogram to the figure
        fig.add_trace(go.Histogram(x=data,
                                   name=i,
                                   hovertemplate='Count %{y}',
                                   text='%{y}',
                                   nbinsx=10,
                                   marker_color=color_histogram[i-2015],
                                   opacity=0.8),
                      row=1, col=i-2014)
        # update fig x range
        fig.update_xaxes(nticks=10)

        # add a vertical line (mean happiness score) to the figure
        fig.add_trace(go.Scatter(x=[round(mean_score, 2)] * 50,
                                 y=np.arange(-1, 40),
                                 line=dict(
                                     color=color_discrete_sequence[2015-i]
                                 ),
                                 name=str(i)+' Mean',
                                 showlegend=False,
                                 mode='lines',
                                 hovertemplate=round(mean_score, 2)),
                      row=1, col=i-2014)

    # update the entire subplots layout
    fig.update_layout(height=400, title='Happiness Score Yearly Distribution',
                      legend_title_text='Year', yaxis=dict(range=[0, 30]),
                      hovermode='x unified')
    fig.show()

    # heatmap
    # correlation of each columns with respect another
    # column in the happiness data
    corr = happiness_data.corr().apply(lambda x: round(x, 2))

    # create high resolution figure
    fig, ax = plt.subplots(1, 1, dpi=200)
    ax = sns.heatmap(corr, square=True, annot=True, cmap='YlGnBu')

    # set the orientation of the x labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=40,
        horizontalalignment='right'
    )
    ax.set_title('Happiness Data Correlation', fontsize=12)
    fig.savefig('./img/Happiness_Data_Correlation.png', bbox_inches = "tight")

    # violion plot of regional happiness score distribution
    fig, ax = plt.subplots(1, 1, dpi=200)
    ax = sns.violinplot(x=happiness_data['happiness score'],
                        y=happiness_data['region'])
    ax.set_xlabel('Happiness Score')
    ax.set_ylabel('Region')
    ax.set_title('Happiness Score Regional Distribution')
    fig.savefig('./img/Happiness_Distribution_Violin.png', bbox_inches = "tight")

    # 3D choropleth map
    # create choropleth map data
    # locations by country
    data = go.Choropleth(locations=happiness_data['country'],
                         locationmode='country names',
                         z=happiness_data['happiness score'],
                         text=happiness_data['country'],
                         colorbar={'title': 'Happiness Score'},
                         colorscale=px.colors.sequential.Pinkyl)

    # display ocean on the map
    # projection is orthographic
    # show grid for both lon and lat
    layout = go.Layout(title='Happiness Score Distribution',
                       geo=dict(showframe=False,
                                showocean=True,
                                projection={'type': 'orthographic'},
                                oceancolor='rgb(0,119,190)',
                                lonaxis=dict(showgrid=True,
                                             gridcolor='rgb(102,102,102)'),
                                lataxis=dict(showgrid=True,
                                             gridcolor='rgb(102,102,102)')))
    # plotting
    fig = go.Figure(data=data,
                    layout=layout)
    fig.show()


# Question 2
# Which five countries have the most happiness score? Which five countries
# have the least happiness score? What are the geolocations of these countries?
def q2(happiness_data):
    '''
    q2 takes a pandas dataframe as a parameter, and then
    explores the five most happy countries and the five least happy
    countries, and plots a world map that shows the geolocations of these
    countries, most happy country will be shown in pink, while least
    happy country will be in purple.
    '''
    avg_happiness = happiness_data.groupby(['country'])['happiness score']\
        .mean()

    print('Happiest Countries')
    print(avg_happiness.nlargest(5))
    print('--------------------------------')
    print('Least Happy Countries')
    print(avg_happiness.nsmallest(5))

    # read the country GeoJSON data from github
    countries = gpd.read_file('https://raw.githubusercontent.com/'
                              + 'Benjaminnnnnn/'
                              + 'CSE-163-FINAL-PROJECT/master/data/'
                              + 'countries.geojson')
    countries = countries[countries['ADMIN'] != 'Antarctica']

    # getting the five most happy countries in the world
    top_5_happy_countries = countries[(countries['ADMIN'] == 'Denmark') |
                                      (countries['ADMIN'] == 'Norway') |
                                      (countries['ADMIN'] == 'Finland') |
                                      (countries['ADMIN'] == 'Switzerland') |
                                      (countries['ADMIN'] == 'Iceland')]

    # getting the five least happy countries in the world
    top_5_sad_countries = countries[(countries['ADMIN'] == 'Burundi') |
                                    (countries['ADMIN'] == 'Central'
                                    + 'African Republic') |
                                    (countries['ADMIN'] == 'Syria') |
                                    (countries['ADMIN'] == 'South Sudan') |
                                    (countries['ADMIN'] == 'Rwanda')]

    # making a world graph, most happy country in pink
    # least happy country in purple
    fig, ax = plt.subplots(1, figsize=(15, 7))
    countries.plot(ax=ax, color='#EEEEEE')
    top_5_happy_countries.plot(color='pink', ax=ax)
    top_5_sad_countries.plot(color='purple', ax=ax)
    plt.savefig('./img/Happy_And_Sad_Countries_Map.png', bbox_inches = "tight")


# Question 3
# What are the trends of happiness among different countries? Are they
# increasing, decreasing, or fluctuating?
def q3(happiness_data):
    '''
    q3 takes a pandas dataframe as a parameter, and then
    attempts to find the trends of happiness among different representative
    countries (i.e. five happiest countries and five saddest countries).
    It will plot the graph using seaborn first. And then it will attempt
    to use plotly to improve the visualization.
    '''
    # Five happiest/saddest countries in 2019
    happy_five_2019 = happiness_data[happiness_data['year'] == 2019]\
        .nlargest(5, 'happiness score')
    happy_five_2019 = happy_five_2019['country'].tolist()

    sad_five_2019 = happiness_data[happiness_data['year'] == 2019]\
        .nsmallest(5, 'happiness score')
    sad_five_2019 = sad_five_2019['country'].tolist()

    # Five happiest/saddest countries overall from 2015-2019
    # Sort rows based on happiness score, then drop
    # duplicates from country column
    happy_five = happiness_data.sort_values(['happiness score'],
                                            ascending=False)\
                               .drop_duplicates(subset='country')\
                               .nlargest(5, 'happiness score')
    happy_five = happy_five['country'].tolist()

    sad_five = happiness_data.sort_values(['happiness score'])\
                             .drop_duplicates(subset='country')\
                             .nsmallest(5, 'happiness score')
    sad_five = sad_five['country'].tolist()

    # trying to plot with seaborn first
    # Plot the trends for the five happiest countries (in 2019)
    years = [2015, 2016, 2017, 2018, 2019]
    happy_countries = happiness_data[happiness_data['country']
                                     .isin(happy_five_2019)]
    ax = sns.lineplot(x="year", y="happiness score",
                      hue='country', data=happy_countries)
    ax.set_xticks(years)

    # Plot the trends for the five saddest countries (in 2019)
    sad_countries = happiness_data[happiness_data['country']
                                   .isin(sad_five_2019)]
    ax = sns.lineplot(x="year", y="happiness score",
                      hue='country', data=sad_countries)
    ax.set_xticks(years)

    # Plot the trends for the five happiest countries (2015-2019)
    happy_overall = happiness_data[happiness_data['country'].isin(happy_five)]
    ax = sns.lineplot(x="year", y="happiness score",
                      hue='country', data=happy_overall)
    ax.set_xticks(years)

    # Plot the trends for the five saddest countries (2015-2019)
    sad_overall = happiness_data[happiness_data['country'].isin(sad_five)]
    ax = sns.lineplot(x="year", y="happiness score",
                      hue='country', data=sad_overall)
    ax.set_xticks(years)

    # now let's use plotly to improve the visualization

    # plotting five happiest countries
    title = 'Five Happiest Countries: Happiness Over Time'
    RdPu = px.colors.sequential.RdPu

    colors = [RdPu[2], RdPu[3], RdPu[4], RdPu[6], RdPu[8]]

    fig = go.Figure()
    for i in range(5):
        df = happy_overall[(happy_overall['country']) == happy_five[i]]
        fig.add_trace(go.Scatter(x=df['year'], y=df['happiness score'],
                                 mode='lines', name=happy_five[i],
                      line=dict(color=colors[i])))
        # points
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['happiness score'],
            mode='markers',
            marker=dict(color=colors[i]),
            showlegend=False
        ))

    fig.update_layout(
        autosize=False,
        height=600,
        width=1100,
        showlegend=False,
        xaxis=dict(
            tickmode='linear',
            tick0=2015,
            dtick=1,
            showline=True,
            showgrid=False,
            ticks='outside',
            color='grey'
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            color='grey'
        ),
        margin=dict(
            autoexpand=True,
            r=175
        ),
        plot_bgcolor='white'
    )

    annotations = []
    # Adding labels
    for country, color in zip(happy_five, colors):
        df = happy_overall[(happy_overall['country']) == country]
        y_data = df['happiness score'].tolist()

        # labeling the right_side of the plot
        annotations.append(dict(xref='paper', x=0.95, y=y_data[4],
                                xanchor='left', yanchor='middle',
                                text='{0:.2f} '.format(y_data[4]) + country,
                                font=dict(size=16,
                                          color=color),
                                showarrow=False))

    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.0,
                            xanchor='left', yanchor='bottom',
                            text=title,
                            font=dict(size=30,
                                      color='grey'),
                            showarrow=False))

    # Y-axis Label
    annotations.append(dict(xref='paper', yref='paper', x=-0.07, y=0.5,
                            xanchor='left', yanchor='middle',
                            text='Happiness Score',
                            textangle=270,
                            font=dict(size=16, color='grey'),
                            showarrow=False))

    # X-axis Label
    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                            xanchor='center', yanchor='top',
                            text='Years',
                            font=dict(size=16, color='grey'),
                            showarrow=False))

    fig.update_layout(annotations=annotations)

    fig.show()

    # plotting five saddest countries
    title = 'Five Saddest Countries: Happiness Over Time'
    YGB = px.colors.sequential.YlGnBu
    colors = [YGB[2], YGB[3], YGB[5], YGB[6], YGB[8]]

    fig = go.Figure()
    for i in range(5):
        df = sad_overall[(sad_overall['country']) == sad_five[i]]
        fig.add_trace(go.Scatter(x=df['year'], y=df['happiness score'],
                                 mode='lines', name=sad_five[i],
                                 line=dict(color=colors[i])))
        # points
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['happiness score'],
            mode='markers',
            marker=dict(color=colors[i]),
            showlegend=False
        ))

    fig.update_layout(
        autosize=False,
        height=600,
        width=1100,
        showlegend=False,
        xaxis=dict(
            tickmode='linear',
            tick0=2015,
            dtick=1,
            showline=True,
            showgrid=False,
            ticks='outside',
            color='grey'
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            color='grey'
        ),
        margin=dict(
            autoexpand=True,
            r=175
        ),
        plot_bgcolor='white'
    )

    annotations = []
    # Adding labels
    for country, color in zip(sad_five, colors):
        df = sad_overall[(sad_overall['country']) == country]
        y_data = df['happiness score'].tolist()
        '''
        # labeling the left_side of the plot
        annotations.append(dict(xref='paper', x=0.05, y=y_data[0],
                                xanchor='right', yanchor='middle',
                                text='{0:.2f}'.format(y_data[0]),
                                font=dict(size=9, color=color),
                                showarrow=False))
                                '''
        # labeling the right_side of the plot
        annotations.append(dict(xref='paper', x=0.95, y=y_data[len(y_data)-1],
                                xanchor='left', yanchor='middle',
                                text='{0:.2f} '.format(y_data[len(y_data)-1])
                                     + country,
                                font=dict(size=16,
                                          color=color),
                                showarrow=False))

    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.0,
                            xanchor='left', yanchor='bottom',
                            text=title,
                            font=dict(size=30,
                                      color='grey'),
                            showarrow=False))

    # Y-axis Label
    annotations.append(dict(xref='paper', yref='paper', x=-0.07, y=0.5,
                            xanchor='left', yanchor='middle',
                            text='Happiness Score',
                            textangle=270,
                            font=dict(size=16, color='grey'),
                            showarrow=False))

    # X-axis Label
    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                            xanchor='center', yanchor='top',
                            text='Years',
                            font=dict(size=16, color='grey'),
                            showarrow=False))

    fig.update_layout(annotations=annotations)

    fig.show()


# Question 4
# What are some measures that contribute to a higher happiness score? And
# what are some measures that contribute to a lower happiness score?
def q4(happiness_data):
    '''
    q4 takes a pandas dataframe as a parameter, and then
    builds a DecisionTreeClassifier to determine the importance of each
    measures
    (six key factors, e.g. GDP per capita, family support, health, etc.)
    in countributing to happiness of a country. The original data is not
    labeled, and we don't have a very academic definition of happy countries.
    But for simplicity, we will just use the mean happiness score to separate
    happy and not happy country. In other word, if Score(country) < mean
    happiness score, label = 1, else, label = -1. For testing the
    reliability of our model, we will withhold the latest data from 2019 as
    our test set.
    '''
    labeled_data = []

    for i in range(2015, 2020):
        data = happiness_data[happiness_data['year'] == i].copy()
        mean = data['happiness score'].mean()
        data['label'] = data['happiness score']\
            .apply(lambda x: -1 if x < mean else 1)
        labeled_data.append(data)

    labeled_data = pd.concat(labeled_data)

    # split the data, 80:20 percent
    train_mask = (labeled_data['year'] >= 2015)\
        & (labeled_data['year'] <= 2018)
    test_mask = labeled_data['year'] == 2019

    train = labeled_data[train_mask].copy()
    test = labeled_data[test_mask].copy()

    # the shape of our training and testing data
    print('Train Shape:', train.shape)
    print()
    print('Test Shape:', test.shape)

    # Let's see how many null values are in the our dataset
    print('Null Values in Training Data')
    print(train.isnull().sum())
    print()
    print('Null Values in Testing Data')
    print(test.isnull().sum())

    # we can simple drop the single missing observation since
    # it won't affect our analysis too much
    train = train.dropna()
    test = test.dropna()

    # extract features
    features = ['GDP per capita',
                'family',
                'freedom',
                'generosity',
                'government corruption',
                'life expectancy',
                'year']

    target = 'label'

    # making a naive model to visualize
    small_tree_model = DecisionTreeClassifier(max_depth=2)
    small_tree_model.fit(train[features], train[target])

    def draw_tree(tree_model, features):
        """
        Takes a tree model and list of features as parameters, and then
        visualizes a Decision Tree
        """
        tree_data = tree.export_graphviz(tree_model,
                                         impurity=False,
                                         feature_names=features,
                                         class_names=tree_model.classes_
                                         .astype(str),
                                         filled=True,
                                         out_file=None)
        graph = graphviz.Source(tree_data)
        graph.render(filename='./img/small_tree', format='png',
                     quiet=True)

    draw_tree(small_tree_model, features)

    # using the first six samples of the data labeled 1 and -1
    samples = pd.concat([test[test['label'] == 1][0:3],
                         test[test['label'] == -1][0:3]])

    print()
    print('Testing Accuracy')
    print(accuracy_score(y_pred=small_tree_model.predict(samples[features]),
                         y_true=samples[target]))
    print()
    print('Training Accuracy')
    print(accuracy_score(y_pred=small_tree_model.predict(train[features]),
                         y_true=train[target]))

    # The accuracy of our small_tree_model is apparently underfitting the data,
    # so we need to make a better model to interpret the data. To do this, we
    # use a GridSearchCV to find the best hyperparamters.
    # Since we have a small dataset, we can use LeaveOneOut cross validation
    # method here to prevent overfitting our model.
    max_depth = [1, 3, 5, 7, 10, 13, 15, 20]
    min_samples_leaf = [1, 5, 10, 15, 50, 100, 200]

    hyperparameters = {
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf
                      }

    clf = GridSearchCV(estimator=DecisionTreeClassifier(),
                       cv=LeaveOneOut(),
                       param_grid=hyperparameters,
                       return_train_score=True)
    clf.fit(X=train[features], y=train[target])

    # best parameters
    print('Best Parameters:', clf.best_params_)
    print()

    # evaluate the model
    print('Testing Accuracy')
    print(accuracy_score(y_pred=clf.best_estimator_.predict(test[features]),
                         y_true=test[target]))
    print()
    print('Training Accuracy')
    print(accuracy_score(y_pred=clf.best_estimator_.predict(train[features]),
                         y_true=train[target]))

    def print_feature_importances(model, features):
        '''
        Takes a ML model and list of features as parameters, and then
        prints the importances of given features in the model
        '''
        print('Feature Importance')
        coefficients = list(zip(features, model.feature_importances_))
        print(*coefficients, sep='\n')

    # print the importance of each measure
    print_feature_importances(clf.best_estimator_, features)

    def plot_scores(clf, hyperparameters, score_key, name, showscale):
        '''
        Takes a GridSearchCV object clf, dict of hyperparameters, the score
        key, type of set (train/test), and boolean showscale if we would like
        the scale to be shown. Plots the scores with the given parameters
        onto the figure.
        '''
        cv_results = clf.cv_results_
        scores = cv_results[score_key]
        scores = scores.reshape(len(hyperparameters['max_depth']),
                                len(hyperparameters['min_samples_leaf']))
        max_depth = cv_results['param_max_depth'].reshape(scores.shape)\
            .data.astype(int)
        min_samples_leaf = cv_results['param_min_samples_leaf']\
            .reshape(scores.shape).data.astype(int)

        return go.Surface(x=max_depth, y=min_samples_leaf, z=scores,
                          showscale=showscale, colorscale='YlGnBu', name=name)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Train Accuracy', 'Test Accuracy'],
                        specs=[[{'type': 'surface'},
                                {'type': 'surface'}]])
    fig.add_trace(plot_scores(clf, hyperparameters,
                              'mean_train_score', 'Train', True),
                  row=1, col=1)
    fig.add_trace(plot_scores(clf, hyperparameters,
                              'mean_test_score', 'Test', False),
                  row=1, col=2)

    # update x axis title and y axis title for both subplots
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Max Depth'
            ),
            yaxis=dict(
                title='Min Samples Leaf'
            ),
            zaxis=dict(
                title='Accuracy'
            ),
            camera_eye=dict(
                x=-2, y=-2, z=0.8
            )
        ),
        scene2=dict(
            xaxis=dict(
                title='Max Depth'
            ),
            yaxis=dict(
                title='Min Samples Leaf'
            ),
            zaxis=dict(
                title='Accuracy'
            ),
            camera_eye=dict(
                x=-2, y=-2, z=0.8
            )
        )
    )

    fig.show()


def main():
    '''
    Processes dataset and runs methods to conduct analysis of data.
    '''
    # read data from my github repository
    base_url = 'https://raw.githubusercontent.com/'\
            + 'Benjaminnnnnn/CSE-163-FINAL-PROJECT/master/data'
    # data to be read
    files = [base_url + '/' + str(i) + '.csv' for i in range(2015, 2020)]

    # read data
    dataframes = processing.read_data(base_url, files)

    # add year column for each dataframe
    dataframes = processing.add_year(dataframes)

    # create the columns to be extracted for data analysis
    # store the regex pattern of column and its full name in
    # a pair of tuple, [0] for pattern, [1] for full name
    columns = list(map(lambda x: ('(?i).*' + x[0] + '(?i).*', x[1]),
                   [('rank', 'happiness rank'),
                    ('country', 'country'),
                    ('score', 'happiness score'),
                    ('GDP', 'GDP per capita'),
                    ('family', 'family'),
                    ('social', 'family'),
                    ('freedom', 'freedom'),
                    ('health', 'life expectancy'),
                    ('corruption', 'government corruption'),
                    ('trust', 'government corruption'),  # same as "corruption"
                    ('generosity', 'generosity'),
                    ('year', 'year')
                    ])
                   )

    # filter out the columns needed for our analysis
    dataframes = processing.column_filter(dataframes, columns)

    # rename the columns so that we can concatenate all the dataframes
    dataframes = processing.rename_cols(dataframes, columns)

    # concatenate the list of dataframes into one big dataframe for analysis
    happiness_data = processing.concatenate_dataframes(dataframes)

    # add region column to the data
    happiness_data = processing.add_region(happiness_data)

    q1(happiness_data)
    print('Q1 Sucess')
    q2(happiness_data)
    print('Q2 Success')
    q3(happiness_data)
    print('Q3 Success')
    q4(happiness_data)
    print('Q4 Success')


if __name__ == '__main__':
    main()
