---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="i4aLXKwOz73V" -->
# Movie recommendation using Matrix Factorization
## IVADO - Online course on Recommender Systems 

Tutorial adapted from content originally created for IVADO's Workshop on Recommender Systems, August 2019.
<!-- #endregion -->

<!-- #region id="lTVq2aDdz73W" -->
# 1. Introduction

This tutorial is an introduction to recommender systems. Specifically, we are going to implement an end-to-end recommender system pipeline based on the matrix factorization algorithm. 

Since this algorithm is strongly associated with the <a href="https://en.wikipedia.org/wiki/Netflix_Prize">Netflix Prize</a>, we will use the <a href="https://grouplens.org/datasets/movielens/">MovieLens</a> dataset to train our model and conduct our experiments. 

The task in this dataset is to recommend movies that users will most likely enjoy, based on their historical movie preferences. This task is also an important one, as it is still widely used to benchmark progress in the recommender systems research community. With that, let's get things started!


<!-- #endregion -->

<!-- #region id="_FJKuH-Rz73Z" -->
## 1.1 Installing libraries

Before we begin, we must make sure to install the libraries for the tutorial. To do this, we can use Python's package installer `pip`. Now go ahead and execute the following cell by selecting it and clicking `shift`+`Enter`.
<!-- #endregion -->

```python id="3iqjoZDDz73a" colab={"base_uri": "https://localhost:8080/", "height": 115} outputId="8d7a1794-5e3f-4535-aa40-d609a2102085"
!rm -rf RecSys-Workshop
!git clone https://github.com/dorajam/RecSys-Workshop

```

<!-- #region id="4HABufSMz73f" -->
To confirm that the installations have completed successfully, import all the libraries and modules we will use for this tutorial by running the next cell.
<!-- #endregion -->

```python id="8rKz4rD0z73g"
import numpy as np
import pandas as pd

# Data vizualisation
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

# Add path to find where to import our custom modules from
sys.path += ['RecSys-Workshop/tutorials/']

# Pre-built module with some helper functions
import utilities as utl
```

<!-- #region id="TOiy1nZ9z73l" -->
We have also prepared some boilerplate functions that we have grouped together in the `utilities` module. It is not necessary for you to look at these in order to complete the tutorial, but if you are curious, it is definitely a good idea to take a look at them!
<!-- #endregion -->

<!-- #region id="9YA2A1vqz73m" -->
## 1.2 Task setup and Data

The objective of a recommender system is to model users' historical behaviors such that we can predict what an individual user will most likely enjoy in the future. In short, we would like to create recommendations that are personalized to each user's interest.

The concept of an item might seem a bit fuzzy at first. But in fact, in the world of recommender systems, we always speak about users and items. Items can be anything a user interacts with, ranging from: products, movies, social media posts, news articles, search results, Instagram photos, restaurants, Pinterest pins and the list goes on. Not so surprisingly, majority of internet companies we interact with on a daily basis run some sort of recommender system in the background.

These recommender systems try to predict what will the most relevant item to show us given our user-item interactions. These interactions are often categorized as follows: implicit signals (e.g. likes, views, searches, purchases, installations, music listening behavior etc.) and explicit singals (e.g. ratings, reviews). Both types of signals can be used to help us define how to capture the users' interest, i.e. the user preference. And naturally, some signals suggest stronger proxies for user preferences (e.g. purchases, ratings) than others (e.g. views, clicks, time spent looking at an item).

<img src = "https://user-images.githubusercontent.com/13997178/90336890-0f8fba00-dfdf-11ea-9e32-d6b00988bd10.png" width = "350">

We can collect these user-item interactions to form a user preference vector, i.e. a vector that contains all historical interactions of the user with items that are relevant to our given task.

But how do we then use these user preference vectors to generate personalized recommendations?

Do we need to use all historical interactions? Doesn't user taste change over time? What if recommendations reinforce a user's previous choices?
Do we want to recommend complementary, sustitute or independent items from what the user has interacted with previously? 
Shall we try instead to show her items that she has not yet been exposed to? 
How do we balance how much exposure items get that are already very popular? What to do in cold-start scenarios when a new user or new item joins our platform?

These are just few of the many considerations that one would have to think about before designing a recommender system for a downstream use case. In fact, there is an outstanding community that actively conducts research on some of these questions (e.g. see [RecSys](https://recsys.acm.org/)). There is not a single answer to these considerations - it really just depends on your individual use case. However, the answer will affect not only which models you will use, but also what data you will feed to your model, and how you will set up your objective.

In the context of this tutorial, we will focus on recommending movies that users will like according to their past movie ratings. In order to carry out this task, we will use users' movie ratings, some relevant sociodemographic data as well as various features describing movies. Finally, we can refine the diagram as follows:

<img src = "https://user-images.githubusercontent.com/13997178/90336926-4b2a8400-dfdf-11ea-8379-0f38fc104e85.png" width = "750">
<!-- #endregion -->

<!-- #region id="E-h67-Loz73n" -->
## 1.3 The MoviesLens dataset(s)

The data used here consist of more or less 100k movie evaluations by 943 users. Over 1,6k movies are available. In addition to the 100k evaluations, additional information related to users and movies is available.

We will use three different datasets to carry out our analyses:

<ul>
<li> Users : related to users' characteristics,
<li> Movies : related to movies' characteristics,
<li> Ratings : containing over 100k evaluations.
</ul>

We used the <a href="https://pandas.pydata.org/">Pandas</a> library in order to download and manipulate the datasets.
<!-- #endregion -->

<!-- #region id="OKlYRrdyz73n" -->
### 1.3.1 Users: Download and preprocessing
<!-- #endregion -->

```python id="l_vJZJVnz73o" colab={"base_uri": "https://localhost:8080/", "height": 191} outputId="909d8e03-259e-49a3-985f-ae921459c706"
# Data location
ROOT_DIR = 'RecSys-Workshop/'
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-100k/')

# Read dataset
users = pd.read_csv(os.path.join(DATA_DIR, 'u.user'), sep='|', header=None, engine='python', encoding='latin-1')

# Columns describing user characteristics
users.columns = ['Index', 'Age', 'Gender', 'Occupation', 'Zip code']

# Quick overview
users.head()
```

```python id="Hh0wLuEBMAaN" colab={"base_uri": "https://localhost:8080/", "height": 33} outputId="e7edd107-c6c3-43db-9386-ccce505494ef"
print('Number of users x features:', users.shape)
```

<!-- #region id="ImaeouHkz78d" -->
Before presenting some descriptive statistics related to the population, we transform the users' data in a <a href="https://en.wikipedia.org/wiki/List_(abstract_data_type)">list</a> in order to be able to handle them more easily. 

We first encode users' gender - originally stored as a string - with either 0 or 1. 
Then similarly, since occupations are also recorded as a string value, we would like to instead encode this into a binary array indicating the presence of each occupation.
Finally, we concatanate each user feature (i.e. age, gender and occupation) into a list per each user.
<!-- #endregion -->

```python id="3y4qvczKz78e" colab={"base_uri": "https://localhost:8080/", "height": 66} outputId="83aabcba-b989-459b-fdbb-f72811166ec7"
# Number of users
nb_users = len(users)

# Gender: Convert 'M' and 'F' to 0 and 1
gender = np.where(np.matrix(users['Gender']) == 'M', 0, 1)[0]

print('Shape of gender features:', gender.shape)

# Occupation
occupation_name = np.array(pd.read_csv(os.path.join(DATA_DIR, 'u.occupation'), 
                                            sep='|', header=None, engine='python', encoding='latin-1').loc[:, 0])

# Boolean transformation of user's occupation
occupation_matrix = np.zeros((nb_users, len(occupation_name)))

for k in np.arange(nb_users):
    occupation_matrix[k, occupation_name.tolist().index(users['Occupation'][k])] = 1

print('Shape of user occupation matrix (num of users x num of occupations):', occupation_matrix.shape)

# Concatenation of the sociodemographic variables 
user_attributes = np.concatenate((np.matrix(users['Age']), np.matrix(gender), occupation_matrix.T)).T.tolist()

print('Shape of final user attribute matrix: (list of users with 23 features):', len(user_attributes), len(user_attributes[0]))

```

<!-- #region id="98e7SN9qz78v" -->
We then explore the descriptive statistics of the users. These include information related to age (continuous variable), gender (binary variable) and occupation of each user (21, all binary).
<!-- #endregion -->

<!-- #region id="t3jJXODCz78w" -->
#### Descriptive statistics related to users'  *age*.
<!-- #endregion -->

```python id="snuJ7gV_z78w" colab={"base_uri": "https://localhost:8080/", "height": 76} outputId="696fbfd9-3b28-4c38-906e-7875a8b40108"
pd.DataFrame(users['Age'].describe()).T
```

<!-- #region id="CcDMUpJZz7-m" -->
#### Percentage of users per *gender*.
<!-- #endregion -->

```python id="dZ1DDpppz7-n" colab={"base_uri": "https://localhost:8080/", "height": 307} outputId="23fef21e-01ef-4fb6-928c-c0496cb762b6"
utl.barplot(['Women', 'Men'], np.array([np.mean(gender) , 1 - np.mean(gender)]) * 100, 
            'Sex', 'Percentage (%)', "User's gender", 0)
```

<!-- #region id="S7WfQi04z7-q" -->
#### Percentage of users per *occupation*.
<!-- #endregion -->

```python id="sRcJ4jYbz7-r" colab={"base_uri": "https://localhost:8080/", "height": 368} outputId="25ac6cab-d807-4a5d-ee31-758b3497cb3e"
attributes, scores = utl.rearrange(occupation_name, np.mean(occupation_matrix, axis=0) * 100)
utl.barplot(attributes, scores, 'Occupation', 'Percentage (%)', "Users' occupation", 90)
```

<!-- #region id="lZ0CCCLJhHmj" -->
It is now a good time to pause and reflect what these various user characteristics tell us. We see that majority of our users are men, and 20+ % of them are students, with a median age of 31. What does that mean for our model?
Ideally, if we were to use this dataset to train a model on it, it would reflect the overall popular of movie watchers we will serve recommendations to. 
<!-- #endregion -->

<!-- #region id="2Y6Hf7EphHmk" -->
##### Question 1
What do you think if tomorrow we launch a new campaign where all movies can be watched for free? How do you think our recommendations trained on this dataset would change? What if we launch another campaign where we increase the price of renting a movie by 3x? How would the underlying population of movie-watchers change?
<!-- #endregion -->

<!-- #region id="CcxqNGfKhHml" -->
##### Answer 1
Generally, we want to make sure that users we use in our training set are representative of future users in terms of their user behaviors (i.e., their movie watching and rating behaviors). This is important to ensure that our model will be able to generalize to new users that weren't part of our training set. Now, if we launch a campaign to decrease the price of movies, our user base may become much younger, whereas if we increase the price of renting a movie, the user base may become older. On a final note, if we ever encounter a situation where the user base over time changes due to various factors, it is important to think about periodically retraining our model on the new set of users to account for their new types of behaviors as well.
<!-- #endregion -->

<!-- #region id="AMXYOKLOz7-t" -->
### 1.3.2 Movies: Download and preprocesssing

In the same way, we will now process and explore the data associated with movies. For each movie, we have the *title*, the *release date* in North America, as well as the corresponding *genres*.
<!-- #endregion -->

```python id="C7LAOhwqz7-u" colab={"base_uri": "https://localhost:8080/", "height": 373} outputId="03f847ec-99b6-448d-b652-6a847c8846bc"
# Read dataset
movies = pd.read_csv(os.path.join(DATA_DIR, 'u.item'), sep='|', header=None, engine='python', encoding='latin-1')

# Number of movies
nb_movies = len(movies)
print('The number of movies is: ', nb_movies)

# Genres
movies_genre = np.matrix(movies.loc[:, 5:])
movies_genre_name = np.array(pd.read_csv(os.path.join(DATA_DIR, 'u.genre'), sep='|', header=None, engine='python', encoding='latin-1').loc[:, 0])

# Quick overview
movies.columns = ['Index', 'Title', 'Release', 'The Not a Number column', 'Imdb'] + movies_genre_name.tolist()
movies.head()
```

<!-- #region id="FFIZ_Wqmz7_E" -->
#### Percentage of movies per *genre*
<!-- #endregion -->

```python id="WoFlTXY4z7_E" colab={"base_uri": "https://localhost:8080/", "height": 365} outputId="0e004e94-7b6b-4eb6-910b-7adf5d131674"
attributes, scores = utl.rearrange(movies_genre_name, 
                                   np.array(np.round(np.mean(movies_genre, axis=0) * 1, 2))[0])
utl.barplot(attributes, np.array(scores) * 100, xlabel='Genre', ylabel='Percentage (%)', 
            title=" ", rotation = 90)
```

<!-- #region id="wGJIb9Dhz7_Q" -->
### 1.3.3 Ratings: Download and preprocessing

The dataset based on users ratings consists of approximately 100k lines (one evaluation per line) where the following are presented: the *user identification number*, the *identification number of the movie*, its associated *rating* and *a time marker*. The training and test sets were provided as is, that is, we do not need to build them ourselves, and have 80k and 20k evaluations respectively.

For practical reasons, we convert the database as a list using our `convert` util function.
<!-- #endregion -->

```python id="nx9oYW5gz7_Q" colab={"base_uri": "https://localhost:8080/", "height": 50} outputId="d2182a44-5665-43a4-c82c-ad15b09262d7"
training_set = np.array(pd.read_csv(os.path.join(DATA_DIR, 'u1.base'), delimiter='\t'), dtype='int')
testing_set = np.array(pd.read_csv(os.path.join(DATA_DIR, 'u1.test'), delimiter='\t'), dtype='int')

print('Example sample (user idx, movie idx, rating, timestamp: ', training_set[0])
print('Shape of original training and test set with shape:     ', training_set.shape, testing_set.shape)

```

```python id="jy25j9S0QpR1" colab={"base_uri": "https://localhost:8080/", "height": 50} outputId="59409f89-3637-41af-a948-35aabfbcfb1b"
train_set = utl.convert(training_set, nb_users, nb_movies)
test_set = utl.convert(testing_set, nb_users, nb_movies)

print('Shape of final training set: (list of users x list of all movies):', len(train_set), len(train_set[0]))
print('Shape of final test set:     (list of users x list of all movies):', len(test_set), len(test_set[0]))

```

```python id="2oScuzmGhHmy"
# np.array(train_set).shape
```

<!-- #region id="7mvWDUNez7_W" -->
As we did before, we can get descriptive statistics associated with the evaluations. At first, it might be interesting to study the average trends of users.
<!-- #endregion -->

```python id="72s3zgm_z7_W" colab={"base_uri": "https://localhost:8080/", "height": 76} outputId="8e20a0c0-1310-42f2-ac0c-cdf6548fd4b2"
train_matrix = np.array(train_set)
assert train_matrix.shape == (943, 1682)

binarized_train_matrix = np.where(train_matrix > 0 , 1, 0)

num_movies_watched = np.sum(binarized_train_matrix, axis=1) ## sum across movies for each user
pd.DataFrame(num_movies_watched).describe().T
```

<!-- #region id="wSf9HN2khHm2" -->
I.e. The median user in our training set has watched 50 movies, with smallest number of movies watched being 4, and the most movies watched being 685.
<!-- #endregion -->

<!-- #region id="opRzwOHIz7_g" -->
#### Histogram of the number of movies watched per user.
<!-- #endregion -->

```python id="y6tRdKbSz7_h" colab={"base_uri": "https://localhost:8080/", "height": 521} outputId="216fc169-57cb-478e-f8ac-5bbbc260e94b"
sns.set(rc={'figure.figsize':(12,8)})
sns.set(font_scale = 1.5)

plt.title('Empirical distribution of \n the number of movies watched per user')
plt.xlabel('Number of movies watched')
plt.ylabel('Number of users')
plt.hist(num_movies_watched, 100);
```

<!-- #region id="TpTDzpvBz7_k" -->
#### Percentage of users having seen a movie
<!-- #endregion -->

```python id="zGmPJgpbz7_l" colab={"base_uri": "https://localhost:8080/", "height": 76} outputId="e3c6393d-2d10-4ffa-ffd0-655666401253"
movie_popularity = np.mean(binarized_train_matrix, axis=0)  ## axis 0 refers to averaging across users
pd.DataFrame(movie_popularity).describe().T
```

<!-- #region id="X40afYkghHm7" -->
That is, 50% of movies were seen by less than 2% of users, whereas there are movies that were seen by either no user or by 51% of all users.
 
We can also plot this as a histogram and see how many movies were seen by what proportion of the population.

<!-- #endregion -->

```python id="mRgymfcWz7_p" colab={"base_uri": "https://localhost:8080/", "height": 483} outputId="dccc0518-9848-4d92-d59e-e605d5e83cb5"
plt.xlabel('Proportion of the population who watched the movie')
plt.ylabel('Number of Movies')
plt.hist(movie_popularity, 100);
```

<!-- #region id="Qp9zQ9LthHm-" -->
##### Question 2

1. What does the above histogram tell us about our user-item training matrix? 
<!-- #endregion -->

<!-- #region id="RyR8-mqYhHm-" -->
##### Answer 2
1. This histogram tells us that a very small fraction of users have seen a large percentage of movies, whereas majority of users have only seen a few movies. From this, we can determine various trends, for example, could it be that some movies were block busters, and were seen by majority of users? Could it be that most of the movies are only ever seen by a few users (and therefore have a very few ratings)? It's generally a good idea, to familiarize oneself with this matrix, as it reveals how sparse user ratings will across various movies.
<!-- #endregion -->

<!-- #region id="w_WxJwK-z8AD" -->
#### Individual preferences according to the type of movie

We could also look at the behavior of a particular individual. Among other things, we could study if there is a bias associated with her evaluation scheme or what are her cinematographic preferences according to the score awarded.
<!-- #endregion -->

```python id="GCy_zSQUz8AF"
def stats_user(data, movies_genre, user_id):
    
    ratings = data[user_id]
    stats = np.zeros(6)
    eva = np.zeros((6, movies_genre.shape[1]))

    for k in np.arange(len(ratings)):
        index = int(ratings[k])
        stats[index] += 1
        eva[index, :] = eva[index, :] + movies_genre[k]

    return stats, eva
```

```python id="eZHVMziCz8AJ" colab={"base_uri": "https://localhost:8080/", "height": 307} outputId="3e59ee12-2260-416a-af2c-ab7ae1f0506a"
user_id = 0
num_of_star_ratings, genre_based_ratings = stats_user(train_set, movies_genre, user_id)
utl.barplot(np.arange(5) + 1, num_of_star_ratings[1:6] / sum(num_of_star_ratings[1:6]), xlabel='Number of stars', ylabel='Percentage of movies (%)', 
            title=" ", rotation = 0)

```

<!-- #region id="nAI0G1Wnz8A6" -->
##### Question 3

1. How can we test to the existence of bias associated to individual's assessment scheme?
<!-- #endregion -->

<!-- #region id="DNEMA0mchHnD" -->
##### Answer 3
1. A simple way would be to perform a t-test in order to check whether the average score of an individual is significantly different from those of the entire population.
<!-- #endregion -->

<!-- #region id="S4nDaiDoz8A7" -->
## 1.4 Construction of the training and validation sets

In all machine learning tasks, we always begin with a task in mind. In image classification, this task could be to identify if an image contains a cat or a dog. In recommender systems, it is to predict a set of items that the user will likely interact with. We could break this down as a task where we want to predict that an item is relevant to a user.
 
In order to build a model that will perform this task, we need to come up with a metric that will measure how good our trained model is.
For image classification tasks such as recognizing cats or dogs, this metric could be the percentage of images that we correctly classify. In short, we can use accuracy. For the recommendation task, this metric could be the number of retrieved items that our model was able to predict as relevant.
 
We often speak of breaking our data into training, validation and test sets, where we use the training set only for model training, and the validation and test sets for evaluation.
 
But why do we do this? It is because we would like to ensure that our model does not just overfit on characteristics of training dataset, but the model will generalize to observations that it did not encounter in the past.
 
Ultimately, generalizing to unseen datapoints is the very reason we built this model for.
 
Generally, we want to ensure that the characteristics of the training, validation and test sets will resemble what future datapoints will look like. Remember? We talked about this when we looked at the various characteristics of our users, movies and ratings.
 
To make this more concrete, people often build on the assumption that these datasets are *iid*, that is, the datapoints are independent and are identically distributed. That is, datapoints seen in the validation and test sets have the same distribution, but were never seen during training.
 
 
<b>! Note! </b>
 
Though this iid assumption is widely used in machine learning, evaluation in recommender systems gets somewhat tricky.
 
In fact, due to our choice of the recommender model here, the data we will use for evaluation is not linked to a new set of users, but rather, it will contain user-item interactions by the same set of users, but that were not seen during training.
 
As a result, the data associated with the training, validation and test sets are no longer independent, which complicates things theoretically.
 
Though it is good to be aware of this, it is outside the scope of this tutorial. Here we will just naively assume that each observation is independent of each other.
<!-- #endregion -->

```python id="Y9Vgrze6z8A8"
def split(data, ratio, tensor=False):
    train = np.zeros((len(data), len(data[0]))).tolist()
    valid = np.zeros((len(data), len(data[0]))).tolist()

    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] > 0:
                if np.random.binomial(1, ratio, 1):
                    train[i][j] = data[i][j]
                else:
                    valid[i][j] = data[i][j]

    return [train, valid]

train = split(train_set, 0.8)
test = test_set
```

<!-- #region id="pIG0GWqkz8BC" -->
# 2. Recommender systems: Matrix Factorization
<!-- #endregion -->

<!-- #region id="sF4nUjipz8Bh" -->
## 2.1 Model

Matrix factorization (MF) supposes that each observed evaluation $r_{ui}$ for $1 \leq u \leq |U|$ and $1 \leq i \leq |I|$, where $|U|$ and $|I|$ are respectively the number of users and movies, can be estimated with respect to a latent (hidden) model. This model presents the estimate $\hat{r}_{ui}$ of the observed evaluation $r_{ui}$ as:


$$
\begin{align}
\hat{r}_{ui} =  \langle p_{u}, q_{i} \rangle, 
\end{align}
$$

where $\langle \cdot \rangle$ is the scalar product and $p_{u}$ and $q_{i}$ are the latent representations associated to user <i>u</i> and item <i>i</i>. The intuition behind this representation suggests that each evaluation can be estimated by considering a latent characterization of users and items.

For example, let's fix the number of latent variables to 3, and suppose that they are associated with the popularity of the movie at the box office, its duration and finally its level of romance. Let us define the <i>u</i> user as a 15-year-old teenager who loves popular and relatively short horror movies. We can model the associated latent vector by:
$$
\begin{align}
p_{u} = [1, 0, 0]^T.
\end{align}
$$

Suppose now that the movie <i>i</i> turns out to be <i>The Lion King</i> with the following latent modelization:

$$
\begin{align}
q_{i} = [1, 0.5, 0]^T.
\end{align}
$$

The estimation of the evaluation for this user and item according to the latent representations will therefore be:

$$
\begin{align}
\hat{r}_{ui} =  \langle p_u, q_i \rangle = 1.
\end{align}
$$

The main challenge in this type of model is to define the set of latent vectors associated with users, grouped in matrix form by $\mathbf{P}_{|U| \times k} = [p_1, p_2, .. ., p_k]$, and to items, grouped in matrix form by $ \mathbf{Q}_{|I| \times k} = [q_1, q_2, ..., q_k] $.
 
Since the initial problem is to present the most accurate estimates, and thus to calculate $\mathbf{P}$ and $\mathbf{Q}$ so as to minimize the distance between the totality of the observed ratings $r_{ui}$ and their estimate $\hat {r}_{ui}$, we can define the task to accomplish with the following optimization problem:

$$
\begin{align}
\mathbf{P}, \mathbf{Q} = \underset{p, q}{\operatorname{argmin}} \sum_{r_{ui} \neq 0} (r_{ui} - \hat{r}_{ui})^2 = \underset{p, q}{\operatorname{argmin}}  \sum_{r_{ui} \neq 0} (r_{ui} - \langle p_u, q_i \rangle)^2.
\end{align}
$$

We may add a regularization on the latent variables, in order to force the associated vectors to have non-zero components:

$$
\begin{align}
\mathbf{P}, \mathbf{Q} = \underset{p, q}{\operatorname{argmin}} \sum_{r_{ui} \neq 0} \{(r_{ui} - \langle p_u, q_i \rangle)^2 + \lambda(||p_u||^2 + ||q_i||^2)\},
\end{align}
$$

where $\lambda$ is the regularization hyperparameter, <i>weigth decay</i> in deep learning, or Lagrange multiplier in math. Although this last remark seems technical, it should be noted that latent vectors with very few zero values will in turn lead to predicted ratings other than zero. Since we are trying to propose new recommendations, this constraint seems useful to avoid a sparse matrix estimate $\hat{\mathbf{R}}$.

In general, the optimization problem above, which turns out to factorize a sparse matrix, cannot be solved as easily as using the least squares in a linear regression context for example. In this tutorial, we will introduce the stochastic gradient descent algorithm, as one of the approaches to solving this optimization problem to estimate $\mathbf{P}$ and $ \mathbf{Q}$ matrices
<!-- #endregion -->

<!-- #region id="-RkDWAtPz8Bj" -->
## 2.2 Implementation

In order to build a recommendation system based on MF, we have to define some specific functions necessary for this type of algorithm. Overall, we will break down the implementation process into three steps:

1. <b>Learning loop</b>: Iterative optimization process that will estimate how far off the model is from satisfying a given objective, and will make the necessary model updates until a given stopping criterion is reached. 

2. <b> Loss Function</b>: Calculates how far off our model's prediction is from the true observation.

3. <b> Estimation</b>: Estimation of the matrices of factors $\mathbf{P}$ and $\mathbf{Q}$ respectively associated with the users and the items.
<!-- #endregion -->

<!-- #region id="uuSCo_qfPRFc" -->
### 2.2.1 Learning loop

We will now set up the learning loop. This will contain a function that performs a number of iterations to update our model's parameters until a given stopping criterion is reached.
<!-- #endregion -->

<!-- #region id="WZJrzBL_z8Bn" -->
##### Question 4
Let's assume we have access to three functions: prediction, sgd, and loss. We do not need to see how these are implmented, but rather let's just focus on the signiture of these functions for now (you can take a peek in the following code cells).

1. We have a missing condition for when we would like to perform the SGD model update. What is this condition? Can you code it?
2. At the end of each epoch, what metric should we look at to ensure our model is learning? Let's code this. (Do not forget to initialize objects at the beginning of the function.)
3. Why do we care to see this metric?
4. The stopping criterion is rather simple. What's problematic about this?
5. Can you improve on this stopping criterion? Let's try something else. 
<!-- #endregion -->

<!-- #region id="P-d6Fj01hHnJ" -->
#### Answer 4
See code below.
<!-- #endregion -->

```python id="yhxDFNIrz8Bo"
def learn_to_recommend(data, features=10, lr=0.0002, epochs=101, weigth_decay=0.02, stopping=0.001):
    """
    Args:
       data: every evaluation
       features: number of latent variables
       lr: learning rate for gradient descent
       epochs: number of iterations or maximum loops to perform
       weigth_decay: L2 regularization to predict rattings different of 0
       stopping: scalar associated with the stopping criterion
      
     Returns:
       P: latent matrix of users
       Q: latent matrix of items
       loss_train: vector of the different values of the loss function after each iteration on the train
       loss_valid: vector of the different values of the loss function after each iteration not on valid
       """
     
    train, valid = data[0], data[1]
    nb_users, nb_items = len(train), len(train[0])

    # Answer 4.2: Initialization of lists
    loss_train, loss_valid = [], []

    P = np.random.rand(nb_users, features) * 0.1
    Q = np.random.rand(nb_items, features) * 0.1
    
    for e in range(epochs):        
        for u in range(nb_users):
            for i in range(nb_items):

                # Answer 4.1: Code the condition
                if train[u][i] > 0:
                    error_ui = train[u][i] - prediction(P, Q, u, i)
                    P, Q = sgd(error_ui, P, Q, u, i, features, lr, weigth_decay)
                               
        # Answer 4.2: Code the statistics
        loss_train.append(loss(train, P, Q))
        loss_valid.append(loss(valid, P, Q))
        
        if e % 10 == 0:
            print('Epoch : ', "{:3.0f}".format(e+1), ' | Train :', "{:3.3f}".format(loss_train[-1]), 
                  ' | Valid :', "{:3.3f}".format(loss_valid[-1]))

        # Answer 4.4: Stopping criterion
#         if abs(loss_train[-1]) < stopping:
#             break
            
        # Answer 4.5 : New stopping criterion
        if e > 1:
            if abs(loss_valid[-1] - loss_valid[-2]) < stopping:
                break
        
    return P, Q, loss_train, loss_valid
```

<!-- #region id="NgT88hU5z8Bu" -->
### 2.2.2 Loss function

The loss function plays a decisive role in the construction of a predictive model. In fact, it is this same cost function that we will try to optimize by iteratively adjusting the values of the latent matrices $\mathbf{P}$ and $\mathbf{Q}$.

Since we consider that the observed evaluations vary between 1 and 5, the mean squared error (MSE) seems an interesting first option. From a recommender system point of view, we will define the MSE as follows:

$$
\begin{align}
MSE(\mathbf{R}, \hat{\mathbf{R}}) = \frac{1}{n} \sum_{r_{ui} \neq 0} (r_{ui} - \hat{r}_{ui})^2, 
\end{align}
$$

where $\mathbf{R}$ and $\hat{\mathbf{R}}$ are respectively the matrices of observed and predicted ratings <i>n</i> is the number of evaluations.

<!-- #endregion -->

<!-- #region id="OAPyYa9QPDpU" -->

##### Question 5

1. Suppose we want to predict the evaluation of user <i>u</i> for the movie <i> i </i>, how should do it? Implement the prediction function.
2. An important detail is missing in the following `loss` function. Let's correct it!
<!-- #endregion -->

<!-- #region id="pOvfZbDuhHnM" -->
#### Answer 5
See below.
<!-- #endregion -->

```python id="Ft_CJd64z8Bu"
# Answer 5.1:
def prediction(P, Q, u, i):
    """
    Args:
        P: user matrix
        Q: matrix of items
        u: index associated with user u
        i: index associated with item i
    Returns:
        pred: the predicted evaluation of the user u for the item i
    """
    
    return np.dot(P[u,:], Q[i,:])

def loss(data, P, Q):
    """
    Args:
       data: ratings
       P: matrix of users
       Q: matrix of items   
    Returns:
        MSE: observed mean of squared errors 
    """
    errors_sum, nb_evaluations = 0., 0
    nb_users, nb_items = len(data), len(data[0])

    for u in range(nb_users):
        for i in range(nb_items):
        
            # Answer 5.2:
            if data[u][i] > 0:
                errors_sum += pow(data[u][i] - prediction(P, Q, u, i), 2)
                nb_evaluations += 1
                
    return errors_sum / nb_evaluations
```

<!-- #region id="10bI2dV5z8CQ" -->
### 2.2.3 Estimation
 
Parameters' estimates are directly associated with the loss function we are trying to minimize. With matrix factorization, two estimation techniques are available to calculate latent matrices $\mathbf{P}$ and $\mathbf{Q}$ respectively associated to users and items. In both cases, these techniques use the linearity of the matrix factorization model.
 
#### Gradient descent
 
First, we implement stochastic gradient descent (SGD); an iterative method that reviews all non-zero evaluations for each user. Formally, and remembering that the function we are trying to minimize is:
 
$$
\begin{align}
\underset{p, q}{\operatorname{min}} L(\mathbf{R}, \lambda) = \underset{p, q}{\operatorname{min}} \sum_{r_{ui} \neq 0} \{(r_{ui} - \langle p_u, q_i \rangle)^2 + \lambda \cdot (||p_u||^2 + ||q_i||^2)\},
\end{align}
$$
 
we calculate the gradients of the previous equation as a function of $p_u$ and $q_i$:
 
$$
\nabla_{p_{u}} L(\mathbf{R}, \lambda) =  -2q_{i} \cdot \epsilon_{ui} + 2\lambda \cdot p_{u} \quad \text{and} \quad
\nabla_{q_{i}} L(\mathbf{R}, \lambda) =  -2p_{u} \cdot \epsilon_{ui} + 2\lambda \cdot q_{i},
$$
 
where we denote the error by:
 
$$
\epsilon_{ui} = r_{ui} - \hat{r}_{ui}.
$$
 
Finally, for each iteration, as long as the user-item rating is non-zero, we perform the following update on the latent vectors as follows:
 
$$
p_{u}^{(t+1)} \leftarrow p_{u}^{(t)} + \gamma \cdot (2q_{i}^{(t)} \cdot \epsilon_{ui} - 2 \lambda \cdot p_{u}^{(t)}) \\
q_{i}^{(t+1)} \leftarrow q_{i}^{(t)} + \gamma \cdot (2p_{u}^{(t)} \cdot \epsilon_{ui} - 2 \lambda \cdot q_{i}^{(t)}),
$$
 
where $ p_{u}^{(t + 1)}$ is the value of $ p_{u}$ after the $t + 1$ iteration and where $\gamma$ is the learning rate of the descent. That is, we take a step in the opposite direction of the gradients such that we minimize the loss function.
 
#### Note on alternate least squares
 
The second technique is based on Alternate Least Squares (ALS). This method is elegant in that it allows an analytical form. We will not implement it in this workshop.

<!-- #endregion -->

<!-- #region id="8R7uiumPPLaq" -->
##### Question 6
 
1. Given the above equations, can you complete the `sgd` function below that will update our model parameters $\mathbf{P}$ and $\mathbf{Q}$?
<!-- #endregion -->

<!-- #region id="Byk5gumehHnQ" -->
#### Answer 6
See code below.
<!-- #endregion -->

```python id="8ic_KU6uz8Ca"
def sgd(error, P, Q, id_user, id_item, features, lr, weigth_decay):
    """
    Args:
        error: difference between observed and predicted evaluation (in that order)
        P: matrix of users
        Q: matrix of items
        id_user: id_user
        id_item: id_item
        features: number of latent variables
        lr: learning for the descent of the gradient
        weigth_decay: scalar multiplier controlling the influence of the regularization term
       
     Returns:
        P: the new estimate for P
        Q: the new estimate for Q
     """    
    
    
    # Answer 6.1
    for f in range(features):
        P[id_user, f] = P[id_user, f] + lr * (2 * Q[id_item, f] * error - 2 * weigth_decay * P[id_user, f])
        Q[id_item, f] = Q[id_item, f] + lr * (2 * P[id_user, f] * error - 2 * weigth_decay * Q[id_item, f])
      
    return P, Q

```

<!-- #region id="7IwjGI_Bz8Ce" -->
## 2.3 Training

Now that matrix factorization is implemented, we can begin to train the model with different parameters and hyperparameters. The idea here is not to adjust the parameters in such a way as to obtain the best model, but simply to understand the role that they can play, both from the point of view of overfitting and computing time. In fact, there are very few wrong answers.
<!-- #endregion -->

```python id="OOMVmv6-z8DQ" outputId="a0ee44ec-4a5d-48ed-b608-84e22629cfe3"
features = 5
lr = 0.01
epochs = 101
weigth_decay = 0.02
stopping = 0.001

P, Q, loss_train, loss_valid = learn_to_recommend(train, features, lr, epochs, weigth_decay, stopping)
```

<!-- #region id="XAs0VXIoz8DS" -->
Once the model is trained, we can visualize the different learning curves.
<!-- #endregion -->

```python id="qGJBjTy9z8DT" outputId="346c3239-3833-4284-897b-1569f3d7ab15"
x = list(range(len(loss_train)))
k=0

sns.set(rc={'figure.figsize':(12,8)})
sns.set(font_scale = 1.5)

plt.plot(x[-k:], loss_train[-k:], 'r', label="Train")
plt.plot(x[-k:], loss_valid[-k:], 'g', label="Validation")
plt.title('Learning curves')
plt.xlabel('Epoch')
plt.ylabel('MSE')
leg = plt.legend(loc='best', shadow=True, fancybox=True)
```

<!-- #region id="ZdVEiUmyz8DV" -->
##### Question 7
 
1. Was it necessary to iterate that much?
2. What do you observe if we use the criterion we defined in Question 4?
<!-- #endregion -->

<!-- #region id="VCQOV81zhHnY" -->
#### Answer 7
1. Using our first criterion, we see that even though the training loss continues going down, the validation error starts to increase after a few iterations. This is a sign that our model started memorizing the characteristics of the training set, and is losing its ability to generalize to a held out validation set. Because of this, it's generally a good idea to stop traning when we see that there are no further improvement on the validation loss, i.e., the metric we mostly care about.

2. We first changes the threshold to be 0.001, and went back to change the stopping criterion to the one we saw in Question 4. Then we retrained our model. And as we see, with this criterion, we see the training terminating at an earlier epoch, thereby avoiding overfitting on the training set. It's generally a good idea to have a few strategies for early stopping such that we can try a few options and choose the most appropriate condition given our problem!

<!-- #endregion -->

<!-- #region id="N7hI0Mzqz8DV" -->
Now that our model is trained, let's evaluate our model's final performance.

##### Question 8
 
1. Which dataset do we use for this? And why?
2. Can you implement this procedure?
<!-- #endregion -->

<!-- #region id="Np_tPfMphHnZ" -->
#### Answer 8
1. We evaluate our model's performance on the test set, as this dataset was never seen during training, therefore it can give us a good proxy for how well our model is able to generalize to new datapoints. This in fact is exactly what we care for in practice, especially when deploying our model in a real-world application where our objective is to make new recommendations for movies that a given set of users haven't seen before!
2. See code below.
<!-- #endregion -->

```python id="QDqJxtywz8DW" outputId="78611951-2fd0-4afa-c0e8-7930d7e2f9c3"
# Answer 8.2
loss(test, P, Q)

```

<!-- #region id="VY2re70Nz8DZ" -->
## 2.4 Analysis
<!-- #endregion -->

<!-- #region id="H9vsE0-Iz8Da" -->
### 2.4.1 Exploring latent layers
 
Thanks to matrix factorization, it is possible to explore the various latent variables associated with users and items. By the nature of the matrices $\mathbf{P}$ and $\mathbf{Q}$, we can explore the <i>k</i> latent variables by looking at the columns of $\mathbf{P}$ and $\mathbf{Q}$.
 
For example, suppose that the first two latent variables in $\mathbf{Q}$ are the following, where each value in these arrays corresponds to an item:
 
$$
\begin{align}
q_1 &= [-1.0, \ -0.8, \ 0.0, \ 1.0, \ 0.5, \ ...]
\qquad \text {and} \qquad
q_2 = [-1.0, \ 0.8, \ 1.0, \ 0.5, \ -0.8, \ ...].
\end{align}
$$
 
Let's suppose that these underlying items correspond to the following movies:
 
1. The Room (2003),
2. Star Wars: Attack of the clones (2002),
3. Titanic (1997),
4. Citizen Kane (1954),
5. The Nigthmare before Christmas (1993).
 
By mapping these movies according to the associated values ​​of the first two latent variables, we obtain the following graph:
 
<img src = "https://user-images.githubusercontent.com/13997178/91663551-9486d300-eab7-11ea-8e9f-c58398eff9fe.png" width = "500">
 
When doing this, we might begin to see some patterns around what each latent variable might correspond to. In this case, we could imagine that the first latent variable here is associated with how well received the movie was, while the second variable may detect the presence of a superstar.
 
This might be an interesting hypothesis! Let's see if we see similar patterns when we look at the values associated with the matrix of users, $ \mathbf{P}$.
 
Suppose now that the first two latent variables of the $\mathbf{P}$ user matrix have the following values:
 
$$
\begin{align}
p_1 &= [1.0, \ 0.0, \ -0.5, \ 1.0, \ -1.0, \ ...]
\qquad \text{and} \qquad
p_2 = [1.0, \ 0.0, \ 0.5, \ -1.0, \ -0.8, \ ...]
\end{align}
$$
 
And that that each value in these two arrays correspond to the following users.
 
1. Serena,
2. Kali,
3. Neil,
4. Mary,
5. David.
 
We will now map the users according to the values associated with $ p_1 $ and $ p_2 $ vectors. Note, here we considered the same two latent factors, such that we can investigate how they compare to the characterization of the axes that we obtained before:
 
<img src = "https://user-images.githubusercontent.com/13997178/91663557-9e103b00-eab7-11ea-9ee1-0de6a5ac3760.png" width = "500">
 
 
 
This approach could allow us to suggest new movies that have never been evaluated by users simply based on certain characteristics. For example, there is a good chance that Serena loves the upcoming Scorsese <i> The Irish man </i> movie and that Neil looks forward to the new <i> Cat </i>.
We will now propose a function that will facilitate the exploration of latent variables.

<!-- #endregion -->

```python id="yDEWyWn_z8Db"
def explore(movie_titles, latent_matrix, frequency_mask, factor_idx, k):
    """
    Args:
       movie_titles: Pandas Series containing movie titles
        latent_matrix: matrix containing the model parameters for movies
        frequency_mask: boolean array masking non-frequent movies
        factor_idx: index of the latent variable
        k: number of movies to show

    Returns:
        names: movie titles
        scores: associated predicted ratings of movies
    """

    # slice the column to obtain latent variable, then apply mask
    latent_variable = latent_matrix[:, factor] * frequency_mask

    # filter out infrequent movies
    nonzero_indices = np.nonzero(latent_variable)
    movies = np.array(movie_titles)[nonzero_indices][:k]
    latent_variable = latent_variable[nonzero_indices][:k]

    return movies, latent_variable

```

<!-- #region id="Xr2ssQUThHnh" -->
Let's now go ahead and use this function to visualize the results. 
We will do this while only considering movies that have been watched by
at least 10% of all users. To do this, let's use the movie_popularity list we created earlier.
<!-- #endregion -->

```python id="6cKSoGCLhHni" outputId="afbd0a6f-baa6-4824-e527-a1418e025458"
print(movie_popularity)
print(movie_popularity.shape)
```

```python id="nqSJGMbfz8Dg" outputId="a37ba678-76c4-4893-d00e-4d8c44238905"
k = 5
factor = 0
threshold = 0.1
names, scores = explore(movies['Title'], Q, np.where(movie_popularity > threshold, 1, 0), factor, k)

df = pd.DataFrame(np.matrix((names, scores)).T, (np.arange(len(scores)) + 1).tolist())
df.columns = ['Title', 'Latent factor']
df
```

<!-- #region id="nZuzfJHfz8Dg" -->
##### Question 9

1. Can some latent variables be interpretable?
<!-- #endregion -->

<!-- #region id="u8BxmF2WhHnm" -->
#### Answer
1. Essentially, that's what we would like to have. However, this is not guaranteed and interpretability can often depend on things like the number of latent factors we use, or the data itself, among other things. We won’t dive into this topic too much, but since we have the codebase already, feel free to play around with changing the size of the latent dimension and looking at how underlying latent characteristics change as well.
<!-- #endregion -->

<!-- #region id="bNCN7hBpz8Di" -->
# 3. Applications
 
Congratulations on making it so far and on building your first recommender system! We can now put our model to use and generate personalized recommendations for users. In this section we will do just that.
 
We will choose a given user, and consider her preferences. Then we will generate recommendations for her using our trained matrix factorization model with the caveat in mind that we only want to recommend movies that she has not seen in the past.
 

<!-- #endregion -->

<!-- #region id="VZztaYI_Glit" -->
##### Question 10
 
1. For a given user, let's generate the top 10 most relevant movie recommendations, based on the user's preferences. We defined the necessary steps for you to do this. And you might also find it helpful to use the below `rank_top_k` function that we provided for you.
<!-- #endregion -->

```python id="7sFPWtOez8Dj"
def rank_top_k(names, ratings, k=10):
   """
   Example:
   a, b = np.array(['a', 'b', 'c']), np.array([6, 1, 3])
   a, b = rank_top_k(a, b, k=2)
   >>> a
   np.array('a', 'c')
   >>> b
   np.array([6, 3])
   """
 
   # rank indices in descending order
   ranked_ids = np.argsort(ratings)[::-1]
 
   return names[ranked_ids][:k], ratings[ranked_ids][:k]
```

```python id="PI2exdGgz8Dm" outputId="44774540-419b-4fea-f3f0-6a040d02b309"
user_id = 0
top_k = 10

# Answer: 10.1
# Step 1: Define the user's preferences using the training set.
user_train = np.array(train[0][user_id])


# Step 2: Define what movies the user has not seen yet.
# We will only generate recommendations for these movies, so make sure we store this in an awway equal to the number of all movies.
movies_not_seen = np.where(user_train == 0, 1, 0)

# Step 3: Predict the user's ratings across all movies.
estimates = np.dot(P[user_id, :], Q.T)


# Step 4: Consider the estimated ratings for movies that were not seen by the user.
unseen_movie_estimates = estimates * movies_not_seen

# Step 5: Retrieve the top k recommendations for that user.
recommendations, scores = rank_top_k(np.array(movies['Title']), unseen_movie_estimates, k=top_k)


# Step 6: Show the title and associated latent feature of the recommendations
df = pd.DataFrame(np.matrix((recommendations, scores)).T, (np.arange(10) + 1).tolist(), 
                  columns=['Title', 'Predicted rating'])
df

```

<!-- #region id="16H76jlxz8Dq" -->
It could also be interesting to recommend movies to a user considering not only her past movie ratings, but also her preference for various movie genres.
 
##### Question 11
 
1. Let's imagine that our user is browsing through a list of animations. To serve her the most relevant set of movies, we may want to limit our recommendations to a specific genre. Can you write a function that will provide the 5 most relevant movies that our user might be interested in?
<!-- #endregion -->

```python id="pS63d_flz8Dq"
def recommend(user_id, data, P, Q, list_of_genre_names, movies_genre, genre):
    """
    args:
       user_id: user_id
        data: user-item ratings
        P: user matrix
        Q: item matrix
       list_of_genre_names: list of genre names
        movies_genre: user's preference for genres
        new: Boolean, do we want to make new recommendations or not?

    Returns:
        the best suggestions based on the genre of movie selected
    """

    # Answer 11.1
    place = movies_genre_name.tolist().index(genre)   
    genre = np.array(movies_genre[:, place])
    predictions = np.array(np.dot(P[user_id, :], Q.T))
    
    return np.array(predictions) * np.array(genre.T)[0]
```

```python id="2bdsvSK9hHnx"
# print(movies_genre_name)
# print(movies_genre.shape)
```

```python id="qVkOjfDhz8Du" outputId="a7a5812f-9f70-426f-b07f-1f5e277c04c6"
genre = "Animation"
user_id = 1
top_k = 5
 
# Estimate recommendations
estimates = recommend(user_id, train, P, Q, list_of_genre_names=movies_genre_name, movies_genre=movies_genre, genre=genre)
 
recommendations, scores = rank_top_k(np.array(movies['Title']), estimates, k=top_k)
 
# Presentation
df = pd.DataFrame(np.matrix((recommendations, scores)).T, (np.arange(top_k) + 1).tolist(), columns = ['Title', 'Predicted rating'])
df

```
