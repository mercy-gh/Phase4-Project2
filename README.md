
# 1 Business Understanding

## 1.0 Overview
With the rapid expansion of internet streaming platforms, users are overwhelmed by the sheer volume of available movies. Providing personalized recommendations is critical for increasing user engagement, satisfaction, and retention. The MovieLens (ml-latest-small) dataset, rom [MovieLens](http://movielens.org) (a movie recommendation service),  includes user-generated 5-star ratings and free-text tags that can be used to create a powerful recommendation engine.


## 1.1 Problem Statement
With thousands of movies accessible on streaming platforms, customers struggle to discover ones they'll like. This choice overload frequently results in dissatisfaction, decision fatigue, and low user engagement.
Many systems rely on generic rankings or trending lists that do not consider individual preferences. This leads to irrelevant movie suggestions that do not match user preferences, longer search times lower user happiness, or low retention rates due to users potentially switching to competitors with better recommendations.

The goal is to create a movie recommendation system that can predict user preferences based on previous interactions. Specifically, we want to:

## 1.2 Objectives
&#x2611; Analyze user ratings and tags to find patterns and trends.

&#x2611; Create a recommendation model that incorporates collaborative filtering, content-based filtering, or hybrid approaches.

&#x2611; Address critical issues such as data scarcity, cold start issues, and bias in user ratings.

&#x2611; Assess model performance using relevant measures such as RMSE or MAE.



## 1.3 Proposed solution
The objective is to examine and use the dataset to boost user engagement by developing a movie recommendation engine. Potential applications include:

&#x2611; Personalized **Movie Recommendations** - Predict user preferences based on previous ratings.

&#x2611; Customize content by **segmenting and clustering users** based on similar preferences.

&#x2611; **Trend Analysis and Insights**: Discover popular genres, top-rated movies, and viewing behaviors.

&#9745; Tag-based **sentiment analysis** provides insights into user perception of movies.


&#x2713; **Next Steps**

&#x1f8ae; Exploratory Data Analysis (EDA) – Understand rating distributions, popular tags, and trends.

&#x1f8ae; Feature Engineering – Transform text tags into meaningful numerical features.

&#x1f8ae; Model Selection – Choose between collaborative filtering or content-based filtering.

&#x1f8ae; Evaluation Metrics – Use RMSE or MAE to assess recommendation performance.


### Challenges & Considerations
* Data Sparsity: Not all users have rated all movies, leading to gaps in the dataset.
* Cold Start Problem: New users/movies lack enough data for accurate recommendations.
* Bias in Ratings: Some users may consistently rate higher or lower than others.
* Scalability: The model should be efficient enough to handle large datasets in real-world applications.


# 2 Data Understanding
The MovieLens dataset (ml-latest-small) comprises user-generated movie ratings as well as free-text tags. This dataset is commonly used in recommendation systems, where businesses try to improve the customer experience by proposing movies based on user interests.Companies like Netflix, Hulu, and Amazon Prime Video use similar databases to boost user engagement, retention, and satisfaction through personalized suggestions.

The dataset consists of 5-star ratings and free-text tags from MovieLens, an online movie recommendation service.

Users rate movies on a 1-5 star scale (higher ratings indicate better user satisfaction).
Tags are free-text descriptions provided by users to describe movies (e.g., "thriller," "comedy," "Oscar-winning").
The dataset is anonymized (users are represented by IDs).

|File Name | Description|
|----------|------------|
|ratings.csv |	Contains user ratings for movies (1-5 scale).|
|movies.csv |	Metadata including movie titles and genres.|
|tags.csv |	Free-text tags assigned by users to movies.|
|links.csv |	Provides mappings to external movie databases (IMDB, TMDb).|


**Identifiers**
- *userId*: random and anonymous IDs given to identify users. MovieLens users were selected at random for inclusion. Their ids have been anonymized. User ids are consistent between `ratings.csv` and `tags.csv` (i.e., the same id refers to the same user across the two files)

- *movieId*: IDs given to identify movies. Since only those movies which have 1 or more rating or tag are selected, it is not a complete sequence. Movie ids are consistent between `ratings.csv`, `tags.csv`, `movies.csv`, and `links.csv` (i.e., the same id refers to the same movie across these four data files)

**Movies Data columns:**
- *movieId*:Unique movie identifier.
- *title:* Movie title, and include the year of release in parentheses.
- *genre:* a pipe-separated list, and are selected from the following: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, 
Sci-Fi, Thriller, War, Western and "no genres listed"

**Ratings Data columns:**
- *userId* – Unique identifier for each user.
- *movieId* – Unique identifier for each movie.
- *rating:* User rating (1-5 stars), with half-star increments (0.5 stars - 5.0 stars).
- *timestamp:* Time when the rating was given.

**Links data features:**
- *movieId:* Unique identifier for each movie.
- *imdbId:* an identifier for movies used by <http://www.imdb.com>
- *tmdbId:* an identifier for movies used by <https://www.themoviedb.org>

**Tags data features:**
- *userId* – Unique user identifier.
- *movieId* – Movie being tagged.
- *tag:* Free-text tag (e.g., "thrilling," "mind-blowing," "classic").
- *timestamp:* Time when the tag was added

# Data understanding

```md
<pre>
<code>
# Example Python Cell
import pandas as pd

print("Hello, MovieLens!")
</code>
</pre>

**imports** 
**# import data manipulation libraries**
import pandas as pd
import numpy as np
import scipy.sparse as sp

**# import data visualisation libraries**
import matplotlib.pyplot as plt
import seaborn as sns

**# import modeling libraries**
from sklearn.model_selection import train_test_split
from surprise.model_selection import GridSearchCV#, cross_validate

**# Surprise Libraries**
from surprise.prediction_algorithms import KNNBasic
from surprise import accuracy

**# import Preprocessing and Metric Evaluation libraries**
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error#, root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import warnings; warnings.simplefilter('ignore')

**# Get summary statistics of movies data**
movies_data.describe()
Understand ratings data

**# view top 5 rows**
ratings_data.head()

# Get Summary info of the ratings dataframe
ratings_data.info()
*Observation*: All features have the right data type and it has no null values
# Get summary statistics of ratings data
ratings_data.describe()
understand links data:
# view top 5 rows
links_data.head()
# Get Summary info of the links dataframe
links_data.info()
*Observation*: All features have the right data type and it has $8$ null values
# Get summary statistics of links data
links_data.describe()
Understand tags data:
# view top 5 rows
tags_data.head()
*Comment*:
The timestamp features may be converted to datetime where need be such as helping to observe trends
# Get Summary info of the tags dataframe
tags_data.info()
*Observation*: All features have the right data type and it has no null values
# Get summary statistics of tags data
tags_data.describe()
# 3 Data Preparation
This involves activities such us handling missing values, checking for duplicates, dropping unnecessary columns, etc.
## 3.1 Data Cleaning

### 3.1.1 Handling Missing Values
# Check movies data for null values
movies_data.isnull().sum()
# Check ratings data for null values
ratings_data.isnull().sum()
# Check tags data for null values
tags_data.isnull().sum()
# Check links data for null values
links_data.isna().sum()

Links data has missing values. Let us see the rows with the missing values since they are few.
# List the rows with nulls
links_data[links_data['tmdbId'].isnull()]
The links data had $8$ missing values. Let's check

Since we dont have the exact tmdbId of the movies with null values, we can then replace the nulls with a '0'.
# Replace nulls with a '0'
links_data.fillna(0, inplace=True)
# Check nulls
links_data.isna().sum()

### 3.1.2 Duplicate Values

Let us che the datasets for duplicate values.
# Check duplicates
movies_data.duplicated().sum()
# Check duplicates
ratings_data.duplicated().sum()
# Check duplicates
tags_data.duplicated().sum()
# Check duplicates
links_data.duplicated().sum()
*Comment*:

No duplicates in all the datasets
### 3.1.3 Column Editing
We can drop columns that will not be of use as of now. This will include the timestamp, imdbId & tmdbId columns

We can now merge the data to have one dataset to explore.
### 3.1.4 Merging the datasets and handling missing values & duplicates
# Merge ratings with movie titles
merged_df = ratings_data.merge(movies_data, on="movieId", how="left")
merged_df.info()
# Merge with the tags data
merged_df = merged_df.merge(tags_data[['userId', 'movieId', 'tag']], on=["userId", "movieId"], how="left")
merged_df.info()
*Observation:*
The tag column has null values
# Check null values
merged_df.isnull().sum()
The merged data has 99,201 values missing on the tags column. The dtype is object (string).

We can then handle the null values by replacing 'null' by 'Unknown'
# Replace nulls
merged_df.fillna("Unknown", inplace=True)
# Recheck nulls
merged_df.isna().sum()
# Check for duplicates
merged_df.duplicated().sum()


# 4. Exploring the Dataset
## 4.1 Univariate

### 4.1.1 Movie Ratings distribution
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/DistributionofMovieRatings.png)

### 4.1.2 Top 10 most rated
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/Top10MostRatedMovies.png)

### 4.1.3 Most common genres
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/MostCommonMovieGenres.png)

### 4.1.4 User activity
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/NumberofRatingsperUser.png)

## 4.2 Bivariate
### 4.2.1 Average Rating per Movie
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/Top10HighestRatedMovies(withaminof50ratings).png)

### 4.2.2 Rating vs. Number of Ratings (Popularity Bias)
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/RelationshipBetweenNumberofRatingsandAverageRating.png)

### 4.2.3 Relationship Between Movie Ratings & Number of Ratings
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/NumberofRatingsperUser.png)

### 4.2.4 Genre vs. Average Rating
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/AverageRatingbyGenre.png)

### 4.2.5 Relationship Between User Activity & Average Rating Given
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/NumberofRatingsGivenvsAverageRatingGiven.png)

### 4.2.6 User Activity vs. Average Rating
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/UserActivityvsAverageRatingGiven.png)

## 4.3 Multivariate
### 4.3.1 Feature Correlations Heatmap
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/FeatureCorrelation.png)

### 4.3.2 Rating Trends Over Time
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/AverageRatingOvertime.png)

### 4.3.3 Genre Distribution Across Different Rating Levels
![Alt text](https://github.com/mercy-gh/Phase4-Project2/blob/main/Images/GenreDistributionAcrossDifferentRatingLevels.png)

# 5. Conclusion
From the model created:
1. A K_Nearest Neighbours (KNN) model performs much better than and SVD model for such a recommendation system even without fine tuning. (Note: Requires resources to fine tune). Meaning it will be much improved.

2. A collaborative-filtering method of recommendation gives different results compared to content-Based filtering.

3. A user-specific model should use content-based as they are coser to user's preference. Though it does not automatically mean that different users will have same preferences despite similarity.

# 6. Recommendations
The recommendations are as follows:
1. Modeling with KNN is much better or a good model for comparison.

2. Other models should be tried to compare performances.

3. When using KNN, consider the resources to use especially when fine-tuning the model.