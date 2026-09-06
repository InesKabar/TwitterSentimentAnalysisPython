# Import the libraries

import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
import math
import string

plt.style.use('fivethirtyeight')
sns.set()


# Twitter API authentication
# Add your own API credentials here

consumerKey = "YOUR_CONSUMER_KEY"
consumerSecret = "YOUR_CONSUMER_SECRET"
accessToken = "YOUR_ACCESS_TOKEN"
accessTokenSecret = "YOUR_ACCESS_TOKEN_SECRET"


# Create the authentication object

authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
authenticate.set_access_token(accessToken, accessTokenSecret)


# Create the API object

api = tweepy.API(
    authenticate,
    wait_on_rate_limit=True
)


# Extract 100 tweets from the Twitter user

posts = api.user_timeline(
    screen_name="NASA",
    count=100,
    lang="en",
    tweet_mode="extended"
)


# Print the last 5 recent tweets

print("Show the 5 recent tweets:\n")

i = 1

for tweet in posts[0:5]:
    print(str(i) + ") " + tweet.full_text + "\n")
    i += 1


# Create a dataframe with a column called Tweets

df = pd.DataFrame(
    [tweet.full_text for tweet in posts],
    columns=['Tweets']
)


# Create a function to clean the tweets

def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text


# Cleaning the text

df['Tweets'] = df['Tweets'].apply(cleanTxt)


# Create a function to get the subjectivity

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Create a function to get the polarity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity


# Create two new columns

df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)


# Plot the Word Cloud

allWords = ' '.join([twts for twts in df['Tweets']])

wordCloud = WordCloud(
    width=500,
    height=300,
    random_state=21,
    max_font_size=119
).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# Create a function to compute the sentiment analysis

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


# Create the Analysis column

df['Analysis'] = df['Polarity'].apply(getAnalysis)


# Show the dataframe

print(df)


# Print the positive tweets

j = 1

sortedDF = df.sort_values(by=['Polarity'])

for i in range(0, sortedDF.shape[0]):
    if sortedDF.iloc[i]['Analysis'] == 'Positive':
        print(str(j) + ') ' + sortedDF.iloc[i]['Tweets'])
        print()
        j += 1


# Print the negative tweets

j = 1

sortedDF = df.sort_values(
    by=['Polarity'],
    ascending=False
)

for i in range(0, sortedDF.shape[0]):
    if sortedDF.iloc[i]['Analysis'] == 'Negative':
        print(str(j) + ') ' + sortedDF.iloc[i]['Tweets'])
        print()
        j += 1


# Plot the polarity and the subjectivity

plt.figure(figsize=(8, 6))

for i in range(0, df.shape[0]):
    plt.scatter(
        df.iloc[i]['Polarity'],
        df.iloc[i]['Subjectivity']
    )

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


# Get the percentage of positive tweets

ptweets = df[df.Analysis == 'Positive']['Tweets']

print(
    "Positive tweets:",
    round((ptweets.shape[0] / df.shape[0]) * 100, 1),
    "%"
)


# Get the percentage of negative tweets

ntweets = df[df.Analysis == 'Negative']['Tweets']

print(
    "Negative tweets:",
    round((ntweets.shape[0] / df.shape[0]) * 100, 1),
    "%"
)


# Show the value counts

print(df['Analysis'].value_counts())


# Plot and visualize the counts

plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')

df['Analysis'].value_counts().plot(kind='bar')
plt.show()


# Plot the sentiment distribution

plt.title('Sentiment Analysis')

df['Analysis'].value_counts().plot(
    kind='pie',
    autopct="%1.1f%%"
)

plt.ylabel('')
plt.show()


# Plot positive, negative and neutral sentiments

for i in range(0, df.shape[0]):

    if df.iloc[i]['Analysis'] == 'Positive':
        plt.scatter(
            df.iloc[i]['Polarity'],
            df.iloc[i]['Subjectivity'],
            label="Positive"
        )

    elif df.iloc[i]['Analysis'] == 'Negative':
        plt.scatter(
            df.iloc[i]['Polarity'],
            df.iloc[i]['Subjectivity'],
            label="Negative"
        )

    else:
        plt.scatter(
            df.iloc[i]['Polarity'],
            df.iloc[i]['Subjectivity'],
            label="Neutral"
        )

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


# K-Means Clustering


# Cleaning tweets for K-Means

def pre_process_tweets():

    tweets = list(df['Tweets'])
    list_of_tweets = []

    for i in range(len(tweets)):

        # Remove new lines

        tweets[i] = tweets[i].strip('\n')

        # Remove words starting with @

        tweets[i] = " ".join(
            filter(
                lambda x: not x.startswith('@'),
                tweets[i].split()
            )
        )

        # Remove URLs

        tweets[i] = re.sub(r"http\S+", "", tweets[i])
        tweets[i] = re.sub(r"www\S+", "", tweets[i])

        # Remove hash-tag symbols

        tweets[i] = tweets[i].replace('#', '')

        # Convert every word to lowercase

        tweets[i] = tweets[i].lower()

        # Remove punctuation

        tweets[i] = tweets[i].translate(
            str.maketrans('', '', string.punctuation)
        )

        # Remove extra spaces

        tweets[i] = " ".join(tweets[i].split())

        # Convert each tweet into a list of words

        if tweets[i]:
            list_of_tweets.append(tweets[i].split())

    return list_of_tweets


# K-Means function

def k_means(tweets, k=3, max_iterations=50):

    if len(tweets) < k:
        raise ValueError("The number of tweets must be greater than or equal to k.")

    centroids = []

    # Initialization: assign random tweets as centroids

    count = 0
    hash_map = dict()

    while count < k:

        random_tweet_idx = rd.randint(0, len(tweets) - 1)

        if random_tweet_idx not in hash_map:
            count += 1
            hash_map[random_tweet_idx] = True
            centroids.append(tweets[random_tweet_idx])

    iter_count = 0
    prev_centroids = []

    # Run iterations until convergence or maximum iterations

    while not is_converged(prev_centroids, centroids) and iter_count < max_iterations:

        print("Running iteration " + str(iter_count))

        # Assignment

        clusters = assign_cluster(tweets, centroids)

        # Keep track of previous centroids

        prev_centroids = centroids

        # Update centroids

        centroids = update_centroids(clusters)

        iter_count += 1

    if iter_count == max_iterations:
        print("Max iterations reached, K-Means not converged")
    else:
        print("Converged")

    sse = compute_SSE(clusters)

    return clusters, sse


# Check if K-Means has converged

def is_converged(prev_centroid, new_centroids):

    if len(prev_centroid) != len(new_centroids):
        return False

    for c in range(len(new_centroids)):

        if " ".join(new_centroids[c]) != " ".join(prev_centroid[c]):
            return False

    return True


# Assign tweets to clusters

def assign_cluster(tweets, centroids):

    clusters = {}

    for t in range(len(tweets)):

        min_dis = math.inf
        cluster_idx = -1

        for c in range(len(centroids)):

            dis = getDistance(
                centroids[c],
                tweets[t]
            )

            if centroids[c] == tweets[t]:
                cluster_idx = c
                min_dis = 0
                break

            if dis < min_dis:
                cluster_idx = c
                min_dis = dis

        # Randomize the centroid assignment when there is no common word

        if min_dis == 1:
            cluster_idx = rd.randint(
                0,
                len(centroids) - 1
            )

        # Assign the tweet to the closest centroid

        clusters.setdefault(cluster_idx, []).append(
            [tweets[t], min_dis]
        )

    return clusters


# Update centroids

def update_centroids(clusters):

    centroids = []

    for c in range(len(clusters)):

        min_dis_sum = math.inf
        centroid_idx = -1

        min_dis_dp = []

        for t1 in range(len(clusters[c])):

            min_dis_dp.append([])
            dis_sum = 0

            for t2 in range(len(clusters[c])):

                if t1 != t2:

                    if t2 < t1:
                        dis = min_dis_dp[t2][t1]
                    else:
                        dis = getDistance(
                            clusters[c][t1][0],
                            clusters[c][t2][0]
                        )

                    min_dis_dp[t1].append(dis)
                    dis_sum += dis

                else:
                    min_dis_dp[t1].append(0)

            if dis_sum < min_dis_sum:
                min_dis_sum = dis_sum
                centroid_idx = t1

        centroids.append(
            clusters[c][centroid_idx][0]
        )

    return centroids


# Jaccard distance between tweets

def getDistance(tweet1, tweet2):

    intersection = set(tweet1).intersection(tweet2)
    union = set().union(tweet1, tweet2)

    if len(union) == 0:
        return 0

    return 1 - (len(intersection) / len(union))


# Compute the SSE

def compute_SSE(clusters):

    sse = 0

    for c in range(len(clusters)):

        for t in range(len(clusters[c])):

            sse += clusters[c][t][1] * clusters[c][t][1]

    return sse


# Run K-Means

if __name__ == '__main__':

    tweets = pre_process_tweets()

    experiments = 4
    k = 3

    for e in range(experiments):

        print(
            "------ Running K-Means for experiment no. "
            + str(e + 1)
            + " for k = "
            + str(k)
        )

        clusters, sse = k_means(
            tweets,
            k
        )

        for c in range(len(clusters)):

            print(
                str(c + 1)
                + ": "
                + str(len(clusters[c]))
                + " tweets"
            )

        print("--> SSE : " + str(sse))
        print('\n')