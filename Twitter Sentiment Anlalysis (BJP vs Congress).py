

Introduction
Sentiment analysis has been predominantly used in data science for analysis of customer feedbacks on products and reviews. They are used to understand user ratings on different kinds of products, hospitality services like travel, hotel bookings.

It has also become popular to analyse user tweets?—?positive, negative or neutral by crawling twitter through APIs.

In this article, we talk about sentiment analysis of the upcoming Lokshobha Elections for Congress and BJP by crawling tweets from different hashtags of either parties, party leaders, as well as news hashtags like NDTV. The sentiments analysed covers different user-reactions not only restricted to positive or negative sentiments but covers an in-depth analysis of various positive and negative moods along with the results of different ML models.

We categorise the analytics and machine learning into 3 sections:

Crawling, cleaning data and labelling un-structured data by using/mapping known English words from various sources
Applying Natural Language based classifiers used for text processing to train tweets and predict moods
Applying standard machine learning algorithms and deep learning to do multi-class mood classification for 2 prominent parties in the election
The objective of this blog is to highlight mechanisms for labelling tweets, and classifying and summarizing them from different viewpoints.


Crawl Weekly tweets and Merge:
We crawl tweets on a weekly basis and merge them with previous weeks to have an overall prediction over a period of few months. The system is designed to learn from tweets every week and consolidates results by eliminating duplicate tweets. It preserves the retweet counts to understand the impact of higher number of retweets.

list_BJP = []
list_Cong = []
if('BJP' in file_[i]):
    df_BJP = pd.read_csv(file_[i],index_col=None, header=0)
if ('Cong' in file_[i]):
    df_Cong = pd.read_csv(file_[i], index_col=None, header=0)
list_BJP.append(df_BJP)
list_Cong.append(df_Cong)
df_BJP = pd.concat(list_BJP, axis = 0, ignore_index = True)
df_BJP = df_BJP.drop_duplicates(subset=['created_at', 'full_text’])#dropping retweets with same text posted at same time
df_BJP = df_BJP[df_BJP.full_text != 'full_text']

df_Cong = pd.concat(list_Cong, axis = 0, ignore_index = True)
df_Cong = df_Cong.drop_duplicates(subset=['created_at', 'full_text'])
df_Cong = df_Cong[df_Cong.full_text != 'full_text']
Crawl Mood Words and labelling unstructured data:
The mood vocabulary is built using english word repository available in the internet. The following mood labels Joy, Sadness, Arousal, Dominance, Neutral, Anger, Fear, Faith(Support) were assigned to tweets by taking the strongest mood in the sentence, by taking each word from the sentence into account, along with the emoji in consideration. For example, the overall mood of the sentence is Dominance when each word in the sentence have the following moods.

[‘dominance’, ‘dominance’, ‘dominance’, ‘dominance’, ‘dominance’, ‘joy’, ‘arousal’, ‘dominance’]

max_mood_item = max(mood_freq_dist.items(), key=operator.itemgetter(1))[0]
The sentiment of each word is derived by assigning an affectual score to it . The lexicon dictionary for 25,000 words are dowloaded from NRC Word -Emotion Association Lexicon (Reference 2). If certain words in a sentence are missing from Vader or the specific mood type is missing, TextBlob is used to determine positive/negative sentiment of the word. For example, for the following tweet “I request all fellow Indians to get rid of this clown coming elections. Please vote wisely”, the word “wisely” encounters a Valence score of 0.878, but it does not differentiate between positive (Joy)/negative (Sadness/Anger) mood, which necessitates a further lookup of word polarity through TextBlob. Finally with a positive score of 0.7 its labelled as sentiment of “Joy”.

While affectual score and TextBlob determines mood of each word, SentimentIntensityAnalyzer is used to calculate the overall polarity of a sentence. It uses Vader’s lexicon (Reference 2) which rates individual words (present in the lexicon) in a sentence on a scale of highly negative to highly positive.

For example, for a tweet, “We stand rock solid behind you @narendramodi Our party has performed well under all odds, we will do better in” has few words in the lexicon with score as “solid”: 0.6, “party” : 1.7, “well” :1.1 and “better” 1.9

These word ratings help to derive four sentiment metrics to represent the proportion the tweet falls under it.

‘compound’: 0.8074, ‘neg’: 0.0, ‘neu’: 0.632, ‘pos’: 0.368

This explains the tweet is how much positive, negative or neutral. The compound score have been standardised to range between -1 and 1 and is calculated by calculating the normalized sum (normalize(sum_s)) of all of individual word ratings (0.6, 1.7, 1.1, 1.9) present in the lexicon.

sia = SentimentIntensityAnalyzer()
ps = sia.polarity_scores(tweet)
overall_score = ps['compound']
area = np.pi * 3
mplt.scatter(df['compound'],df['mood'], s=area, alpha=0.5)
mplt.title('Tweet Intensity for ' + labels[i])
mplt.xlabel('Sentiment Intensity')
mplt.ylabel('Moods')
mplt.show()
All tweets vary in intensity from -1 to +1. As the below figures shows strong positive sentiments like “Joy” and “Faith” incline more 0 to +1 for both BJP and Congress, while negative sentiments like “Anger” and “Sadness” incline more between -1 to 0. “Neutral” sentiment is centred around zero. Sentiments like “Arousal” and “Dominance” are more or less distributed equally between -1 to +1 which signify they can be either tweeted in a positive or negative mind.

For example, the tweet “In 2014 when Modi elected PM candidate, people eected change will happen” records “Dominance” with positive sentiment . While the tweet “2019 elections will be fought on completely different lines’, says @amitmalviya, National Spokesperson, BJP in conversation” records a negative sentiment because of the word “fought”. SentimentIntensityAnalyzer calculates the compound metric of the tweet as -0.3182, while positive, negative and neutral scores are 0.0, 0.247 and 0.753 respectively. Further rating the word “fought” in terms of “Valence”?—?(Joy/Sadness/Anger/Fear/Faith), “Arousal” or “Dominance”, the measuremnets are “Valence” : 0.531, “Arousal” : 0.809, “Dominance” : 0.868, justifying the predominance of “Dominance” mood.

Similarly, both positive and negative sentiments can be observed with “Arousal” mood. “Arousal” incorporates any feeling that causes state change or prompts to rise and undertake any activity. The tweet “Govt today introduced a bill in to make provisions regarding recognition of, drawing opposition from the as well as the CPI(M) which staged a walkout calling it a ‘draconian and unconstitutional’ legislation” records a compound score of -0.3182 showing a negative “Arousal” sentiment. While the tweet “God bless you all. Now do the job well, Dems, it’s been way too long since it was done properly. Show them how it’s done” is a positive “Arousal” sentiment with a compound score of 0.95 .


The upper and bottom figures demonstrate how sentiments differ for Congress and BJP. The most visually distinguishing aspects are seen in the 2 sentiments “Faith” and “Arousal”. BJP records a higher recording in “Faith” while Congress shows higher predominance of “Arousal”.


The following figure illustrates tweet that belongs to both BJP and Congress. “Dominance” is still seen as the predominant mood. A sample tweet involving both parties : “Very close fight in . The difference between BJP and Congress not too many seats”?—?— clearly shows close and stiff competition between the two.


Tweet Analytics
This section, we structurize the blog into different areas of analytics and provide visual representations for comparisions.

Frequency of different Moods
Sentiment representation by Word Cloud
N-gram model
Location-wise tweet distribution
Retweet frequency distribution
Frequency of different Moods
sns.set(font_scale=0.8)
df_BJP =  pd.read_csv(plot_path + files[i+1])
df_Cong = pd.read_csv(plot_path + files[i+2])
fields = ['tweet', 'mood']
# Create a figure instance, and the two subplots
fig = mplt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

size = fig.get_size_inches() * fig.dpi  # get fig size in pixels
ax1.set_title("LokShobha Elections 2019 " + labels[i] + " Sentiments", fontsize = 8.5, loc ='right')
ax2.set_title("LokShobha Elections 2019 " + labels[i+1] + " Sentiments", fontsize = 8.5, loc ='right')

# Tell countplot to plot on ax1 with one party and ax2 with another party
g = sns.countplot(x="mood", data=df_BJP,  palette="PuBuGn_d",  ax=ax1, order = df_BJP['mood'].value_counts().index)
g = sns.countplot(x="mood", data=df_Cong,  palette="PuBuGn_d",  ax=ax2, order = df_BJP['mood'].value_counts().index)
mplt.show()
The different mood frequencies show public reactions towards both the parties before elections. “Dominance” mood dominates in case of both the parties followed by “Joy” mood. SNS countplot provides functionality to plot total frequency distribution of each individual mood which helps to compare within party different moods as well compare a specific mood for both the parties. For instance, for the following graphs of BJP and Congress shows the total number of tweets received for BJP is more than Congress and consequently each corresponding mood gets a higher percentage of tweets for BJP than Congress.


Sentiment Comparisions
Sentiment Representation by WordCloud
The different kinds of tweet sentiments are represented by means of different WordClouds. WordClouds are ideal representatives of labelled sentiments as the most common words specific to a mood appear bigger and bolded than other less frequent words. WordClouds are fast and easy mechanism of representing the most relevant words for a theme or context. Its one of the most convenient ways to convey information visually appealing and engaging manner.

Here 2 different sentiments of BJP “Faith/Support” and “Fear” are represented by 2 different WordClouds.

The below code snippet represents all tweets specific to “Faith” sentiment through a WordCloud.

df_faith = df[df['mood'] == 'faith']
wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(str(df_faith.tweet.values))
mplt.figure(figsize=(12, 10))
mplt.imshow(wordcloud, interpolation="bilinear")
mplt.title(labels[i] + "  Faith", fontsize = 10)
mplt.xlabel('Support/Faith')
mplt.axis("off")
mplt.show()
From the figure below, you can see certain words of BJP like “modi”, “pm” are more frequent and the tweets exibit a tendency to “support”, “congratulate” , “thank” Prime Minister Narendra Modi for country’s development. Words like “vikas”, “development”, “honest team” , “agree”, “sath” , point out positive sentiment towards Modi government. Futher tweets that honour Prime Minister, are visible though words like “hon pm”, “dearest”, “fan”.


One kind of negative sentiment like “Fear” for the BJP government is analysed and represented through a separate WordCloud. The “Fear” WordCloud shows a kind of negative feeling, fear/threat in people’s mind from opposition parties.

df_fear = df[df['mood'] == 'fear']
wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(str(df_fear.tweet.values))
mplt.figure(figsize=(12, 10))
mplt.imshow(wordcloud, interpolation="bilinear")
mplt.title(labels[i] + "  Fear", fontsize = 10)
mplt.xlabel('Fear')
mplt.axis("off")
mplt.show()
The “Fear” WordCloud has prominent bolded words like “worry”, “failure”, “mistrust”, “fighting” “worried” , “unexpected” , “wounded” that raises questions about doubts and uncertainities in people’s minds.


Similarly, doing the sentiment analysis for Congress, 2 different moods one Positive?—?Joy and another negative -Sadness are represented by means of WordCloud. The “Sadness” WordCloud of Congress have clearly distinguishable words like “lost”, “refused”, “defeat”, “destroy”, “crying”, “missed”, “loot”, “slaps” that remark a sense of negative disheartened feeling in the tweets. Further the occurrence of most frequent words “Gandhi” , “Rahul” shows Rahul Gandhi as one of the foremost leaders of Congress.

df_sadness = df[df['mood'] == 'sadness']
wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(str(df_sadness.tweet.values))
mplt.figure(figsize=(12, 10))
mplt.imshow(wordcloud, interpolation="bilinear")
mplt.xlabel('Sadness')
mplt.title(labels[i] + "  Sadness", fontsize = 10)
mplt.axis("off")
mplt.show()

The positive tweet sentiments for Congress are represented by means of “Joy” WordCloud. Similar to the previous WordCloud “Rahul Gandhi”, “Congress” dominates the word cloud.

df_joy = df[df['mood'] == 'joy']
wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(str(df_joy.tweet.values))
mplt.figure(figsize=(12, 10))
mplt.imshow(wordcloud, interpolation="bilinear")
mplt.xlabel('Joy')
mplt.title(labels[i] + "  Joy", fontsize = 10)
mplt.axis("off")
mplt.show()
Words like “win”, “good”, “congratulation”, “great”, “truth”, “happy”, “love”, “victory”, “dancing” , “grand” , “cheer” , “laugh” exhibits a strong “Happy” and “Joyous” public sentiment for Congress.


N-gram Model
The most popular bag-of words in NLP has n-gram models comprising of 1 -word text (Unigram) , 2-word text (Bi-gram) , 3-gram text (Tri-gram), where the number of occurrences of single word, side-by-side 2 words, side-by-side 3 words are counted and fed as feature-vectors to Text Classifiers (Naive Bayes, Maxium Entropy and Support Vector Machines). Word occurrences are counted after cleaning the tweets from hashtags, urls, emojis stopwords and character repeatations. This helps to extract most popular 1-word, 2-words, 3-words from tweet and construct feature vectors to determine the overall sentiment score of the text.

#splits up a sentence to 1-word, 2-word,3-words depending on input n 
def get_ngrams(tweet_words, n):
    ngrams = []
    num_words = len(tweet_words)
    for i in range(num_words -(n-1)):
        lookUpTweets = []

        for j in range(i, i+n):
            lookUpTweets.append(tweet_words[j])

        ngrams.append(tuple(lookUpTweets))

    return ngrams
#calculates the frequency distribution of 1-word, 2-word,3-words 
def get_ngram_freqdist(ngrams):
    freq_dict = {}
    for ngram in ngrams:
        if(ngram in freq_dict):
            freq_dict[ngram] += 1
        else:
            freq_dict[ngram] = 1
    counter = Counter(freq_dict)
    return counter
#Unigram Frequency Distribution
word_counter_df = pd.read_csv(word_disb_path + uni_gram_files[i])
word_popular_df = word_counter_df.nlargest(25, columns=['F'])
word_popular_df['unigram_word'] = word_popular_df.W1
fig = sns.barplot(x=word_popular_df["unigram_word"], y=word_popular_df["F"])
sns.set(font_scale=.3)
mplt.xlabel("Unigram Words", fontsize=10)
mplt.ylabel("Frequency", fontsize=10)
mplt.title("LokShobha Elections 2019 " +  labels[i], fontsize=10) 
mplt.show(fig)
#Bigram Frequency Distribution
sns.set(font_scale=0.5)
word_popular_df['bigram_word'] = word_popular_df.W1 + "  " + word_popular_df.W2
fig = sns.barplot(x=word_popular_df["bigram_word"], y=word_popular_df["F"])
sns.set(font_scale=.5)
mplt.xlabel("Bigram Words", fontsize = 10)
mplt.ylabel("Frequency", fontsize = 10)
mplt.title("LokShobha Elections 2019 " + labels[i], fontsize = 10)  
mplt.show(fig)
Unigram Frequency Distribution for Congress and BJP shows the most dominant 1-word occurring in the respective tweets.



Similarly, Bigram Frequency Distribution for Congress and BJP shows the most dominant 2-word occurring in the respective tweets.



Location wise tweet distribution
A pie-chart is constructed for each of Congress and BJP by taking into account percentages of tweets from some of the known states of India. While both of them have larger percentages of tweets from unknown location and unknown states of India, New Delhi, Mumbai and Bangalore still dominates the percentages of tweets from India.

location_df = combined_df['location'].value_counts()
filter_loc = location_df[location_df>35]
mplt.rcParams['font.size'] = 5.0
mplt.title(labels[i])

patches, texts, autotexts = mplt.pie(
    filter_loc,
    labels=filter_loc.index.values,
    shadow=False,
    startangle=90,
    pctdistance=0.7, labeldistance=1.15,
    # with the percent listed as a fraction
    autopct='%1.1f%%',
)
mplt.axis('equal')
mplt.tight_layout()
mplt.show()


Retweet Frequency Distribution
df_raw = pd.read_csv(full_statspath + stats_files[i]).dropna()
df_raw.drop_duplicates(subset="full_text",
                     keep='first', inplace=True)

df_raw_retweets = df_raw.nlargest(25, columns=['retweet_count'])

x = df_raw_retweets["full_text"].values
y = df_raw_retweets["retweet_count"].values

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
fig, ax = mplt.subplots()

offset = 0.75
for k in range(len(x)):
   ax.text(offset, k, x[k], color='blue', fontweight='bold', fontsize = 7)
   offset = offset+1

width = 0.75  # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color = colors)
mplt.title(labels[i])
mplt.xlabel('Retweet Frequency', fontsize = 7)
mplt.ylabel('Tweets', fontsize = 7)
mplt.show()
The popularity of tweets have been represented with the retweet count . Only first 25 unique retweets are selected. Its seen, that BJP tweets are much more frequent than Congress and ranges between 100–250 while average retweet frequency for Congress is 20–30. The retweet frequency along with the tweet text have been graphically displayed below.



Conclusion
This post mainly discusses about labelling tweets from known word dictionaries and rating them between -1 and 1 . It further compares BJP and Congress side by side considering tweet sentiments, frequency of different tweet sentiments, commonly used words in tweets (anargrams 1–2 words), location of users who tweeted as well as the most popular tweets obtained from the retweet count. The following posts will cover on different ML techniques used for NLP, comparing them side by side with different metrics of accuracy like Precision , Recall and F1 Score as well as processing time to train the models. The election results for 2019 is still few months to go and the study hopes to find more interesting results through more weekly tweet crawls.