#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[1]:


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)


# # Scraping

# I used Selenium and beautifulsoup to scrape NBA.com. It's important to note that I used the data from the traditional, advanced and scoring pages. Scoring and traditional use this chunk while advanced uses the next chunk. For some reason, the HTML on the advanced page includes things that don't work with this chunk and I had to modify it

# In[2]:


def scrape_nba_com(drive, table_name):

    
    soup = BeautifulSoup(drive.page_source, 'html.parser')
    table = soup.find("table", {"class" : table_name}) # Find the table that has table_name
    
    thead = table.find("thead") # Get the head

    # th stands for table-header
    table_headers = thead.find_all("th") # Get all table headers

    # extract actual header name from th elements
    cleaned_headers = [i.text for i in table_headers]

    # more clean up
    cleaned_headers = [i for i in cleaned_headers if "RANK" not in i]
    
    # Get rows
    table_rows = table.find("tbody").find_all("tr")
    
    td_in_rows = [r.find_all("td") for r in table_rows]

    # nested list comprehension to extract actual data from each row
    # code is basically identical to above cell
    table_data = [[td.text for td in i] for i in td_in_rows]
    
    # Change everything into a dataframe
    df = pd.DataFrame(data=table_data, 
             columns=cleaned_headers)
    
    return df


# In[3]:


def scrape_nba_com_ad(drive, table_name):
    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    #driver.get(url)
    
    soup = BeautifulSoup(drive.page_source, 'html.parser')
    table = soup.find("table", {"class" : table_name})
    
    thead = table.find("thead")

    # th stands for table-header
    table_headers = thead.find_all("th")

    # extract actual header name from th elements
    cleaned_headers = [i.text for i in table_headers]

    # more clean up
    cleaned_headers = [i for i in cleaned_headers if "RANK" not in i]
    
    cleaned_headers = cleaned_headers[:24] # This is the part that is different. Only take columns up to 24
    
    # Get rows
    table_rows = table.find("tbody").find_all("tr")
    
    td_in_rows = [r.find_all("td") for r in table_rows]

    # nested list comprehension to extract actual data from each row
    # code is basically identical to above cell
    # 
    table_data = [[td.text for td in i] for i in td_in_rows]
    
    df = pd.DataFrame(data=table_data, 
             columns=cleaned_headers)
    
    return df


# Some important things to note: I only took data from 2022-2023 and I only took traditional, advanced and scoring. In addition, when you run the chunk below, the nba.com page will show up. Click accept for the cookies, and change the drop down arrow that is next to page from 1 to All. This will make it so that all the players show up. Otherwise, you will only get the first 50. This must be done with each of the chunks below that take you to the website. Once you change it to all, minimize the page and run the df_2022 = cell and you should get the dataframe

# This one is for traditional

# In[4]:


url = "https://www.nba.com/stats/players/traditional?PerMode=PerGame&sort=PTS&dir=-1&SeasonType=Regular+Season&Season=2022-23"
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(url)


# In[5]:


df_2022 = scrape_nba_com(driver, "Crom_table__p1iZz") # Get the dataframe


# In[6]:


df_2022 # Check that it looks correct


# This one is for advanced

# In[7]:


url = "https://www.nba.com/stats/players/advanced?PerMode=PerGame&sort=PTS&dir=-1&SeasonType=Regular+Season&Season=2022-23"
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(url)


# In[8]:


ad_2022 = scrape_nba_com_ad(driver, "Crom_table__p1iZz")


# In[9]:


ad_2022


# This one is for scoring

# In[10]:


url = "https://www.nba.com/stats/players/scoring?PerMode=PerGame&sort=PTS&dir=-1&SeasonType=Regular+Season&Season=2022-23"
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(url)


# In[11]:


score_2022 = scrape_nba_com(driver, "Crom_table__p1iZz")


# In[12]:


score_2022


# In[13]:


# This will merge the dataframes together
data = pd.merge(pd.merge(df_2022, ad_2022, left_on="Player", right_on="PLAYER"), score_2022, on="Player")


# In[14]:


data # make sure it worked


# In[15]:


# Drop all duplicate columns and those that are not needed. Could also change the merge above to a different join
data = data.drop(['\xa0_x', '\xa0_y', 'GP_x', 'W_x', 'L_x', 'TEAM_x', 'PLAYER', 'AGE_x', 'GP_y', 'W_y', 'L_y', 'MIN_x', 'TEAM_y', 'AGE_y', 'MIN_y', ' '], axis=1)


# In[16]:


data.head() # Make sure everything looks nice


# In[ ]:


# Write our data to a csv so we don't have to run everything again

#data.to_csv('data.csv', index=False)


# # Data Exploration

# In[3]:


df = pd.read_csv('data.csv') # I wrote the data to a csv and now I read it in


# In[4]:


df.head() # Check that it looks correct


# In[5]:


df.info() # Just to make sure that we have no null values, which we don't


# In[6]:


df.describe() # Look at some descriptive statistics


# We can already see some important things from the descriptive statistics. The first thing is that there are many variables whose minimum is 0. This means a player averaged 0 points per game or 0 free throw attempts per game, or whatever the variable may be. That's not going to help us, because it means those players are generally "garbage time players". They only play in the last few minutes when the score is lopsided. Even worse is a NETRTG of -100 or a PIE of -10.8. These are outliers caused by a lack of playing time and they could affect what we are trying to do. I will filter out those that don't play enough. The second thing is that there doesn't seem to be any data errors. There's no negative numbers where there shouldn't be or things of that nature. The third thing is that these variables are on different scales. We may need to standardize or normalize later on.

# ### Correlations

# Let's look at the pairwise correlations and see what variables are highly correlated

# In[7]:


correlations = df.corr()
correlations


# In[8]:


# Lets now only see the correlations that meet certain thresholds. I set mine as 0.75. I also eliminate those that are the correlation of a column against itself

threshold = 0.75

positive_correlations = correlations[(correlations > threshold) & (correlations < 1)]


# In[9]:


# Now we print the correlations and columns that have those correlations that meet the threshold

for column in positive_correlations:
    above_threshold = positive_correlations[column].dropna()
    for col, corr in above_threshold.iteritems():
        print(f"Correlation: {corr:.3f}, Columns: {column} and {col}")


# In[10]:


# This one is for variables that are negatively correlated
threshold = -0.75

negative_correlations = correlations[correlations < threshold]

for column in negative_correlations:
    above_threshold = negative_correlations[column].dropna()
    for col, corr in above_threshold.iteritems():
        print(f"Correlation: {corr:.3f}, Columns: {column} and {col}")


# There are many variables that are highly correlated that make sense. For example, FTM and FTA or REB and DREB. Knowing what we know about basketball, it makes sense that as the number of defensive rebounds go up, so do the number of total rebounds. In addition, the negative correlations make sense as well. For example, if you take more 2 point shots, then your percentage of 3 point shots will go down. There are many examples of this. These heavily correlated features will need to be taken care of, or we will need to use an algorithm that is not sensitive to high correlations

# ### Visualizations

# There are 60 columns in our dataframe, 58 of which are numeric. It's difficult to view all of those at the same time, so we will split them up into 3 sections so we can get a good look at histograms of all of them. I find it easier to look at the distibution using a KDE plot, so I have included those as well.

# In[11]:


num_rows = 5
num_cols = 4

# Create subplots for histograms
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Generate histograms for all variables
for i, column in enumerate(df.columns[2:22]):
    ax = axes[i // num_cols, i % num_cols]  # Get the appropriate subplot
    ax.hist(df[column])
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

# Adjust spacing between subplots
fig.tight_layout()

# Display the plot
plt.show()


# In[12]:


num_rows = 5
num_cols = 4

# Create subplots for histograms
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Generate KDE plots for all variables
for i, column in enumerate(df.columns[2:22]):
    ax = axes[i // num_cols, i % num_cols]  # Get the appropriate subplot
    sns.kdeplot(data=df[column], ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Density')

# Adjust spacing between subplots
fig.tight_layout()

# Display the plot
plt.show()


# In[13]:


num_rows = 5
num_cols = 4

# Create subplots for histograms
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Generate histograms for all variables
for i, column in enumerate(df.columns[22:42]):
    ax = axes[i // num_cols, i % num_cols]  # Get the appropriate subplot
    ax.hist(df[column])
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')


# Adjust spacing between subplots
fig.tight_layout()

# Display the plot
plt.show()


# In[14]:


num_rows = 5
num_cols = 4

# Create subplots for histograms
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Generate KDE plots for all variables
for i, column in enumerate(df.columns[22:42]):
    ax = axes[i // num_cols, i % num_cols]  # Get the appropriate subplot
    sns.kdeplot(data=df[column], ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Density')


# Adjust spacing between subplots
fig.tight_layout()

# Display the plot
plt.show()


# In[15]:


num_rows = 5
num_cols = 4

# Create subplots for histograms
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Generate histograms for all variables
for i, column in enumerate(df.columns[42:]):
    ax = axes[i // num_cols, i % num_cols]  # Get the appropriate subplot
    ax.hist(df[column])
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

# Hide empty subplots
if i < num_rows * num_cols - 1:
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])

# Adjust spacing between subplots
fig.tight_layout()

# Display the plot
plt.show()


# In[16]:


num_rows = 5
num_cols = 4

# Create subplots for histograms
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Generate KDE plots for all variables
for i, column in enumerate(df.columns[42:]):
    ax = axes[i // num_cols, i % num_cols]  # Get the appropriate subplot
    sns.kdeplot(data=df[column], ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Density')

# Hide empty subplots
if i < num_rows * num_cols - 1:
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])

# Adjust spacing between subplots
fig.tight_layout()

# Display the plot
plt.show()


# In[17]:


# If you want a closer look at each histogram, you can run this code. It will be commented out for now since there are many graphs

#for column in df.columns[2:]:
 #   plt.hist(df[column])
  #  plt.xlabel(column)
   # plt.ylabel('Frequency')
    #plt.show()


# There are a number of things we can see from these graphs. Many of the offensive statistics are positively skewed. This makes sense given that there are rather few players that would have large amounts of points, rebounds, assists, etc. in a game. These players would be considered superstars, and there aren't that many in the league. There are also several variables that are not skewed or at least close to normal. These include DEFRTG, NETRTG, EFG%, TS%, PIE and PACE. There are other variables that are oddly distributed, such as W, POSS and Min. 

# In[ ]:





# ### Clustering

# The main point of my project is to try to cluster the different players in the NBA based on their statistics from this past year. That involves some sort of clustering. I will try to just run the clusters and look at the labels of each, and later I will use dimension reduction to be able to visualize the clusters a little better. 

# In[18]:


df.head()


# In[19]:


#We need to drop all non-numeric columns. We will drop Player and Team.

df1 = df.drop(['Player', 'Team'], axis=1)


# In[20]:


df1.head()


# In[21]:


# Next we need to standardize the data since clustering is sensitive to scale. I will use standard scaler

scaler = StandardScaler()
X = scaler.fit_transform(df1)


# The common way of determining the correct number of clusters for kmeans clustering is the elbow method. However, the elbow method may not always be the most accurate, particularly when the number of clusters is high. The Calinski-Harabasz score, Davies-Bouldin score and the Silhouette score can all give a more accurate number of clusters. I will run each of the four methods here, and I will compare. Check out the Sklearn notes for each of these and other methods under clustering metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# 
# In addition, check out this blog post: https://towardsdatascience.com/are-you-still-using-the-elbow-method-5d271b3063bd
# This was one of the main reasons I chose the metrics I used. Because we don't have the true labels, many of the other clustering metrics don't work. That's why I chose the three scores given above 

# In[22]:


wcss = []
ch_scores = []
db_scores = []
s_scores = []

for i in range(2, 25):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 9)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    ch_scores.append(calinski_harabasz_score(X, kmeans.labels_))
    db_scores.append(davies_bouldin_score(X, kmeans.labels_))
    s_scores.append(silhouette_score(X, kmeans.labels_))


# In[23]:


fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(range(2, 25), wcss)
ax[0,0].set_title('K Means with Elbow Method (Lowest)')
ax[0,0].set_xticks(range(2,25))
ax[0,1].plot(range(2, 25), ch_scores)
ax[0,1].set_title('K Means with Calinski-Harabasz Method (Highest)')
ax[0,1].set_xticks(range(2,25))
ax[1,0].plot(range(2, 25), s_scores)
ax[1,0].set_title('K Means with Silhouette Method (Highest)')
ax[1,0].set_xticks(range(2,25))
ax[1,1].plot(range(2, 25), db_scores)
ax[1,1].set_title('K Means with Davies-Boulding Method (Lowest)')
ax[1,1].set_xticks(range(2,25))

fig.show()


# The four methods don't come to a concensus on the correct number of clusters, though they don't necessarily need to. The thing that concerns me is that none of them are even close. The elbow method is smooth and curved, with no distinct "elbow". The same can be said for the Calinski-Harabasz method. The silhouette method recommends anything from 2 to 5 and maybe 10. The Davies-Boulding method recommends 16 or 20. I suspect there are issues here that we can address, so I'm not even going to perform the clustering here. We already know there are variables that are highly correlated. So I will perform PCA and later on I will also filter the data. 

# ### Using PCA and clustering

# In[24]:


pca1 = PCA(n_components=25)
X_pca1 = pca1.fit_transform(X)


# In[25]:


plt.bar(range(1, 21), pca1.explained_variance_ratio_[0:20])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1, 21))
plt.show()


# In[26]:


pca1.explained_variance_ratio_


# The first 3 components explain almost 53% of the variation in the data so I will just use those 3. You could make an argument for other numbers as well such as 2 or 4. I feel comfortable with the first 3. 

# In[27]:


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)


# In[28]:


# This is the same as above. I get the 4 scores and plot them for this new PCA data

wcss = []
ch_scores = []
db_scores = []
s_scores = []

for i in range(2, 25):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 9)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
    ch_scores.append(calinski_harabasz_score(X_pca, kmeans.labels_))
    db_scores.append(davies_bouldin_score(X_pca, kmeans.labels_))
    s_scores.append(silhouette_score(X_pca, kmeans.labels_))


# In[29]:


fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(range(2, 25), wcss)
ax[0,0].set_title('K Means with Elbow Method (Lowest)')
ax[0,0].set_xticks(range(2,25))
ax[0,1].plot(range(2, 25), ch_scores)
ax[0,1].set_title('K Means with Calinski-Harabasz Method (Highest)')
ax[0,1].set_xticks(range(2,25))
ax[1,0].plot(range(2, 25), s_scores)
ax[1,0].set_title('K Means with Silhouette Method (Highest)')
ax[1,0].set_xticks(range(2,25))
ax[1,1].plot(range(2, 25), db_scores)
ax[1,1].set_title('K Means with Davies-Boulding Method (Lowest)')
ax[1,1].set_xticks(range(2,25))

fig.show()


# Performing PCA has given a more concrete answer to the number of clusters. The elbow method might give about 6, though it's still pretty smooth. However, the other 3 methods all have definitive spikes in their respective directions at 6. In addition, the bottom two methods both have distinctive spikes at 17. I will look at the clustering with both 6 and 17 clusters. 

# In[30]:


kmeans = KMeans(n_clusters=6)
kmeans.fit(X_pca)

labels6 = kmeans.labels_

df['labels_6'] = labels6

kmeans2 = KMeans(n_clusters=17)
kmeans2.fit(X_pca)

labels17 = kmeans2.labels_

df['labels_17'] = labels17


# In[31]:


# Make sure it looks correct
df.head()


# In[32]:


df['labels_6'].value_counts() # See how many are in each


# In[33]:


df['labels_17'].value_counts() # See how many are in each


# In[34]:


df.loc[df['labels_6'] == 1, 'Player'] # Look at some of the players. 


# In[35]:


# This creates a new dataframe that we will use for the visualizations. Includes the player names, the PCA components and the cluster labels for 6 and 17
df_new = pd.DataFrame({'Player': df.Player, 'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'PC3': X_pca[:, 2], 'Cluster6': df.labels_6, 'Cluster17':df.labels_17})


# In[36]:


# Make sure the cluster labels are categorical 
df_new['Cluster6'] = df_new['Cluster6'].astype('category')
df_new['Cluster17'] = df_new['Cluster17'].astype('category')


# In[37]:


fig = px.scatter(df_new, x='PC1', y='PC2', color='Cluster6', hover_data=['Player'],
                 title="2023 NBA Regular Season Stats Clustering with 6 clusters on 2 Components")

fig.show()


# In[38]:


fig = px.scatter(df_new, x='PC1', y='PC2', color='Cluster17', hover_data=['Player'],
                 title="2023 NBA Regular Season Stats Clustering with 17 clusters on 2 Components",
                color_discrete_sequence=px.colors.qualitative.Alphabet)

fig.show()


# I did include all 3 components in the data, so we may as well look at the 3d version

# In[39]:


fig = px.scatter_3d(df_new, x='PC1', y='PC2',z='PC3', color='Cluster6', hover_data=['Player'],
                 title="2023 NBA Regular Season Stats Clustering with 6 clusters on 3 Components")

fig.show()


# In[40]:


fig = px.scatter_3d(df_new, x='PC1', y='PC2', z='PC3', color='Cluster17', hover_data=['Player'],
                    title="2023 NBA Regular Season Stats Clustering with 17 clusters on 3 Components",
                    color_discrete_sequence=px.colors.qualitative.Alphabet)

fig.show()


# All 4 of these graphs show interesting trends amongst the data. I am not attempting to label any of these clusters, though we could do that. I won't pretend I know enough about the intricate nature of basketball or all the players that played this year. However, I can tell that in the graphs with 6 clusters, cluster 2 would probably be considered high rebounding, traditional big men, cluster 5 would be "superstars" clusters 0 and 3 would probably be more "role players" and clusters 1 and 4 would likely be players that don't play as much. With the 17 clusters it's much more difficult for me to define. However, it's interesting to look at who is in each cluster. 

# ### Running clustering with filtered data

# One of the issues I can see is that there a number of players who don't play much. I said before I would filter out these players, but I wanted to see what the clusters would look like with those players in the data set. I will now filter out those players.

# In[41]:


df.head()


# There's a definitive problem in figuring out where the threshold is for players to be kept in the dataset. I landed on 21 games played and 8 minutes per game. In this sense, the player must have played over a quarter of the games of the NBA season and played at least a sixth of the minutes per game. You could certainly make the case for other numbers as well. I felt this was a reasonable amount that would eliminate garbage time players and players that were injured for most of the season.

# In[42]:


filtered_df = df[(df['GP'] > 21) & (df['Min'] > 8)].copy()
filtered_df1 = filtered_df.drop(['Player', 'Team', 'labels_6', 'labels_17'], axis=1)


# In[43]:


filtered_df1.head() # Take a look


# In[44]:


filtered_df1.info() # Make sure we have 416 players


# We will do the same scaling, as well as scoring and visualizations to determine how many clusters we should include

# In[45]:


scaler = StandardScaler()
filtered_X = scaler.fit_transform(filtered_df1)


# In[46]:


wcss = []
ch_scores = []
db_scores = []
s_scores = []

for i in range(2, 25):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 9)
    kmeans.fit(filtered_X)
    wcss.append(kmeans.inertia_)
    ch_scores.append(calinski_harabasz_score(filtered_X, kmeans.labels_))
    db_scores.append(davies_bouldin_score(filtered_X, kmeans.labels_))
    s_scores.append(silhouette_score(filtered_X, kmeans.labels_))


# In[47]:


fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(range(2, 25), wcss)
ax[0,0].set_title('K Means with Elbow Method (Lowest)')
ax[0,0].set_xticks(range(2,25))
ax[0,1].plot(range(2, 25), ch_scores)
ax[0,1].set_title('K Means with Calinski-Harabasz Method (Highest)')
ax[0,1].set_xticks(range(2,25))
ax[1,0].plot(range(2, 25), s_scores)
ax[1,0].set_title('K Means with Silhouette Method (Highest)')
ax[1,0].set_xticks(range(2,25))
ax[1,1].plot(range(2, 25), db_scores)
ax[1,1].set_title('K Means with Davies-Boulding Method (Lowest)')
ax[1,1].set_xticks(range(2,25))

fig.show()


# This is really encouraging. By filtering out the players that don't play much, the elbow method has a little more of an elbow at 3 (though it's still pretty smooth), and the other 3 methods all point to 3 clusters. While the Davies-Boulding method suggests 15 or 18 as well, I will stick with 3

# In[48]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(filtered_X)

labels3 = kmeans.labels_ # Save the labels

# Use this line to make sure a warning does not appear as a copy of a slice
filtered_df.loc[:, 'labels_3'] = labels3


# In[49]:


filtered_df['labels_3'] = filtered_df['labels_3'].astype('category') # make sure these are categorical


# Let's look at some of the players in each of the clusters

# In[50]:


filtered_df.loc[filtered_df['labels_3'] == 1, 'Player']


# In[51]:


filtered_df.loc[filtered_df['labels_3'] == 0, 'Player']


# In[52]:


filtered_df.loc[filtered_df['labels_3'] == 2, 'Player']


# In[53]:


filtered_df.head() # Take a look


# When I don't have dimension reduction, it's a little more difficult to get a good visualization. There are so many columns to look at that with 2 or even 3 dimensions we are not likely to see definitive splits. That being said, it's interesting to see. We can change the variables as well to look at different combinations of variables.

# In[54]:


fig = px.scatter(filtered_df, x='NETRTG', y='PIE', color='labels_3', hover_data=['Player'],
                 title="2023 NBA Regular Filtered Season Stats Clustering on 2 Variables")

fig.show()


# In[55]:


fig = px.scatter_3d(filtered_df, x='NETRTG', y='PIE', z='TS%', color='labels_3', hover_data=['Player'],
                 title="2023 NBA Regular Filtered Season Stats Clustering on 3 Variables")

fig.show()


# In this case, I honestly don't have any definitive labels for these clusters. The only thing I can see is that many of the players in cluster 2 are big men that generally have a lot of rebounds. Other than that, I'm not entirely sure.

# ### Using PCA and clustering on the filtered data

# Like before, I will look at PCA for this filtered data

# In[56]:


filtered_pca1 = PCA(n_components = 25)
filtered_X_pca1 = filtered_pca1.fit_transform(filtered_X)


# In[57]:


plt.bar(range(1, 21), filtered_pca1.explained_variance_ratio_[0:20])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1, 21))
plt.show()


# In[58]:


filtered_pca1.explained_variance_ratio_


# The first 4 components explain about 65% of the variance in the filtered data. So I am going to keep those components. An argument could have been made to just keep the first 2 considering they explain 48% of the variance. I felt 4 components was an appropriate amount

# This is the same as before, I run PCA, get the scores and plot them to find the correct number of clusters

# In[59]:


filtered_pca = PCA(n_components = 4)
filtered_X_pca = filtered_pca.fit_transform(filtered_X)


# In[60]:


wcss = []
ch_scores = []
db_scores = []
s_scores = []

for i in range(2, 25):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 9)
    kmeans.fit(filtered_X_pca)
    wcss.append(kmeans.inertia_)
    ch_scores.append(calinski_harabasz_score(filtered_X_pca, kmeans.labels_))
    db_scores.append(davies_bouldin_score(filtered_X_pca, kmeans.labels_))
    s_scores.append(silhouette_score(filtered_X_pca, kmeans.labels_))


# In[61]:


fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(range(2, 25), wcss)
ax[0,0].set_title('K Means with Elbow Method (Lowest)')
ax[0,0].set_xticks(range(2,25))
ax[0,1].plot(range(2, 25), ch_scores)
ax[0,1].set_title('K Means with Calinski-Harabasz Method (Highest)')
ax[0,1].set_xticks(range(2,25))
ax[1,0].plot(range(2, 25), s_scores)
ax[1,0].set_title('K Means with Silhouette Method (Highest)')
ax[1,0].set_xticks(range(2,25))
ax[1,1].plot(range(2, 25), db_scores)
ax[1,1].set_title('K Means with Davies-Boulding Method (Lowest)')
ax[1,1].set_xticks(range(2,25))

fig.show()


# We can see from our graphs above that the silhouette, Davies-Boulding and Calinski-Harabasz scores all give 3 clusters as the best number of clusters, just like with the original filtered data. We will perform k-means with 3 clusters and see the results. 

# In[62]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(filtered_X_pca)

labels3_pca = kmeans.labels_

# Use this line to make sure a warning does not appear as a copy of a slice
filtered_df.loc[:, 'labels_3_pca'] = labels3_pca


# In[63]:


filtered_df.head()


# In[64]:


filtered_df.labels_3_pca.value_counts() # Look at the distribution


# In[65]:


filtered_df_new_pca = pd.DataFrame({'Player': filtered_df.Player, 'PC1': filtered_X_pca[:, 0], 'PC2': filtered_X_pca[:, 1], 'PC3': filtered_X_pca[:, 2], 'Cluster': filtered_df.labels_3_pca})
filtered_df_new_pca['Cluster'] = filtered_df_new_pca['Cluster'].astype('category')


# In[66]:


fig = px.scatter(filtered_df_new_pca, x='PC1', y='PC2', color='Cluster', hover_data=['Player'],
                 title="2023 NBA Regular Filtered PCA Season Stats Clustering on 2 Components")

fig.show()


# In[67]:


fig = px.scatter_3d(filtered_df_new_pca, x='PC1', y='PC2',z='PC3', color='Cluster', hover_data=['Player'],
                 title="2023 NBA Regular Filtered PCA Season Stats Clustering on 3 Components")

fig.show()


# In terms of the players in each cluster, PCA and the regular filtered data aren't that different. That tells me that the players that were only playing a few minutes were causing some problems. In terms of the visualizations, PCA obviously makes it easier to see the clusters since we are looking at components and not actual variables. The interesting this is that I can see a more definitive break in the clusters. I can essentially label them. Label 2 contains your more traditional high-rebounding and blocking, lower three point shooting big men. This includes Mitchell Robinson, Rudy Gobert, Deandre Ayton, etc. Cluster 0 is your more traditional high volume players. Essentially these are the best players on teams and can often be considered superstars. This includes Luka Doncic, Devin Booker, Giannis Antetokounmpo, Nikola Jokic and many others. Cluster 1 is essentially everyone else. This includes 3 and d players (Grand Williams, Cam Johnson, Michael Porter Jr.) and other role players along with those that don't play as much. I like this because we can see the break in where players end up a little more.
