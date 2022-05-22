#!/usr/bin/env python
# coding: utf-8

# In[1]:


from requests import get
from bs4 import BeautifulSoup
from warnings import warn
from time import sleep
from random import randint
import numpy as np, pandas as pd


# In[6]:


# initialize empty lists to store the variables scraped
titles = []
genres = []
plot = []


# In[41]:


def crawl(genre_str):
    pages = np.arange(1, 5051, 50)  # can only go to 10000 items
    for page in pages:
        # get request
        response = get("https://www.imdb.com/search/title?genres="
                       + genre_str
                       + "&"
                       + "start="
                       + str(page)
                       + "&explore=title_type,genres&ref_=adv_prv")

        sleep(randint(3, 6))

        # throw warning for status codes that are not 200
        if response.status_code != 200:
            warn('Request: {}; Status code: {}'.format(response, response.status_code))

        # parse the content of current iteration of request
        page_html = BeautifulSoup(response.text, 'html.parser')

        movie_containers = page_html.find_all('div', class_='lister-item mode-advanced')

        # extract the 50 movies for that page
        for container in movie_containers:
            # title
            try:
                title = container.h3.a.text
            except AttributeError:
                title = '-'
            if title in titles:
                continue

            titles.append(title)

            # genre
            try:
                genre = container.p.find('span', class_='genre').text
            except AttributeError:
                genre = '-'
            genres.append(genre)

            # plot
            try:
                m_score = container.find_all('p', class_='text-muted')
                plot.append(m_score[1].text)
            except:
                plot.append('-')


# # Crawling Action Movies

# In[8]:


crawl("action")


# # Crawling Sci-Fi Movies

# In[9]:


crawl("sci-fi")


# # Crawling Comedy Movies

# In[10]:


crawl("comedy")


# # Crawling Horror Movies

# In[11]:


crawl("horror")


# # Crawling Drama Movies

# In[14]:


crawl("drama")


# # Crawling Animation Movies

# In[16]:


crawl("animation")


# # Crawling Mystery Movies

# In[17]:


crawl("mystery")


# # Crawling Crime Movies

# In[19]:


crawl("crime")


# # Crawling Fantasy Movies

# In[21]:


crawl("fantasy")


# # Crawling Thriller Movies

# In[23]:


crawl("thriller")


# # Crawling Romance Movies

# In[25]:


crawl("romance")


# # Crawling Adventure Movies

# In[32]:


crawl("adventure")


# # Crawling Superhero Movies

# In[33]:


crawl("superhero")


# # Crawling Biography Movies

# In[50]:


crawl("biography")


# # Create CSV

# In[53]:


df = pd.DataFrame({'movie': titles,
                    'genre': genres,
                    'plot': plot,
                    })


# In[54]:


# remove the "\n" at the beginning of the genres
df['genre'] = df['genre'].apply(lambda x: x.replace("\n", ""))

# remove the "\n" at the beginning of the plots
df['plot'] = df['plot'].apply(lambda x: x.replace("\n", ""))

# I found that there was whitespace in the arrays of genres so I removed that with .rstrip()
df['genre'] = df['genre'].apply(lambda x: x.rstrip())

# split the string list into an actual array of values to do NLP on later
df['genre'] = df['genre'].str.split(",")


# In[57]:


# write the final dataframe to the working directory
df.to_csv("../data/raw/raw_data_v2.csv", index=False)


# In[58]:


print(len(titles))


# In[ ]:




