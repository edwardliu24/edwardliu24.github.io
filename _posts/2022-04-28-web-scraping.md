---
layout: post
title: IMDB scraping
---

# IMDB website scraping

In this project, we will find the movie with most shared actors, given your favourite movie

## Link to github

"https://github.com/edwardliu24/web-scraping"

## Preparation

First import all these modules for the project function properly


```python
import scrapy
from scrapy.http import Request
import pandas as pd
from plotly import express as px
from plotly.io import write_html
```

Open the terminal, at the directory you want your project be, run the command "scrapy startproject IMDB_scraper",navigate to the directory, add CLOSESPIDER_PAGECOUNT = 20 to the file "settings.py"

## Coding the spider

Create a file named IMDB_spider.py under the folder named spider, and in the py file, define a Imdbspider class.


```python
class ImdbSpider(scrapy.Spider):
    ## spider class with name "imdb_spider",the url starts with is "https://www.imdb.com/title/tt1533117/?ref_=fn_al_tt_1"
    name = 'imdb_spider'
    
    start_urls = ["https://www.imdb.com/title/tt1533117/?ref_=fn_al_tt_1"]
```

This is the parse method would be used at the initial page


```python
def parse(self, response):
    '''
    Scrape on the initilia page and open the 'cast&crew' link
    Input: response, the html source code of the initial page
    Output: link to the cast page, and call parse_full_credits on that page
    
    '''
    ##Extract the link of "cast&crew"
    full_credits = response.css("li.ipc-inline-list__item a[href*=fullcredits]").attrib['href']
    
    ##Make the next url
    prefix = "https://www.imdb.com/title/tt1533117/"
    cast_url = prefix + full_credits

    ##Request the parse_full_credits method on the next url
    yield Request(cast_url, callback = self.parse_full_credits)

```

This is the parse method would be used at the cast&crew page


```python
def parse_full_credits(self, response):
    '''
    Find all the actor page of the actors in the movie
    Input: response, the html source code of the cast page
    Output: link to the actor page, and call parse_actor_page on that page
    '''
    ##Get all the links of each ators
    actor_page = [a.attrib["href"] for a in response.css("td.primary_photo a")]
    
    ##Make a list of all the links to actor page
    prefix = "https://www.imdb.com/"
    actor_url = [prefix + suffix for suffix in actor_page]

    ##For each link, open the actor page and call the parse_actor_page method
    for url in actor_url:
        yield Request(url, callback = self.parse_actor_page)

```

This is the parse method would be used at the actor page


```python
def parse_actor_page(self, response):
    '''
    Scrape on the actor page, scrape all the movies that this actor participated,output a dictionary with actor names and movies
    Input: response, the html source code of the actor page
    Output: A dictionary with actor names and movie names
    '''
    ##Scrpae the names of the actor
    name = response.css("h1.header span::text").get()
    
    ##Scrape all the movies that this actor was in
    for movie in response.css("div.filmo-row"):
        movies = movie.css("a::text").get()
        
        ##yield a dictionary
        yield {
            "actor" : name,
            "movies" : movies
        }
```

All these three functions should be defined under the imdbspider class

## Method implementing

After finishing coding the spider file, we could run "scrapy crawl imdb_spider -o results.csv" in the terminal to get the results, then we get a file named "resultys.csv"


```python
result=pd.read_csv("results.csv")
```


```python
df = result.value_counts(['movies'])
```

Do some data cleaning to get the desired results.


```python
result = pd.read_csv("results.csv")
df = result.value_counts(['movies'])
df = pd.DataFrame(df)
df = df.reset_index()
df.columns = ['movies','number of shared actors']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movies</th>
      <th>number of shared actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Let the Bullets Fly</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gone with the Bullets</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Sun Also Rises</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hidden Man</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Founding of a Republic</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



Draw a scatter plot to visualize the results.


```python
fig = px.scatter(data_frame = df, 
                 x = 'movies',
                 y = 'number of shared actors', 
                 width = 1000,
                 height = 700,
                 title = "Scatter plot of the counts of shared actors")
write_html(fig, "movie_recommend.html")
```
{% include movie_recommend.html %}