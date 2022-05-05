---
layout: post
title: Temperature Analysis and Visualization
---
# Temperature Analysis and Visualization

## Preparation

First we need to import all the modules we need


```python
import pandas as pd
import sqlite3
import numpy as np
from plotly import express as px
from sklearn.linear_model import LinearRegression
import calendar
import plotly.graph_objects as go
from plotly.io import write_html
```

## Data Import

Import the temps, countries and stations as data frames


```python
temps = pd.read_csv("temps_stacked.csv")
temps.head()
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
      <th>ID</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>1</td>
      <td>-0.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>2</td>
      <td>2.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>3</td>
      <td>4.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>4</td>
      <td>7.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>5</td>
      <td>11.28</td>
    </tr>
  </tbody>
</table>
</div>




```python
countries = pd.read_csv('countries.csv')
countries.head()
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
      <th>FIPS 10-4</th>
      <th>ISO 3166</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>




```python
countries = countries.rename(columns= {"FIPS 10-4": "FIPS_10-4"})
countries.head()
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
      <th>FIPS_10-4</th>
      <th>ISO 3166</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>




```python
stations = pd.read_csv('station-metadata.csv')
stations.head()
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
      <th>ID</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>STNELEV</th>
      <th>NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AE000041196</td>
      <td>25.3330</td>
      <td>55.5170</td>
      <td>34.0</td>
      <td>SHARJAH_INTER_AIRP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AEM00041184</td>
      <td>25.6170</td>
      <td>55.9330</td>
      <td>31.0</td>
      <td>RAS_AL_KHAIMAH_INTE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AEM00041194</td>
      <td>25.2550</td>
      <td>55.3640</td>
      <td>10.4</td>
      <td>DUBAI_INTL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AEM00041216</td>
      <td>24.4300</td>
      <td>54.4700</td>
      <td>3.0</td>
      <td>ABU_DHABI_BATEEN_AIR</td>
    </tr>
  </tbody>
</table>
</div>



## Create a Database

Create a database with the three data frames.


```python
conn = sqlite3.connect("temps.db")
temps.to_sql("temperatures", conn, if_exists="replace", index=False)
countries.to_sql("countries", conn, if_exists="replace", index=False)
stations.to_sql("stations", conn, if_exists="replace", index=False)
# always close your connection
conn.close()
```

    C:\Users\tfq21\anaconda3\lib\site-packages\pandas\core\generic.py:2872: UserWarning:
    
    The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
    
    

## The Query Function

Define a query function of sqlite to get the data we need.


```python
def query_climate_database(country, year_begin, year_end, month):
    #Input: three variables
    #Country: the country we want to investigate
    #year_begin,year_end: the year period starting at year_begin, ending at year_end
    #month: the month that we want to investigate
    #Output: the data frame with all the desired data
    conn = sqlite3.connect("temps.db")
    cmd = f"""SELECT S.name, S.latitude, S.longitude, C.name, T.year, T.month, T.temp\
              FROM temperatures T\
              LEFT JOIN stations S ON T.id=S.id\
              LEFT JOIN countries C ON C."FIPS_10-4" =substring(T.id,1,2)\
              WHERE C.Name = '{country}' AND T.year >= {year_begin} AND T.year<={year_end} AND T.month ={month}"""
    
    df = pd.read_sql(cmd, conn)
    conn.close()
    return df    
```

Get the data of Inida from 1980 to 2020 on Janurary 


```python
df = query_climate_database("India",1980,2020,1)
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Name</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
  </tbody>
</table>
</div>



## Geographic Scatter Function

First we need a helper function coef to compute the slope


```python
def coef(data_group):
    #Input:a data frame
    #output: the slope of the temperature
    X = data_group[["Year"]]
    y = data_group["Temp"]
    LR = LinearRegression()
    LR.fit(X, y)
    slope = LR.coef_[0]
    return slope
```

Define the plot function.


```python
def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    #Input: six variables
    #Country: the country we want to investigate
    #year_begin,year_end: the year period starting at year_begin, ending at year_end
    #month: the month that we want to investigate
    #Output: the plot we generate
    df = query_climate_database(country, year_begin, year_end, month)
    df['len']=df.groupby(["NAME"])["Temp"].transform(len)
    df=df[df['len']>= min_obs]
    
    coefs = df.groupby(["NAME"]).apply(coef)
    coefs = coefs.reset_index()
    coefs[0]=coefs[0].round(4)
    df =pd.merge(df,coefs,how='right')
    df.rename(columns={0:'Estimated Yearly Increase (°C)'},inplace = True)
    fig = px.scatter_mapbox(df, # data for the points you want to plot
                            lat = "LATITUDE", # column name for latitude informataion
                            lon = "LONGITUDE", # column name for longitude information
                            hover_name = "NAME", # what's the bold text that appears when you hover over
                            zoom = 1, # how much you want to zoom into the map
                            height = 500, # control aspect ratio
                            mapbox_style="carto-positron", # map style
                            opacity=0.2,# opacity for each data point
                            color = "Estimated Yearly Increase (°C)",# represent temp increase using color
                            color_continuous_midpoint = 0,
                            title = f"Estimated Yearly Increase in temperature in {calendar.month_name[month]} for stations in {country}, years {year_begin}-{year_end}"
                           )
    return fig
```

Draw the graph of India from 1980 to 2020 Janurary.


```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig_india = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)

write_html(fig_india, "fig_india.html")
```
{% include fig_india.html %}

Draw the graph of China from 1985 to 2015, Janurary.

{% include fig_China.html %}

```python
fig_China = temperature_coefficient_plot("China", 1985, 2015, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron")
write_html(fig_China, "fig_China.html")
```

## The Query Function 2

Extract the temperature data, given a longtitude range and time period


```python
def query_climate_database2(year_begin,year_end, month, long_begin, long_end):
    #Input: five variables
    #year_begin,year_end: the year period starting at year_begin, ending at year_end
    #month: the month that we want to investigate
    #long_begin, long_end: the longitude range starting at long_begin, ending at long_end
    #Output: the data frame with all the desired data
    conn = sqlite3.connect("temps.db")
    cmd = f"""SELECT S.name, S.latitude, S.longitude, C.name, T.year, T.month, T.temp\
              FROM temperatures T\
              LEFT JOIN stations S ON T.id=S.id\
              LEFT JOIN countries C ON C."FIPS_10-4" =substring(T.id,1,2)\
              WHERE S.longitude >={long_begin} AND S.longitude <={long_end} \
              AND T.year >= {year_begin} AND T.year <= {year_end} AND T.month ={month}"""
    
    df = pd.read_sql(cmd, conn)
    conn.close()
    return df
```

Extract the temperature for longitude 30 to 32, from 2018 to 2020.


```python
df2 = query_climate_database2(2018,2020, 1, 30, 32)
df2.head()
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Name</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VITEBSK</td>
      <td>55.167</td>
      <td>30.217</td>
      <td>Belarus</td>
      <td>2018</td>
      <td>1</td>
      <td>-3.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VITEBSK</td>
      <td>55.167</td>
      <td>30.217</td>
      <td>Belarus</td>
      <td>2019</td>
      <td>1</td>
      <td>-6.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VITEBSK</td>
      <td>55.167</td>
      <td>30.217</td>
      <td>Belarus</td>
      <td>2020</td>
      <td>1</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORSHA</td>
      <td>54.500</td>
      <td>30.417</td>
      <td>Belarus</td>
      <td>2018</td>
      <td>1</td>
      <td>-3.58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORSHA</td>
      <td>54.500</td>
      <td>30.417</td>
      <td>Belarus</td>
      <td>2019</td>
      <td>1</td>
      <td>-6.01</td>
    </tr>
  </tbody>
</table>
</div>



## Scatter plot function

We want to analyze the relation between temperature and latitude


```python
def scatter_plot(year_begin,year_end, month, long_begin, long_end, **kwags):
    #Input: five variables
    #year_begin,year_end: the year period starting at year_begin, ending at year_end
    #month: the month that we want to investigate
    #long_begin, long_end: the longitude range starting at long_begin, ending at long_end
    #Kwags: optional argument for graph
    #Output: the scatter plot
    df = query_climate_database2(year_begin,year_end, month, long_begin, long_end)


    fig = px.scatter(data_frame = df,
                      x = "LATITUDE",
                      y = "Temp",
                      hover_name = "NAME",
                      hover_data =["Name","LONGITUDE"],
                      color ="Name",
                      facet_row = "Year"
                      )
    return fig
```

Visualize the realtion between temperatures and latitude for longtitude 30 to 32, from 2018 to 2020


```python
fig_scatter = scatter_plot(2018,2020, 1, 30, 32)
write_html(fig_scatter, "fig_scatter.html")
```
{% include fig_scatter.html %}

We can see from the plot that as the latitude increases, the temperature drops.

## The Query Function 3

Extract the temperature data, given a location with latitude range and longitude, for a year


```python
def query_climate_database3(year,long_begin, long_end, lat_begin,lat_end):
    #Input: five variables
    #year: the year to be analyzed
    #long_begin, long_end: the longitude range starting at long_begin, ending at long_end
    #lat_begin, lat_end: the latitude range starting at lat_begin, ending at lat_end
    #Output: the data frame with all the desired data
    conn = sqlite3.connect("temps.db")
    cmd = f"""SELECT S.name, S.latitude, S.longitude, C.name, T.year, T.month, T.temp\
              FROM temperatures T\
              LEFT JOIN stations S ON T.id=S.id\
              LEFT JOIN countries C ON C."FIPS_10-4" =substring(T.id,1,2)\
              WHERE T.year={year} AND (T.month=1 or T.month=7) AND S.longitude >={long_begin} \
              AND S.longitude <={long_end} AND S.latitude >={lat_begin} AND S.latitude <={lat_end}
              """
    
    df = pd.read_sql(cmd, conn)
    conn.close()
    return df
```

## Density_heat plot function

Visualize the temperature difference between Janurary and July

Define a plot function with the given data


```python
def heat_density_plot(year,long_begin, long_end, lat_begin,lat_end,**kwags):
    #Input: five variables
    #year: the year to be analyzed
    #long_begin, long_end: the longitude range starting at long_begin, ending at long_end
    #lat_begin, lat_end: the latitude range starting at lat_begin, ending at lat_end
    #Kwags: optional argument for graph
    #Output: the heat density plot
    df = query_climate_database3(year,long_begin, long_end, lat_begin,lat_end)
    fig = px.density_heatmap(df, 
                             x = "LATITUDE",
                             y = "LONGITUDE",
                             z = "Temp",
                             facet_row = "Month",
                             hover_name = "NAME",
                             nbinsx = 25,
                             nbinsy = 25)
    return fig
```

Draw the density graph for the location with latitude from 30 to 50, longitude from 80 to 110 of 2019


```python
fig_heat = heat_density_plot(2019,80,110,30,50)
write_html(fig_heat, "fig_heat.html")
```
{% include fig_heat.html %}