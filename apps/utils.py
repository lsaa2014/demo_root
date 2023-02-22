import streamlit as st
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# adtk
from adtk.detector import ThresholdAD
from adtk.detector import LevelShiftAD
from adtk.detector import VolatilityShiftAD
from adtk.detector import SeasonalAD
from adtk.data import validate_series
from adtk.visualization import plot
import plotly.graph_objects as go
import plotly.express as px

# nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string



def date_cols(data):
    ''' Create year, month, month name and day from a datetime column'''

    data['Quarter'] = data.index.quarter
    data['Month'] = data.index.month
    data['Month_name'] = data.index.month_name()
    data['Day'] = data.index.day
    data['weekday'] = data.index.weekday
    data['weekday_name'] = data.index.day_name()
    data['Hour'] = data.index.hour
    data['weekend'] = np.where(data['weekday'].isin([5, 6]), 'Yes' ,'No') 

    return data 

def date_transform(df, encode_cols):
    # extract a few features from datetime
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['WeekofYear'] = df.index.weekofyear
    df['DayofWeek'] = df.index.weekday
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    # one hot encoder for categorical variables
    for col in encode_cols:
        df[col] = df[col].astype('category')
    df = pd.get_dummies(df, columns=encode_cols)
    return df



def TsHourAgg(data, target, period, func):
    
    ## Create time features
    date_cols(data)

    order_month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    order_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
      
    if period == 'Month':
      dg = data.groupby(['Hour','Month_name'])[target].agg(func).unstack()
      dg = dg[order_month]
      dg.columns.name = None
        
    elif period == 'weekday':
        dg = data.groupby(['Hour', 'weekday_name'])[target].agg(func).unstack()[order_week]   
    else:
        dg = data.groupby(['Hour', period])[target].mean().unstack()

    return dg


### Anomaly detection

def health_bar(outliers_proportion):
    labels = "None"
    outliers = outliers_proportion * 100
    inliers = (1 - outliers_proportion) * 100
    fig, ax = plt.subplots(figsize = (24,0.5))
    ax.barh(labels, inliers , color = "green")
    ax.barh(labels, outliers, left=inliers, color = "red")
    # Hide the spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Only show ticks on the  bottom spine
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(labelsize=20)
    #Hide the y xis
    ax.yaxis.set_visible(False)
    ax.set_title('Percentage of outliers: {:.2f}%'.format(outliers),
                 fontsize=25)
    ax.set_xlim(0,100)


def adtk_thres(data, target, low, high):
    ''' Detect outliers whithin low and high range'''
    data = data.sort_index()
    s_train = validate_series(data[target])
    threshold_ad = ThresholdAD(high= high, low=low)
    anomalies = threshold_ad.detect(s_train)
    outlier_frac = (anomalies==1).sum()/len(s_train)
    return (s_train, threshold_ad, anomalies, outlier_frac)

def adtk_thres_plot(s_train, threshold_ad, anomalies, outlier_frac):
  plot_output = plot(s_train, anomaly=anomalies, ts_linewidth=1, ts_markersize=2, 
                  anomaly_markersize=3, anomaly_color='red', anomaly_tag="marker",
                  figsize=(14,8))


def adtk_shift(data, target, c = 6.0, window = 5, freq = 'H'):
    ''' Detects shift of value level by 
    tracking the difference between median 
    values at two sliding time windows next
     to each other.'''
    data = data.sort_index()
    s_train = validate_series(data[target])
    s_train1 = validate_series(data.asfreq(freq=freq, fill_value=0.0)[target])

    level_shift_ad = LevelShiftAD(c=c, side='both', window=window)
    anomalies = level_shift_ad.fit_detect(s_train1)
    outlier_frac = (anomalies==1).sum()/len(s_train1)
    return (s_train, anomalies, outlier_frac)
    

def adtk_shift_plot(s_train, anomalies):   
    plot_output = plot(s_train, anomaly=anomalies, anomaly_alpha=0.6, 
      anomaly_color='red', figsize=(14,8))    


def adtk_vol(data, target, c = 6.0, window = 30, freq = 'H'):
    ''' detects shift of volatility level by tracking the 
        difference between standard deviations at two 
        sliding time windows next to each other.'''
    data = data.sort_index()    
    s_train = validate_series(data[target])
    s_train1 = validate_series(data.asfreq(freq=freq, fill_value=0.0)[target])

    volatility_shift_ad = VolatilityShiftAD(c=c, side='positive', window=window)
    anomalies = volatility_shift_ad.fit_detect(s_train1)
    outlier_frac = (anomalies==1).sum()/len(s_train1)
    return(s_train, anomalies, outlier_frac)
    

def adtk_vol_plot(s_train, anomalies):
    plot(s_train, anomaly=anomalies, anomaly_color='red', figsize=(14,8))      
    

def adtk_season(data, target, c = 4.0, freq = 'H'):
    ''' detects anomalous violations of seasonal pattern'''
    data = data.sort_index()
    s_train = validate_series(data[target])
    s_train1 = validate_series(data.asfreq(freq=freq, fill_value=0.0)[target])

    seasonal_ad = SeasonalAD()
    anomalies = seasonal_ad.fit_detect(s_train1)
    outlier_frac = (anomalies==1).sum()/len(s_train1)
    return(s_train1, anomalies, outlier_frac)
   

def adtk_season_plot(s_train1, anomalies):
    plot(s_train1, anomaly=anomalies, anomaly_color="red", figsize=(14,8))


def resample_plot(data, target, freq):
    dfg = data[target].resample(freq).mean()
    st.line_chart(dfg, use_container_width = False, width = 700)

def resample_plot2(data, target, freq):
    dfg = data[target].resample(freq).mean()
    fig = px.line(dfg, x = dfg.index, y = target)
    fig.update_layout(xaxis_title="", yaxis_title=target[0])
    st.write(fig)    


def aggFreq(df, col, method, selectAg):

    if selectAg == 'H':
      st.subheader("Hourly detection")
      if method == adtk_shift:
        return(adtk_shift(df, col, c = 6.0, window = 5, freq = selectAg))

      elif method == adtk_vol:  
        return adtk_vol(df, col, freq = selectAg)
        
      else:
        return adtk_season(df, col, freq = selectAg)
         
    elif selectAg == 'D':
      st.subheader("Daily detection")
      if method == adtk_shift:
          return(adtk_shift(df, col, c = 6.0, window = 5, freq = selectAg))
          
      elif method == adtk_vol:  
          return adtk_vol(df, col, freq = selectAg)
          
      else:
          return adtk_season(df, col, freq = selectAg)
          
    elif selectAg == 'W':
      st.subheader("Weekly detection")
      if method == adtk_shift:
          return(adtk_shift(df, col, c = 6.0, window = 5, freq = selectAg))
          
      elif method == adtk_vol:  
          return adtk_vol(df, col, freq = selectAg)
          
      else:
          return adtk_season(df, col, freq = selectAg)
          
    elif selectAg == 'Q':
      st.subheader("Quarterly detection")
      if method == adtk_shift:
          return(adtk_shift(df, col, c = 6.0, window = 5, freq = selectAg))
          
      elif method == adtk_vol:  
          return adtk_vol(df, col, freq = selectAg)
          
      else:
          return adtk_season(df, col, freq = selectAg)
          
    else:
      st.subheader("Monthly detection")
      if method == adtk_shift:
          return(adtk_shift(df, col, c = 6.0, window = 5, freq = selectAg))
          
      elif method == adtk_vol:  
          return adtk_vol(df, col, freq = selectAg)
          
      else:
          return adtk_season(df, col, freq = selectAg)

### nlp

def search(data, title):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(data["clean_title"])
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = data.iloc[indices].iloc[::-1]
    
    return results[['movieId', 'title', 'genres', 'rating']]

def find_similar_movies(data, movie_id):
    similar_users = data[(data["movieId"] == movie_id) & (data["rating"] > 3)]["userId"].unique()
    similar_user_recs = data[(data["userId"].isin(similar_users)) & (data["rating"] > 3)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = data[(data["movieId"].isin(similar_user_recs.index)) & (data["rating"] > 3)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.merge(data, left_index=True, right_on="movieId")[["score", "title", "genres"]]


          
