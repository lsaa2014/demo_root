
import pickle as p
import xgboost as xgb
from xgboost.sklearn import XGBRegressor  # wrapper

from utils import *


st.set_option('deprecation.showPyplotGlobalUse', False)

def write():
    """Writes content to the app"""
    st.title("Time Series Analysis\n")
    st.markdown("This application can be used to build prediction and "
                " anomaly detection model for time series 📈")


    df = pd.read_csv('data/house_consumption_18k.csv', index_col= 'Date_Time', parse_dates=True)  


    st.sidebar.markdown("## Anomaly Detection")

    st.header("Plot the raw data based on a frequency")
    st.markdown("First have a look on the general trend of the data based on time frequency "
                  )

    freq_list = ['H', 'D', 'W', 'M', 'BM', 'Q', 'Y']

    selectAg = st.select_slider('Select a frequency', options=freq_list)

    with st.expander("*See code frequency description*"):
         st.write("""D:Calendar day, B:Business day,
            W:Weekly, M:Month end, BM:Business month end, 
            Q:Quarter end, H:Hours, Y:Year
            """)

    options = st.multiselect('Select target', [col for col in df.columns], ['Global_active_power'])  
    #st.write(df.tail())
    # dfg = df[options].resample(selectAg).mean()
    # st.write(dfg.head())
    # st.write(df.columns)
    
    # fig = px.line(dfg, x = dfg.index, y = options)
    # st.write(fig)
    
    names = (op.replace('_', ' ') for op in options)
    if selectAg == 'H':
      st.subheader(f"Hourly average of {', '.join(names)}")
      resample_plot2(df, options, selectAg)           
    elif selectAg == 'D':
      st.subheader(f"Daily average of {', '.join(names)}")
      resample_plot2(df, options, selectAg)
    elif selectAg == 'W':
      st.subheader(f"Weekly average of {', '.join(names)}")
      resample_plot2(df, options, selectAg)
      target = st.selectbox('Select target', (col for col in df.columns))
      st.subheader(f"Weekly average of {target.replace('_', ' ')} by hour")
      # st.write(date_cols(df))
      dg = TsHourAgg(df, target, 'weekday', 'mean')
      fig = px.line(dg, x = dg.index, y = dg.columns, 
        color_discrete_sequence=px.colors.sequential.Viridis, 
        line_shape='spline', labels={"weekday_name": "Day of week"})
      fig.update_traces(mode='lines+markers')
      fig.update_layout(yaxis_title=target)
      st.write(fig)
    elif selectAg == 'Q':
      st.subheader(f"Quarterly average of {', '.join(names)}")
      resample_plot2(df, options, selectAg)
    else:
      st.subheader(f"Monthly average of {', '.join(names)}")
      resample_plot2(df, options, selectAg)


    st.header("Predict Energy consumption")
    number = int(st.number_input('Insert the number of hours', value = 50))
    #modelfile = 'energy_pred.pickle'
    #model = p.load(open('models/energy_pred.pickle', 'rb'))
    
    model = xgb.Booster()
    model.load_model("models/energy_pred.json")
    
    encode_cols = ['Month', 'DayofWeek', 'Hour']
    dfG = df['Global_active_power'].resample('60T').mean().to_frame()
    index = pd.date_range(dfG.index[-1], periods=number, freq='60T') 
    df_unseen = pd.Series(np.zeros(number), index=index, name='Global_active_power').to_frame()
    dfT = pd.concat([dfG, df_unseen], axis=0)
    dfT1 = date_transform(dfT, encode_cols) 

    val = dfT1.loc[dfG.index[-number:]].iloc[:-1]
    val_X = val.drop('Global_active_power', axis=1)
    val_y = val['Global_active_power']
    test = dfT1.loc[index[0]:].iloc[1:, 1:]
    pred = model.predict(val_X)
    forecasts = model.predict(test)
    
    st.subheader(f"The predicted consumption for the next {number} hours") 

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=val_y.index, y=val_y.values, name='Actual',))
    fig.add_trace(go.Scatter(x=val_y.index, y=pred, name='Predicted'))
    fig.add_trace(go.Scatter(x=test.index, y=forecasts, name='forecast'))
    st.write(fig)


    add_selectbox = st.sidebar.selectbox("Select the type of Detector",
        ("Outliers", "Shift Detection", "Volatility Detection", "Seasonal Detection"))
    st.header("Anomaly Detection with ADTK")
    selectF = st.selectbox('Select target', [col for col in df.columns])
    if add_selectbox == "Outliers": 
      st.subheader('Outliers Detection')
      st.markdown("This method detects outliers within "
                   "low and high range")
      values = st.slider('Select a range of values',
        df[selectF].min(), df[selectF].max(), (0.1, 10.0))
      fig = plt.figure(figsize=(14, 16))
      ax = fig.add_subplot(1,1,1)
      s_train, threshold_ad, anomalies, outlier_frac = adtk_thres(df, selectF, values[0], values[1])
      st.pyplot(health_bar(outlier_frac))
      st.pyplot(adtk_thres_plot(s_train, threshold_ad, anomalies, outlier_frac))
          
    elif add_selectbox == "Shift Detection": 
      st.subheader('Shift Detection')
      st.markdown("This method detects shift of value level by tracking "
            "the difference between median values at two sliding time windows next "
            "to each other ")
      s_train, anomalies, outlier_frac = aggFreq(df, selectF, adtk_shift, selectAg)
      st.pyplot(health_bar(outlier_frac))
      st.pyplot(adtk_shift_plot(s_train, anomalies))

    elif add_selectbox == "Volatility Detection": 
      st.subheader('Volatility Detection')
      st.markdown("This method detects shift of value level by tracking "
            "the difference between median values at two sliding time windows next "
            "to each other ")
      s_train, anomalies, outlier_frac = aggFreq(df, selectF, adtk_vol, selectAg)
      st.pyplot(health_bar(outlier_frac))
      st.pyplot(adtk_shift_plot(s_train, anomalies))
      
    else:
      st.subheader(add_selectbox)
      st.markdown("This method detects anomalous violations of seasonal pattern ")
      
      s_train, anomalies, outlier_frac = aggFreq(df, selectF, adtk_season, selectAg)
      st.pyplot(health_bar(outlier_frac))
      st.pyplot(adtk_season_plot(s_train, anomalies))
  


if __name__ == "__main__":
    write()      
