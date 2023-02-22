from pycaret.regression import *

import streamlit as st

import pandas as pd

def write():
    """Writes content to the app"""

    st.title("Predict house sales prices\n")
    st.markdown("This application can be used to predict "
                    " the price of a house🏠")

    st.write('''Previsione del costo di una casa usando come variabili:   
               - L'area del lotto    
               - L'anno di costruzione      
               - Numero di camere da letto     
               - Numero di cucine   
               ''')

    features = ['LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr']
    train = pd.read_csv('../data/train.csv', index_col=0)
    train1 = train[features]
    #st.write(train1.columns)
    #st.write(train1.head())
    st.markdown("#### Compila i campi")

    col1, col2 = st.columns(2)

    la = col1.number_input('Area del lotto', min_value = 1300, value = 9600, max_value = 215245)
    yeb = col1.number_input('Anno di costruzione', min_value = 1872, value = 1976, max_value = 2010)
    bed = col2.number_input("Numero di camere da letto", min_value = 0, value = 3, max_value = 8)
    kit = col2.number_input("Numero di cucine", min_value = 0, value = 1, max_value = 3)

    test = pd.DataFrame([[la, yeb, bed, kit]], columns = features)

    # load pipeline
    lm = load_model('../models/gb_pipeline_House')

    if st.button('Calcola'):
        st.subheader(f"**Prezzo stimato**")
        res = predict_model(lm, data = test)['Label']
        st.write(f"Il prezzo stimato di questa casa è: {round(res.values[0],2)}$")



if __name__ == "__main__":
    write()           

