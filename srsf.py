# SRSF : Simple Retail Sales Forecasting

# print('Hey!') # goes to console

import streamlit as st

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder #LabelEncoder

from prophet import Prophet

#from datetime
import datetime

st.set_page_config(
    page_title="RSF App Start Page",
    #page_icon="👋",
    layout="wide",
)

st.markdown("# SRSF")
st.write("Simple Retail Sales Forecaster")

train_df = pd.read_csv(str(Path(__file__).parent) + '/' + 'TRAIN.csv', parse_dates=['Date']) # , index_col='Date' for time series
test_df = pd.read_csv(str(Path(__file__).parent) + '/' + 'TEST_FINAL.csv', parse_dates=['Date'])

store_nbr = train_df['Store_id'].nunique() # 365 magasins train
store_types = train_df['Store_Type'].value_counts()
location_types = train_df['Location_Type'].value_counts()
region_codes = train_df['Region_Code'].value_counts()

# make some useful features (complements Date time serie)
train_df['Year'] = train_df['Date'].apply( lambda d : d.year )
train_df['Month'] = train_df['Date'].apply( lambda d : d.month )
train_df['Day'] = train_df['Date'].apply( lambda d : d.day )
train_df['DoW'] = train_df['Date'].apply( lambda d : d.day_of_week )

train_df['Discount'] = train_df['Discount'].map({'No':0, 'Yes':1})

week_days = train_df['DoW'].value_counts()
oli_days = train_df['Holiday'].value_counts()
discount_days = train_df['Discount'].value_counts()

sales_desc = train_df['Sales'].describe()
noamount_sales_count = train_df[ train_df['Sales'] <= 0 ].shape[0]

p_df = train_df.rename(columns={'Date': 'ds', 'Sales': 'y'})
#p_df = p_df.drop(['ID','#Order'], axis=1)

pct = ColumnTransformer(transformers=[
                            #(name, trans, columns),
                            ('my_drops', 'drop', ['ID', '#Order']), # storeID ?
                            ('my_cats', OrdinalEncoder(categories='auto', dtype=np.int64, \
                                handle_unknown='error'), \
                                    ['Store_Type', 'Location_Type', 'Region_Code', 'Discount']),
                            #('my_scaler', MinMaxScaler( feature_range=(0,1) ), ['Sales']), # y
                        ],
                        remainder='passthrough', # not 'drop'
                        sparse_threshold=0.3,
                        n_jobs=None, # "None means 1 unless..."
                        transformer_weights=None, # x1
                        verbose=True, # default False
                        verbose_feature_names_out=False, # default True
                        )

pct.fit(p_df)
preprocessed_df = pct.transform(p_df)

pro_df = pd.DataFrame(preprocessed_df, columns=pct.get_feature_names_out())

the_cols = pro_df.columns.to_list()
the_cols.remove('ds')
the_cols.remove('y')
new_col_types = { col: int for col in the_cols }
new_col_types['y'] = np.float64

pro_df = pro_df.astype(new_col_types)

prophet_model = Prophet()

#prophet_model.add_regressor('Store_id') # TODO codes, types, discount, holiday, DoW, etc...
#need to be in the "future" df also => need copy test.csv

prophet_model.fit(pro_df) # 4 minutes

future = prophet_model.make_future_dataframe(periods=30+31) # june & july

forecast = prophet_model.predict(future) # 2s

tab1, tab2, tab3 = st.tabs(["Welcome", "Input Data", "Prediction"])

with tab1:
    st.header("Welcome")
    st.write('SRSF is the Simple Retail Sales Forecaster')
    st.write('Using a Kaggle DataSet, but another data source would fit.')
    st.write('Given past sales...', train_df['Date'].min(), train_df['Date'].max()) # 12+5 = 17 elapsed months
    st.write(train_df.head(n=4))
    st.write('...SRSF should forecast the next 2 months.', test_df['Date'].min(), test_df['Date'].max())
    st.write(test_df.head(n=4))

with tab2:
    st.header("Input Data")
    st.write('Some insights on past sales')
    st.markdown("- 12+5=17 months of data.")
    st.line_chart(data = train_df, x='Date', y='Sales')
    st.markdown("- Somewhat precleaned DS : no holes, no duplicates, almost balanced...")
    st.markdown("- " + str(store_nbr) + " stores (not to be confused with days in year)")
    st.markdown("- Categorial and Y/N features: Store_Type, Location_Type, Region_Code, Discount, Holiday. But opaque types and codes.")
    st.write(store_types)
    st.bar_chart(data=store_types, width=10)
    st.write(location_types)
    st.bar_chart(data=location_types, width=10)
    st.write(region_codes)
    st.bar_chart(data=region_codes, width=10)
    st.write(oli_days)
    st.bar_chart(data=oli_days, width=10)
    st.write(discount_days)
    st.bar_chart(data=discount_days, width=10)
    st.markdown("- Sales happen every week day from Monday to Sunday")
    st.write(week_days)
    st.bar_chart(data=week_days, width=10)
    st.markdown("- the target (Sales) distribution")
    st.write(sales_desc)
    st.write("There are only " + str(noamount_sales_count) + " sales with 0 as amount.")
    st.write("There is a short list of stores doing much better than average.")

with tab3:
    st.header("Prediction")
    st.write('SRSF in action')

    st.write(p_df.columns)
    st.write(pct.get_feature_names_out())

    st.write(forecast.tail(n=4))
    st.line_chart(data = forecast, x='ds', y='yhat')

    st.date_input(label="Date for Sales forecast ?", \
        key='my_date', value=datetime.date(2019, 7, 1), \
        min_value=datetime.date(2019, 6, 1), max_value=datetime.date(2019, 7, 31) )

    # st.slider(label="Store id for Sales forecast ?", \
    #     key='my_store', value=252, \
    #     min_value=0, max_value=365 )

    if ('my_date' in st.session_state): # and ('my_store' in st.session_state):
        st.write("Choosen date: " + str(st.session_state['my_date']) )
        #st.write("Choosen store: " + str(st.session_state['my_store']) )

        #f_val = forecast[ (forecast['ds'].dt.date == st.session_state['my_date']) & (forecast['Store_id'] == st.session_state['my_store']) ]['yhat']
        f_val = forecast[ (forecast['ds'].dt.date == st.session_state['my_date']) ]['yhat']
        st.write("Forecast for " + str(st.session_state['my_date']) + ": ")
        st.write(f_val)
        #st.write("for store id=" + str(st.session_state['my_store']))
