# SRSF : Simple Retail Sales Forecasting

# print('Hey!') # goes to console

import streamlit as st

st.set_page_config(
    page_title="RSF App Start Page",
    #page_icon="👋",
    #layout="wide"
)

st.markdown("# SRSF")
st.write("Simple Retail Sales Forecaster")

tab1, tab2, tab3 = st.tabs(["Welcome", "Input Data", "Prediction"])

with tab1:
    st.header("Welcome")
    st.write('SRSF is the Simple Retail Sales Forecaster')

with tab2:
    st.header("Input Data")
    st.write('Some insights on past sales')

#    st.write(df2)
#    st.write(df2['second column'].describe())

#    st.line_chart(df2)

with tab3:
    st.header("Prediction")
    st.write('SRSF in action')
