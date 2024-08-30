import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import json

# Load the trained model
model = pickle.load(open('RFCModelNew.pkl', 'rb'))

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction (CSV)" , "About What We Predict"])

# additionnal css
st.markdown(
    """
    <style>

    /* Custom CSS for table styling */
    .custom-table {
        width: 100%;
        margin: 20px 0;
        font-size: 16px;
        text-align: left;


    }
    .custom-table th, .custom-table td {
        border: 1px solid #ddd;
        padding: 8px;

    }
    .custom-table th {
        font-weight: bold;
    }
    .custom-table tr:nth-child(even) {
    }
    .custom-table tr:hover {
        background-color: DarkGrey;
    }

    </style>
    """,
    unsafe_allow_html=True
)

#choose the page
if page == "Single Prediction":

    st.title('Credit Risk Prediction')

    # Sidebar for user input
    st.sidebar.header('Input Parameters To Predict')
    # numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled

    def user_input_features():
        age = st.sidebar.slider('Age', 18, 100, 30)
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        job = st.sidebar.slider('Job Rating', 0, 3, 1)
        housing = st.sidebar.selectbox('Housing', ('own', 'free', 'rent'))
        saving_accounts = st.sidebar.selectbox('Saving accounts', ('little', 'moderate', 'rich', 'quite rich'))
        checking_account = st.sidebar.selectbox('Checking account', ('little', 'moderate', 'rich'))
        credit_amount = st.sidebar.slider('Credit amount', 0, 20000, 5000)
        duration = st.sidebar.slider('Duration', 0, 72, 24)
        purpose = st.sidebar.selectbox('Purpose', ('car', 'furniture/equipment', 'radio/TV', 'domestic appliances', 'repairs', 'education', 'vacation/others', 'business'))

        # Original input data without dummies
        input_data = {
            'Age': age,
            'Sex': sex,
            'Job': job,
            'Housing': housing,
            'Saving accounts': saving_accounts,
            'Checking account': checking_account,
            'Credit amount': credit_amount,
            'Duration': duration,
            'Purpose': purpose
        }

        data = {
            'Age': age,
            'Job': job,
            'Credit amount': credit_amount,
            'Duration': duration,
            'Sex_male': 1 if sex == 'male' else 0,
            'Housing_free': 1 if housing == 'free' else 0,
            'Housing_own': 1 if housing == 'own' else 0,
            'Housing_rent': 1 if housing == 'rent' else 0,
            'Saving accounts_little': 1 if saving_accounts == 'little' else 0,
            'Saving accounts_moderate': 1 if saving_accounts == 'moderate' else 0,
            'Saving accounts_quite rich': 1 if saving_accounts == 'quite rich' else 0,
            'Saving accounts_rich': 1 if saving_accounts == 'rich' else 0,
            'Checking account_little': 1 if checking_account == 'little' else 0,
            'Checking account_moderate': 1 if checking_account == 'moderate' else 0,
            'Checking account_rich': 1 if checking_account == 'rich' else 0,
            'Purpose_business': 1 if purpose == 'business' else 0,
            'Purpose_car': 1 if purpose == 'car' else 0,
            'Purpose_domestic appliances': 1 if purpose == 'domestic appliances' else 0,
            'Purpose_education': 1 if purpose == 'education' else 0,
            'Purpose_furniture/equipment': 1 if purpose == 'furniture/equipment' else 0,
            'Purpose_radio/TV': 1 if purpose == 'radio/TV' else 0,
            'Purpose_repairs': 1 if purpose == 'repairs' else 0,
            'Purpose_vacation/others': 1 if purpose == 'vacation/others' else 0
        }
        features = pd.DataFrame(data, index=[0])
        original_features = pd.DataFrame(input_data, index=[0])
        return features,original_features
 
    input_df , original_features= user_input_features()

    # Main panel
    st.subheader('User Input parameters')
    # st.write(original_features)
    # Convert DataFrame to HTML with custom CSS class
    original_features_html = original_features.to_html(classes='custom-table', index=False)

    # Display the styled table
    st.markdown(original_features_html, unsafe_allow_html=True)

    # Make predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    risk_label = np.array(['bad', 'good'])

    # Display prediction with thumbs-up or thumbs-down
    if prediction[0] == 1:
        st.markdown('<div class="thumbs">👍</div>', unsafe_allow_html=True)
        st.write('**Result:** Good')
    else:
        st.markdown('<div class="thumbs">👎</div>', unsafe_allow_html=True)
        st.write('**Result:** Bad')

    st.subheader('Prediction Probability')
    # Convert the prediction probabilities to 'good' or 'bad'
    st.write(f"Probability of Good Risk: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Bad Risk: {prediction_proba[0][0]:.2f}")

elif page == "Batch Prediction (CSV)":
    st.title('Credit Risk Prediction - Batch Entry via CSV')

    st.subheader('Upload CSV File for Batch Prediction')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        input_df = pd.read_csv(uploaded_file , index_col=0)

        # Display the input dataframe
        st.subheader('Uploaded CSV Data')
        st.write(input_df)

        # Convert categorical variables
        input_df = pd.get_dummies(input_df, columns=['Sex'] , drop_first= True)
        input_df = pd.get_dummies(input_df, columns=['Housing', 'Saving accounts', 'Checking account', 'Purpose'])

        # Make predictions
        predictions = model.predict(input_df)

        #reverting the dummies :
        # Reverting 'Sex' column
        input_df['Sex'] = input_df['Sex_male'].apply(lambda x: 'male' if x == 1 else 'female')
        input_df.drop(['Sex_male'], axis=1, inplace=True)

        # Reverting 'Housing' column
        input_df['Housing'] = input_df[['Housing_free', 'Housing_own', 'Housing_rent']].idxmax(axis=1)
        input_df['Housing'] = input_df['Housing'].map({
            'Housing_free': 'free',
            'Housing_own': 'own',
            'Housing_rent': 'rent'
        })
        input_df.drop(['Housing_free', 'Housing_own', 'Housing_rent'], axis=1, inplace=True)

        # Reverting 'Purpose' column
        input_df['Purpose'] = input_df[['Purpose_car', 'Purpose_furniture/equipment', 'Purpose_radio/TV', 
                            'Purpose_domestic appliances', 'Purpose_repairs', 'Purpose_education', 
                            'Purpose_vacation/others', 'Purpose_business']].idxmax(axis=1)

        # Mapping dummy column names back to the original categories
        input_df['Purpose'] = input_df['Purpose'].map({
            'Purpose_car': 'car',
            'Purpose_furniture/equipment': 'furniture/equipment',
            'Purpose_radio/TV': 'radio/TV',
            'Purpose_domestic appliances': 'domestic appliances',
            'Purpose_repairs': 'repairs',
            'Purpose_education': 'education',
            'Purpose_vacation/others': 'vacation/others',
            'Purpose_business': 'business'
        })

        # Dropping the dummy columns for 'Purpose'
        input_df.drop(['Purpose_car', 'Purpose_furniture/equipment', 'Purpose_radio/TV', 
                'Purpose_domestic appliances', 'Purpose_repairs', 'Purpose_education', 
                'Purpose_vacation/others', 'Purpose_business'], axis=1, inplace=True)

        # Reverting 'Saving accounts' column
        input_df['Saving accounts'] = input_df[['Saving accounts_little', 'Saving accounts_moderate', 'Saving accounts_rich', 'Saving accounts_quite rich']].idxmax(axis=1)
        input_df['Saving accounts'] = input_df['Saving accounts'].map({
            'Saving accounts_little': 'little',
            'Saving accounts_moderate': 'moderate',
            'Saving accounts_rich': 'rich',
            'Saving accounts_quite rich': 'quite rich'
        })
        input_df.drop(['Saving accounts_little', 'Saving accounts_moderate', 'Saving accounts_rich', 'Saving accounts_quite rich'], axis=1, inplace=True)

        # Reverting 'Checking account' column
        input_df['Checking account'] = input_df[['Checking account_little', 'Checking account_moderate', 'Checking account_rich']].idxmax(axis=1)
        input_df['Checking account'] = input_df['Checking account'].map({
            'Checking account_little': 'little',
            'Checking account_moderate': 'moderate',
            'Checking account_rich': 'rich'
        })
        input_df.drop(['Checking account_little', 'Checking account_moderate', 'Checking account_rich'], axis=1, inplace=True)

        #Adding the prediction column
        input_df['Risk Prediction'] = np.where(predictions == 1, 'good', 'bad')


        st.subheader('Batch Prediction Results')
        st.write(input_df)

        # Option to download the updated CSV file
        st.download_button(
            label="Download data with predictions as CSV",
            data=input_df.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv'
        )

elif page == "About What We Predict":
        st.title('Credit Risk Prediction with AI')
        st.write("""
    This application uses a machine learning model designed to predict the likelihood of a customer defaulting on a loan. 
    This model takes into account various features such as age, sex, job rating, housing situation, and more to assess the risk. 
    The model's output helps in making informed decisions about lending and managing financial risks.
    The prediction is categorized as 'Good' or 'Bad' based on the likelihood of default, with probabilities provided to quantify the risk.
    """)
