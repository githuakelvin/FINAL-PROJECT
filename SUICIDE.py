import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib # For saving/loading models and preprocessors
import os
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings

# Create the 'models' directory if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")

# ... (rest of your code, including the joblib.dump calls) ...
# --- 1. Data Loading and Simulation (More Realistic Data) ---
# we'll simulate a more complex dataset to demonstrate improved functionalities.


@st.cache_data
def load_and_simulate_data():
    # Simulate a more complex dataset for demonstration purposes
    np.random.seed(42)
    num_entries = 500
    
    countries = ['KENYA', 'TANZANIA', 'UGANDA', 'RWANDA', 'ETHIOPIA', 'BURUNDI', 'SOUTH SUDAN']
    sexes = ['Male', 'Female', 'Both sexes'] 
    age_groups = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    education_levels = ['Primary', 'Secondary', 'Tertiary', 'None']
    employment_statuses = ['Employed', 'Unemployed', 'Student', 'Retired']
    urban_rural = ['Urban', 'Rural']
    
    data = {
        'Country': np.random.choice(countries, num_entries),
        'Year': np.random.randint(2010, 2025, num_entries),
        'Sex': np.random.choice(sexes, num_entries, p=[0.45, 0.45, 0.1]), # More males/females, fewer 'both'
        'Age_Group': np.random.choice(age_groups, num_entries, p=[0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]),
        'GDP_Per_Capita': np.random.normal(1500, 700, num_entries).clip(500, 5000), # Simulating economic factor
        'Unemployment_Rate': np.random.normal(0.08, 0.03, num_entries).clip(0.02, 0.20), # Simulating economic factor
        'Mental_Health_Spend_Per_Capita': np.random.normal(5, 3, num_entries).clip(1, 15), # Simulating health system factor
        'Education_Level': np.random.choice(education_levels, num_entries, p=[0.25, 0.4, 0.2, 0.15]),
        'Employment_Status': np.random.choice(employment_statuses, num_entries),
        'Urban_Rural': np.random.choice(urban_rural, num_entries),
        'Suicide_Rate': np.random.normal(8, 5, num_entries).clip(1, 30) # Simulating target variable
    }
    
    df = pd.DataFrame(data)

    # Introduce some correlations for a more realistic dataset
    # Higher unemployment -> higher suicide rates
    df['Suicide_Rate'] = df.apply(lambda row: row['Suicide_Rate'] + (row['Unemployment_Rate'] * 50), axis=1)
    # Lower GDP -> higher suicide rates
    df['Suicide_Rate'] = df.apply(lambda row: row['Suicide_Rate'] + (5000 - row['GDP_Per_Capita']) / 500, axis=1)
    # Lower mental health spend -> higher suicide rates
    df['Suicide_Rate'] = df.apply(lambda row: row['Suicide_Rate'] + (20 - row['Mental_Health_Spend_Per_Capita']) * 0.5, axis=1)

    # Add specific nuances for Kenya if needed (e.g., slightly higher average rate)
    df.loc[df['Country'] == 'KENYA', 'Suicide_Rate'] = df.loc[df['Country'] == 'KENYA', 'Suicide_Rate'] * 1.2
    
    df['Suicide_Rate'] = df['Suicide_Rate'].clip(1, 30).round(1) # Ensure rates are within a reasonable range

    return df

df = load_and_simulate_data()

# Clean column names (already done by pandas if creating from dict, but good practice)
df.columns = df.columns.str.strip()

# --- 2. Data Preprocessing and Feature Engineering (Modularized) ---

# Define target variable - still using binary for simplicity, but could be regression
@st.cache_data
def create_target(dataframe): 
    median_rate = dataframe['Suicide_Rate'].median()
    dataframe['High_Risk'] = (dataframe['Suicide_Rate'] > median_rate).astype(int)
    return dataframe

df = create_target(df.copy()) # Use .copy() to avoid SettingWithCopyWarning

@st.cache_resource # Cache the preprocessors
def get_preprocessors(df_input):
    numerical_cols = ['GDP_Per_Capita', 'Unemployment_Rate', 'Mental_Health_Spend_Per_Capita']
    categorical_cols = ['Country', 'Sex', 'Age_Group', 'Education_Level', 'Employment_Status', 'Urban_Rural']

    # Imputer for numerical features (if any NaNs expected)
    # Our simulated data has no NaNs, but this is for robustness
    numerical_imputer = SimpleImputer(strategy='median')
    
    # Scaler for numerical features
    scaler = StandardScaler()
    
    # One-hot encoder for categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for dense array

    # Fit preprocessors
    numerical_imputer.fit(df_input[numerical_cols])
    scaler.fit(df_input[numerical_cols])
    encoder.fit(df_input[categorical_cols])

    return numerical_imputer, scaler, encoder, numerical_cols, categorical_cols

numerical_imputer, scaler, encoder, numerical_cols, categorical_cols = get_preprocessors(df)

def preprocess_features(df_to_process, numerical_imputer, scaler, encoder, numerical_cols, categorical_cols):
    df_processed = df_to_process.copy()

    # Handle missing numerical data
    df_processed[numerical_cols] = numerical_imputer.transform(df_processed[numerical_cols])

    # Scale numerical features
    df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])

    # Encode categorical features
    encoded_features = encoder.transform(df_processed[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Drop original categorical columns and concatenate encoded ones
    df_processed = pd.concat([df_processed.drop(columns=categorical_cols), encoded_df], axis=1)
    
    return df_processed

# Apply preprocessing
df_processed = preprocess_features(df.drop(columns=['Suicide_Rate', 'High_Risk']),
   numerical_imputer, scaler, encoder, numerical_cols, categorical_cols)

# Prepare data for modeling
X = df_processed
y = df['High_Risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify for binary target

# --- 3. Model Training and Evaluation (Improved) ---

@st.cache_resource # Cache the trained model
def train_and_evaluate_model(X_train_data, y_train_data, X_test_data, y_test_data):
    st.subheader("Model Training & Evaluation")
    st.write("Training a RandomForestClassifier for suicide rate in countries.")

    # Model Selection & Hyperparameter Tuning (Example with GridSearchCV)
    # For a real project, you'd iterate through more models and parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 2, 4]
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_train_data)
    
    best_model = grid_search.best_estimator_
    st.write(f"Best Model Parameters: {grid_search.best_params_}")

    # Evaluate the best model
    y_pred = best_model.predict(X_test_data)
    y_proba = best_model.predict_proba(X_test_data)[:, 1] # Probability of High_Risk (class 1)

    st.write("### Model Performance Report (on Test Set):")
    st.text(classification_report(y_test_data, y_pred))

    st.write(f"Accuracy: {accuracy_score(y_test_data, y_pred):.2f}")
    st.write(f"Precision (High Risk): {precision_score(y_test_data, y_pred):.2f}")
    st.write(f"Recall (High Risk): {recall_score(y_test_data, y_pred):.2f}")
    st.write(f"F1-Score (High Risk): {f1_score(y_test_data, y_pred):.2f}")
    st.write(f"ROC-AUC Score: {roc_auc_score(y_test_data, y_proba):.2f}")

    # Feature Importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_train_data.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.write("### Feature Importance:")
        st.dataframe(feature_importance.head(10))

        fig_feature_imp = px.bar(feature_importance.head(10), x='Importance', y='Feature', orientation='h',
                                 title='Top 10 Feature Importances',
                                 labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'})
        fig_feature_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_feature_imp, use_container_width=True)

    return best_model

# Run model training and evaluation
model = train_and_evaluate_model(X_train, y_train, X_test, y_test)

# --- Save Model and Preprocessors (for later prediction in Streamlit) ---
joblib.dump(model, 'models/trained_rf_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')
joblib.dump(numerical_imputer, 'models/numerical_imputer.pkl')

# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="East Africa Suicide Risk Dashboard", page_icon="❤️‍🩹")

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: green;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
        text-color: red;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: pink ;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    <style>
    h1, h2, h3, h4 { color: #2c3e50; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .sidebar .sidebar-content { background-color: #f7f9fb; }
</style>

    
    """, unsafe_allow_html=True)

st.title("❤️‍🩹 East Africa Mental Health & Suicide Risk")
st.markdown("A data-driven exploration and prediction tool for suicide risk factors.")
menu = st.sidebar.radio("Navigation", ["Overview", "Insights", "Prediction", "About"])
if menu == "Overview":
    ...

col1, col2, col3 = st.columns(3)
col1.metric("Countries", df['Country'].nunique())
col2.metric("Avg Suicide Rate", f"{df['Suicide_Rate'].mean():.1f}/100k")
col3.metric("High Risk %", f"{df['High_Risk'].mean() * 100:.1f}%")


# --- Tabbed Interface for better organization ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview & Trends", "Country Insights", "Risk Prediction", "About"])

with tab1:
    st.header("Overview: Suicide Rates Across East Africa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<p class='medium-font'>Suicide Rates by Age Group and Sex</p>", unsafe_allow_html=True)
        fig_age_sex = px.bar(df, x='Age_Group', y='Suicide_Rate', color='Sex',
                             title='Average Suicide Rates by Age Group and Sex',
                             labels={'Suicide_Rate': 'Suicide Rate (per 100k)', 'Age_Group': 'Age Group'},
                             barmode='group', height=400)
        st.plotly_chart(fig_age_sex, use_container_width=True)
        
    with col2:
        st.markdown("<p class='medium-font'>Distribution of Suicide Rates</p>", unsafe_allow_html=True)
        fig_hist = px.histogram(df, x='Suicide_Rate', nbins=20,
                                title='Distribution of Simulated Suicide Rates',
                                labels={'Suicide_Rate': 'Suicide Rate (per 100k)', 'count': 'Number of Data Points'},
                                height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<p class='medium-font'>Suicide Rates by Country</p>", unsafe_allow_html=True)
    avg_suicide_by_country = df.groupby('Country')['Suicide_Rate'].mean().reset_index().sort_values('Suicide_Rate', ascending=False)
    fig_country_avg = px.bar(avg_suicide_by_country, x='Country', y='Suicide_Rate',
                             title='Average Suicide Rate by Country',
                             labels={'Suicide_Rate': 'Average Suicide Rate (per 100k)', 'Country': 'Country'},
                             color='Country', height=450)
    st.plotly_chart(fig_country_avg, use_container_width=True)

    st.markdown("<p class='medium-font'>Correlation Heatmap of Numerical Factors</p>", unsafe_allow_html=True)
    # Ensure only numerical columns for correlation
    numerical_data_for_corr = df[numerical_cols + ['Suicide_Rate']]
    corr_matrix = numerical_data_for_corr.corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         color_continuous_scale=px.colors.sequential.RdBu,
                         title='Correlation Matrix of Numerical Factors', height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.header("Deep Dive: Country-Specific Insights")
    
    selected_country_insight = st.selectbox("Select Country for Detailed Analysis", df['Country'].unique(), key='country_insight_select')
    country_data_insight = df[df['Country'] == selected_country_insight]

    if not country_data_insight.empty:
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"<p class='medium-font'>Suicide Rates in {selected_country_insight} by Age and Sex</p>", unsafe_allow_html=True)
            fig_country_age_sex = px.bar(country_data_insight, x='Age_Group', y='Suicide_Rate', color='Sex',
                                         title=f'Suicide Rates in {selected_country_insight} by Age and Sex',
                                         labels={'Suicide_Rate': 'Suicide Rate (per 100k)', 'Age_Group': 'Age Group'},
                                         barmode='group', height=400)
            st.plotly_chart(fig_country_age_sex, use_container_width=True)
        
        with col4:
            st.markdown(f"<p class='medium-font'>Impact of Economic Factors in {selected_country_insight}</p>", unsafe_allow_html=True)
            fig_economic = px.scatter(country_data_insight, x='GDP_Per_Capita', y='Suicide_Rate', color='Unemployment_Rate',
                                      size='Suicide_Rate', hover_name='Age_Group',
                                      title=f'Suicide Rate vs. Economic Factors in {selected_country_insight}',
                                      labels={'GDP_Per_Capita': 'GDP Per Capita (USD)', 'Unemployment_Rate': 'Unemployment Rate (%)'},
                                      height=400)
            st.plotly_chart(fig_economic, use_container_width=True)
        
        st.markdown(f"<p class='medium-font'>Suicide Rate Trends Over Years in {selected_country_insight}</p>", unsafe_allow_html=True)
        country_trend_data = country_data_insight.groupby('Year')['Suicide_Rate'].mean().reset_index()
        fig_trend = px.line(country_trend_data, x='Year', y='Suicide_Rate',
                            title=f'Average Suicide Rate Trend in {selected_country_insight}',
                            labels={'Suicide_Rate': 'Average Suicide Rate (per 100k)'},
                            markers=True, height=400)
        st.plotly_chart(fig_trend, use_container_width=True)

    else:
        st.warning("No data available for the selected country.")

with tab3:
    st.header("Predict Suicide Risk for a Specific Profile")
    st.markdown("Enter the demographic and socioeconomic details to get a risk prediction.")

    # Custom styling for input fields to make them more visually appealing
    with st.expander("Enter Profile Details", expanded=True):
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        with col_pred1:
            pred_country = st.selectbox("Country", df['Country'].unique(), key='pred_country')
            pred_sex = st.selectbox("Sex", df['Sex'].unique(), key='pred_sex')
        with col_pred2:
            pred_age = st.selectbox("Age Group", df['Age_Group'].unique(), key='pred_age')
            pred_education = st.selectbox("Education Level", df['Education_Level'].unique(), key='pred_education')
        with col_pred3:
            pred_employment = st.selectbox("Employment Status", df['Employment_Status'].unique(), key='pred_employment')
            pred_urban_rural = st.selectbox("Urban/Rural", df['Urban_Rural'].unique(), key='pred_urban_rural')

        st.markdown("---")
        st.markdown("<p class='medium-font'>Socioeconomic Indicators (Adjust these for prediction)</p>", unsafe_allow_html=True)
        col_socio1, col_socio2, col_socio3 = st.columns(3)
        with col_socio1:
            # Provide sensible default ranges based on your simulated data
            pred_gdp = st.slider("GDP Per Capita (USD)", min_value=500, max_value=5000, value=int(df['GDP_Per_Capita'].mean()), step=100)
        with col_socio2:
            pred_unemployment = st.slider("Unemployment Rate (%)", min_value=0.02, max_value=0.20, value=round(df['Unemployment_Rate'].mean(), 2), step=0.01, format="%.2f")
        with col_socio3:
            pred_mh_spend = st.slider("Mental Health Spend Per Capita (USD)", min_value=1, max_value=15, value=int(df['Mental_Health_Spend_Per_Capita'].mean()), step=1)
        
        # Add a year input, though it's not used in this simplified model, it could be for time-series models
        pred_year = st.number_input("Year for Prediction (Optional, for context)", min_value=2010, max_value=2030, value=2024, step=1)

    if st.button("Predict Risk Level", key='predict_button'):
        # Create input features DataFrame for prediction
        input_data_raw = pd.DataFrame([{
            'Country': pred_country,
            'Year': pred_year, # Model might not use Year directly, but good for consistent input structure
            'Sex': pred_sex,
            'Age_Group': pred_age,
            'GDP_Per_Capita': float(pred_gdp),
            'Unemployment_Rate': float(pred_unemployment),
            'Mental_Health_Spend_Per_Capita': float(pred_mh_spend),
            'Education_Level': pred_education,
            'Employment_Status': pred_employment,
            'Urban_Rural': pred_urban_rural
        }])

        try:
            # Preprocess the input data using the trained preprocessors
            input_data_processed = preprocess_features(input_data_raw.drop(columns=['Year']), # Drop Year if not a feature
                                                       numerical_imputer, scaler, encoder, numerical_cols, categorical_cols)
            
            # Align columns: Crucial for robust prediction in production
            # The model expects features in the same order as during training.
            # Create a full DataFrame with all possible feature columns (from X_train) filled with 0,
            # then fill in the values from input_data_processed.
            
            # Get all columns that the model was trained on
            trained_columns = X_train.columns
            
            # Create a new DataFrame for prediction with all trained columns, initialized to 0
            final_input_df = pd.DataFrame(0, index=[0], columns=trained_columns)
            
            # Fill in the values for the columns present in input_data_processed
            for col in input_data_processed.columns:
                if col in final_input_df.columns:
                    final_input_df[col] = input_data_processed[col].iloc[0]
                else:
                    st.warning(f"Warning: Feature '{col}' from input not found in trained model features.")

            # Make prediction
            prediction = model.predict(final_input_df)
            probability = model.predict_proba(final_input_df)[:, 1] # Probability of High_Risk

            st.markdown("---")
            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.markdown(f"**<span style='color:red; font-size:24px;'>High Risk of Suicide Rate</span>**", unsafe_allow_html=True)
                st.markdown(f"Probability of High Risk: **<span style='color:red; font-size:20px;'>{probability[0]*100:.1f}%</span>**", unsafe_allow_html=True)
            else:
                st.markdown(f"**<span style='color:green; font-size:24px;'>Low Risk of Suicide Rate</span>**", unsafe_allow_html=True)
                st.markdown(f"Probability of High Risk: **<span style='color:green; font-size:20px;'>{probability[0]*100:.1f}%</span>**", unsafe_allow_html=True)
            
            st.write("---")
            st.subheader("Contextual Information:")
            # Show actual data for context
            context_data = df[
                (df['Country'] == pred_country) &
                (df['Sex'] == pred_sex) &
                (df['Age_Group'] == pred_age)
            ]
            
            if not context_data.empty:
                avg_actual_rate = context_data['Suicide_Rate'].mean()
                st.info(f"Average suicide rate for {pred_sex}s aged {pred_age} in {pred_country} (historical data): **{avg_actual_rate:.1f} per 100k**")
                
                # Show distribution of Suicide_Rate for the specific country-sex-age group
                fig_context_dist = px.histogram(context_data, x='Suicide_Rate', title=f'Distribution of Suicide Rates for {pred_sex}s aged {pred_age} in {pred_country}',
                                                labels={'Suicide_Rate': 'Suicide Rate (per 100k)'})
                st.plotly_chart(fig_context_dist, use_container_width=True)
            else:
                st.warning("No historical data available for the exact combination of Country, Sex, and Age Group for context.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please check the input values and try again.")
            st.warning("Ensure the data types and format match the expected model input.")
import plotly.express as px
country_avg = df.groupby("Country")["Suicide_Rate"].mean().reset_index()
fig = px.choropleth(country_avg, locations="Country", locationmode="country names", 
                    color="Suicide_Rate", color_continuous_scale="Reds",
                    title="Average Suicide Rate by Country")
st.plotly_chart(fig, use_container_width=True)

import base64
def download_link(object_to_download, download_filename, link_text):
    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{link_text}</a>'
    return href
report = "Risk prediction report..."
st.markdown(download_link(report, 'report.txt', '📄 Download Report'), unsafe_allow_html=True)

with tab4:
    st.header("About This Project")
    st.markdown("""
    This interactive dashboard aims to provide insights into suicide rates and related factors in East African countries, with a particular focus on Kenya.
    
    **Project Goals:**
    - To visualize trends and patterns in suicide rates across different demographics and socioeconomic conditions.
    - To identify potential contributing factors to suicide risk using machine learning.
    - To offer a predictive tool for estimating suicide risk based on various inputs.
    - To raise awareness and support data-driven decision-making in mental health initiatives.

    **Data Sources:**
    * **Simulated Data:** For this demonstration, a synthetic dataset has been generated to illustrate the functionalities. In a real-world application, this would be replaced by actual datasets from sources like:
        * World Health Organization (WHO) suicide data
        * Kaggle datasets on global suicide statistics
        * World Bank data for socioeconomic indicators (GDP, unemployment, education) in African countries.
        * Humanitarian Data Exchange (HDX) for specific regional/country-level health data.

    **Methodology:**
    * **Data Preprocessing:** Handling missing values, scaling numerical features, and one-hot encoding categorical variables.
    * **Feature Engineering:** Creating relevant features (e.g., from combined data).
    * **Machine Learning Model:** A **Random Forest Classifier** is used to predict the likelihood of a high suicide risk, based on a binary classification (high vs. low risk relative to the median rate). Hyperparameter tuning (using GridSearchCV) is applied to optimize model performance.
    * **Visualization:** Interactive charts using `Plotly Express` are employed for clear and insightful data representation.

    **Limitations:**
    * This model is based on **simulated data** for demonstration purposes. Real-world applications require extensive, validated, and granular datasets.
    * The 'High Risk' classification is based on a **median split** of the suicide rate, which is a simplified approach. A more nuanced classification would involve clinical thresholds or expert definitions.
    * **Causation vs. Correlation:** The model identifies correlations; it does not imply direct causation.
    * **Data Availability:** Obtaining comprehensive and high-quality mental health data, especially at sub-national levels in African countries, can be challenging.
    * **Ethical Considerations:** Predictions should be interpreted with caution and **never** used for individual diagnosis or stigmatization. This tool is for **exploratory data analysis and awareness**, not clinical decision-making.

    **Future Enhancements:**
    * Integration of actual, larger datasets from diverse sources.
    * Development of a regression model to predict the exact suicide rate.
    * Inclusion of more complex features (e.g., healthcare access, poverty indices, conflict indicators).
    * Advanced natural language processing (NLP) if text-based mental health data becomes available.
    * Deployment of the model as an API for broader integration.
    """)