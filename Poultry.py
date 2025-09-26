import streamlit as st
import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus import Image, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import urllib.request
import os
import tempfile
from io import BytesIO
from PIL import Image as PILImage
from datetime import datetime
import warnings
import base64
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
warnings.filterwarnings('ignore')

# ===========================
# SESSION STATE INITIALIZATION
# ===========================
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'pred_result' not in st.session_state:
    st.session_state.pred_result = None
if 'feat_shap' not in st.session_state:
    st.session_state.feat_shap = None
if 'chart_fig' not in st.session_state:
    st.session_state.chart_fig = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Poultry ROI & Profit Predictor", 
    page_icon="🐔", 
    layout="wide"
)

st.title("🐔 Poultry Batch Profit & ROI Prediction App")
st.write("**Advanced ML predictions with feature engineering and comprehensive model evaluation.**")

# ===========================
# DATA LOADING
# ===========================
@st.cache_data
def load_data():
    """Load and preprocess the poultry dataset"""
    try:
        df = pd.read_csv("poultry_profitability_time_series_dataset.csv")
        
        # Date preprocessing
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df.sort_values(['Batch_ID', 'Date'], inplace=True)
        
        # Use latest row per Batch_ID
        latest = df.groupby('Batch_ID').tail(1).reset_index(drop=True)
        
        # Basic data cleaning
        latest = latest.select_dtypes(include=[np.number, 'object']).dropna()
        
        return latest
    except FileNotFoundError:
        st.error("Dataset file 'poultry_profitability_time_series_dataset.csv' not found!")
        st.stop()

data = load_data()

# ===========================
# FEATURE DEFINITIONS
# ===========================
BASE_FEATURES = [
    'Number_of_Birds', 'Cost_per_Chick', 'Feed_Cost_Total', 'Labor_Cost', 
    'Rent_Cost', 'Medicine_Cost', 'Land_Cost', 'Infrastructure_Cost', 
    'Equipment_Cost', 'Utilities_Cost', 'Feed_Conversion_Ratio',
    'Mortality_Rate', 'Manure_Sales', 'Age_of_Birds_at_Sale', 'Sale_Price_per_Bird'
]

CATEGORICAL_FEATURES = ['Farmer_Training_Level']
TARGETS = {'Net Profit/Loss': 'Net_Profit_Loss', 'ROI (%)': 'ROI_Percentage'}

# ===========================
# FEATURE ENGINEERING
# ===========================
@st.cache_data
def engineer_features(df):
    """Create advanced features from base data"""
    df_eng = df.copy()
    
    # Revenue calculations
    df_eng['Total_Revenue'] = df_eng['Number_of_Birds'] * df_eng['Sale_Price_per_Bird']
    df_eng['Revenue_per_Bird_Alive'] = df_eng['Total_Revenue'] / (df_eng['Number_of_Birds'] * (1 - df_eng['Mortality_Rate']/100))
    
    # Cost efficiency ratios
    df_eng['Feed_Cost_per_Bird'] = df_eng['Feed_Cost_Total'] / df_eng['Number_of_Birds']
    df_eng['Total_Fixed_Costs'] = (df_eng['Rent_Cost'] + df_eng['Land_Cost'] + 
                                  df_eng['Infrastructure_Cost'] + df_eng['Equipment_Cost'])
    df_eng['Total_Variable_Costs'] = (df_eng['Feed_Cost_Total'] + df_eng['Labor_Cost'] + 
                                     df_eng['Medicine_Cost'] + df_eng['Utilities_Cost'])
    df_eng['Total_Costs'] = df_eng['Total_Fixed_Costs'] + df_eng['Total_Variable_Costs']
    
    # Efficiency metrics
    df_eng['Cost_per_Bird'] = df_eng['Total_Costs'] / df_eng['Number_of_Birds']
    df_eng['Survival_Rate'] = 100 - df_eng['Mortality_Rate']
    df_eng['Days_to_Market'] = df_eng['Age_of_Birds_at_Sale']
    df_eng['Revenue_Cost_Ratio'] = df_eng['Total_Revenue'] / (df_eng['Total_Costs'] + 1e-8)
    
    # Production efficiency
    df_eng['Feed_Efficiency'] = 1 / (df_eng['Feed_Conversion_Ratio'] + 1e-8)
    df_eng['Profit_Margin'] = ((df_eng['Total_Revenue'] - df_eng['Total_Costs']) / 
                              (df_eng['Total_Revenue'] + 1e-8)) * 100
    
    # Scale indicators
    df_eng['Farm_Size_Category'] = pd.cut(df_eng['Number_of_Birds'], 
                                         bins=[0, 100, 500, 1000, np.inf], 
                                         labels=['Small', 'Medium', 'Large', 'Industrial'])
    
    return df_eng

# Apply feature engineering
data_engineered = engineer_features(data)

# ===========================
# SIDEBAR CONTROLS
# ===========================
with st.sidebar:
    st.header("🔧 Model Configuration")
    
    # Target selection
    target_label = st.radio("Target Variable", list(TARGETS.keys()))
    target_name = TARGETS[target_label]
    
    # Model selection
    #model_name = st.selectbox("ML Model", ["Linear Regression", "Random Forest", "XGBoost"])
    model_name ="Linear Regression"
    
    # Data split
    #test_size = st.selectbox("Test Data %", [10, 20, 30, 40, 50], index=1)
    test_size = 30
    
    # Scoring metric
    #scoring_choice = st.selectbox(
    #    "Scoring Metric",
    #    ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    #    index=0
    #)
    scoring_choice = "r2"

    # Language selection
    st.subheader("🌐 Language")
    language_choice = st.radio("Select Language:", ["English", "Marathi", "Hindi"])

# ===========================
# DATA PREPARATION
# ===========================
def prepare_data(df, use_engineered=True, use_polynomial=False, use_scaling=True):
    """Prepare features for modeling"""
    
    # Select base features
    features_to_use = BASE_FEATURES.copy()
    
    # Add engineered features if requested
    if use_engineered:
        engineered_cols = [
            'Total_Revenue', 'Revenue_per_Bird_Alive', 'Feed_Cost_per_Bird',
            'Total_Fixed_Costs', 'Total_Variable_Costs', 'Total_Costs',
            'Cost_per_Bird', 'Survival_Rate', 'Revenue_Cost_Ratio',
            'Feed_Efficiency', 'Profit_Margin'
        ]
        features_to_use.extend([col for col in engineered_cols if col in df.columns])
    
    # Prepare feature matrix
    X = df[features_to_use].copy()
    
    # Handle categorical variables
    if 'Farmer_Training_Level' in df.columns:
        X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
        X = pd.concat([X, X_cat], axis=1)
    
    # Create pipeline components
    pipeline_steps = []
    
    if use_scaling:
        pipeline_steps.append(('scaler', StandardScaler()))
    
    if use_polynomial:
        pipeline_steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))
    
    return X, pipeline_steps

# ===========================
# MODEL DEFINITIONS
# ===========================
def get_model_with_params(model_name):    
    #if model_name == "Random Forest":
    #    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    #    param_dist = {
    #        'model__n_estimators': [100, 200, 300, 500],
    #        'model__max_depth': [None, 5, 10, 15, 20],
    #        'model__min_samples_split': [2, 5, 10],
    #        'model__min_samples_leaf': [1, 2, 4],
    #        'model__max_features': ['sqrt', 'log2', None],
    #        'model__bootstrap': [True, False]
    #    }
    
    #elif model_name == "XGBoost":
    #    model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    #    param_dist = {
    #        'model__n_estimators': [100, 200, 300, 500],
    #        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    #        'model__max_depth': [3, 4, 5, 6, 8],
    #        'model__subsample': [0.6, 0.8, 1.0],
    #        'model__colsample_bytree': [0.6, 0.8, 1.0],
    #        'model__reg_alpha': [0, 0.1, 0.5, 1.0],
    #        'model__reg_lambda': [0, 0.1, 0.5, 1.0]
    #    }
    
    #else:  # Linear Regression
    model = LinearRegression()
    param_dist = {}  # No hyperparameters to tune
    
    return model, param_dist

def train_evaluate_model(X, y, model_name, test_size_pct, scoring_metric, 
                        tuning_iter, cv_folds, pipeline_steps, log_transform=False):
    """Train and evaluate model with comprehensive metrics"""
    
    # Target transformation
    y_transformed = y.copy()
    if log_transform and (y > 0).all():
        y_transformed = np.log1p(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transformed, test_size=test_size_pct/100, random_state=42
    )
    
    # Get model and parameters
    base_model, param_dist = get_model_with_params(model_name)
    
    # Create pipeline
    pipeline_steps.append(('model', base_model))
    pipeline = Pipeline(pipeline_steps)
    
    # Hyperparameter tuning (if applicable)
    #if param_dist and model_name != "Linear Regression":
    #    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    #    search = RandomizedSearchCV(
    #        pipeline, param_dist, n_iter=tuning_iter, cv=cv,
    #        scoring=scoring_metric, n_jobs=-1, random_state=42
    #    )
        
    #    with st.spinner(f"Training {model_name} with hyperparameter tuning..."):
    #        search.fit(X_train, y_train)
    #    
    #    best_model = search.best_estimator_
        
    #else:
    with st.spinner(f"Training {model_name}..."):
        pipeline.fit(X_train, y_train)
    best_model = pipeline
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Inverse transform if log was applied
    if log_transform and (y > 0).all():
        y_train_pred = np.expm1(y_train_pred)
        y_test_pred = np.expm1(y_test_pred)
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
    else:
        y_train_actual = y_train
        y_test_actual = y_test
    
    # Calculate metrics
    train_r2 = r2_score(y_train_actual, y_train_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    
    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'overfitting_score': train_r2 - test_r2
    }
    
    return best_model, metrics, X_train, X_test, y_test_actual, y_test_pred

# ===========================
# MAIN MODEL TRAINING
# ===========================
# Prepare data
X, pipeline_steps = prepare_data(
    data_engineered, 
    use_engineered=True,       # always use engineered
    use_polynomial=False,      # keep polynomial optional (default off)
    use_scaling=True           # always scale
)

y = data_engineered[target_name]

# Train model with auto-tuning
model, metrics, X_train, X_test, y_test_actual, y_test_pred = train_evaluate_model(
    X, y, model_name, test_size, scoring_choice, 
    tuning_iter=30,          # fixed best choice
    cv_folds=5,              # fixed best choice
    pipeline_steps=pipeline_steps, 
    log_transform=True       # always apply log transform if y > 0
)

# ===========================
# DISPLAY MODEL PERFORMANCE
# ===========================
#st.header("📊 Model Performance Dashboard")

#col1, col2, col3 = st.columns(3)

#with col1:
#    st.metric("**Train R²**", f"{metrics['train_r2']:.3f}")
#    st.metric("**Train MAE**", f"{metrics['train_mae']:.2f}")

#with col2:
#    st.metric("**Test R²**", f"{metrics['test_r2']:.3f}")
#    st.metric("**Test MAE**", f"{metrics['test_mae']:.2f}")

#with col3:
#    overfitting_indicator = "🟢" if metrics['overfitting_score'] < 0.1 else "🟡" if metrics['overfitting_score'] < 0.2 else "🔴"
#    st.metric("**Overfitting**", f"{overfitting_indicator} {metrics['overfitting_score']:.3f}")
#    st.metric("**Test RMSE**", f"{metrics['test_rmse']:.2f}")

# Performance visualization
#fig_performance = go.Figure()

#fig_performance.add_trace(go.Scatter(
#    x=y_test_actual, 
#    y=y_test_pred,
#    mode='markers',
#    name='Predictions vs Actual',
#    opacity=0.6
#))

# Perfect prediction line
#min_val, max_val = min(y_test_actual.min(), y_test_pred.min()), max(y_test_actual.max(), y_test_pred.max())
#fig_performance.add_trace(go.Scatter(
#    x=[min_val, max_val],
#    y=[min_val, max_val],
#    mode='lines',
#    name='Perfect Prediction',
#    line=dict(dash='dash', color='red')
#))

#fig_performance.update_layout(
#    title=f"Model Performance: {model_name}",
#    xaxis_title="Actual Values",
#    yaxis_title="Predicted Values"
#)

#st.plotly_chart(fig_performance, use_container_width=True)

# ===========================
# FEATURE IMPORTANCE
# ===========================
if hasattr(model.named_steps['model'], 'feature_importances_'):    
    # Get feature names (accounting for preprocessing)
    feature_names = X.columns.tolist()
    if hasattr(model.named_steps.get('poly'), 'get_feature_names_out'):
        feature_names = model.named_steps['poly'].get_feature_names_out(feature_names)
    
    importances = model.named_steps['model'].feature_importances_
    
    # Create feature importance dataframe
    fi_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)
    
    fig_importance = px.bar(fi_df, x='importance', y='feature', orientation='h',
                           title="Top 15 Most Important Features")
    #st.plotly_chart(fig_importance, use_container_width=True)

# ===========================
# USER PREDICTION INTERFACE
# ===========================
st.header("🧮 New Batch Prediction")

with st.form("enhanced_prediction_form"):
    st.write("**Enter batch features for prediction:**")
    
    input_dict = {}

    features_order = [
        'Number_of_Birds', 'Cost_per_Chick', 'Age_of_Birds_at_Sale', 'Sale_Price_per_Bird',
        'Feed_Cost_Total', 'Labor_Cost', 'Rent_Cost', 'Medicine_Cost',
        'Land_Cost', 'Infrastructure_Cost', 'Equipment_Cost', 'Utilities_Cost',
        'Feed_Conversion_Ratio', 'Mortality_Rate', 'Manure_Sales',
        'Farmer_Training_Level'   # <--- moved inside grid
    ]

    # Create rows of 4 columns
    for i in range(0, len(features_order), 4):
        cols = st.columns(4)
        for j, feature in enumerate(features_order[i:i+4]):
            label = feature.replace('_', ' ').title()

            if feature == "Farmer_Training_Level":
                chosen_level = cols[j].selectbox(label, sorted(data['Farmer_Training_Level'].unique()))
                for level in ['Beginner', 'Intermediate', 'Advanced']:
                    input_dict[f'Farmer_Training_Level_{level}'] = int(chosen_level == level)
            else:
                default_val = float(data[feature].median())
                input_dict[feature] = cols[j].number_input(label, value=default_val)

    predict_button = st.form_submit_button("🔮 Generate Prediction", type="primary")
use_engineered_features = True

# ===========================
# PREDICTION PROCESSING
# ===========================
if predict_button:
    # Create input dataframe
    input_df = pd.DataFrame([input_dict])
    
    # Apply feature engineering to input
    input_engineered = engineer_features(input_df)
    
    # Prepare input features (same as training)
    input_features, _ = prepare_data(
        input_engineered, 
        use_engineered=True,
        use_polynomial=False,
        use_scaling=False  # Scaling will be applied by the pipeline
    )
    
    # Ensure all required features are present
    missing_features = set(X.columns) - set(input_features.columns)
    for feature in missing_features:
        input_features[feature] = 0
    
    input_features = input_features[X.columns]
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Inverse transform if log was applied
    if (y > 0).all():
        prediction = np.expm1(prediction)
    
    # Store results
    st.session_state.pred_result = prediction
    st.session_state.target_label = target_label
    st.session_state.prediction_made = True
    
    # Display prediction
    st.success(f"🎯 **Predicted {target_label}: {prediction:,.2f}**")

    # Feature Analysis Section
    st.subheader("📊 Feature Impact Analysis")
    
    if target_label == "ROI (%)":
        # Show ROI Trend Chart for ROI predictions
        st.write("**Historical ROI Trend Analysis**")
        
        if 'ROI_Percentage' in data_engineered.columns and not data_engineered.empty:
            # Create ROI trend chart using Plotly
            roi_data = data_engineered.sort_values('Batch_ID')
            
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Scatter(
                x=roi_data['Batch_ID'],
                y=roi_data['ROI_Percentage'],
                mode='lines+markers',
                name='ROI Trend',
                line=dict(width=3, color='blue'),
                marker=dict(size=8, color='blue')
            ))
            
            # Add break-even line
            fig_roi.add_hline(y=0, line_dash="dash", line_color="red", 
                             annotation_text="Break-even Line")
            
            fig_roi.update_layout(
                title="ROI Trend Across All Batches",
                xaxis_title="Batch ID",
                yaxis_title="ROI Percentage (%)",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig_roi, use_container_width=True)
            st.session_state.chart_fig = fig_roi
        else:
            st.warning("ROI data not available for trend analysis.")
    
    else:
        # Show SHAP Analysis for Net P/L predictions
        st.write("**SHAP Feature Impact Analysis**")
        
        try:
            # Get SHAP explainer
            if model_name == "Linear Regression":
                explainer = shap.LinearExplainer(model.named_steps['model'], X_train)
            else:
                explainer = shap.TreeExplainer(model.named_steps['model'])
                
            # Transform input for SHAP (apply all preprocessing except the model)
            input_transformed = input_features.copy()
            for step_name, transformer in model.named_steps.items():
                if step_name != 'model':
                    input_transformed = transformer.transform(input_transformed)
            
            shap_values = explainer.shap_values(input_transformed)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_array = shap_values[0]
            else:
                shap_array = shap_values
                
            if len(shap_array.shape) > 1:
                shap_array = shap_array[0]
            
            # Create feature impact dictionary
            feature_names_final = X.columns.tolist()
            if len(shap_array) != len(feature_names_final):
                feature_names_final = feature_names_final[:len(shap_array)]
                
            feat_shap = dict(zip(feature_names_final, shap_array))
            st.session_state.feat_shap = feat_shap
            
            # Create SHAP visualization
            sorted_features = sorted(feat_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            
            fig_shap = go.Figure(go.Bar(
                x=[impact for _, impact in sorted_features],
                y=[feat.replace('_', ' ').title() for feat, _ in sorted_features],
                orientation='h',
                marker_color=['red' if impact < 0 else 'green' for _, impact in sorted_features]
            ))
            
            fig_shap.update_layout(
                title="Feature Impact on Prediction (SHAP Values)",
                xaxis_title="Impact on Prediction",
                yaxis_title="Features",
                height=500
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
            st.session_state.chart_fig = fig_shap
            
        except Exception as e:
            st.warning(f"SHAP analysis failed: {str(e)}")
            st.info("Feature importance from the trained model is shown above instead.")
            # Show feature importance as fallback
            if hasattr(model.named_steps['model'], 'feature_importances_'):    
                feature_names = X.columns.tolist()
                if hasattr(model.named_steps.get('poly'), 'get_feature_names_out'):
                    feature_names = model.named_steps['poly'].get_feature_names_out(feature_names)
                
                importances = model.named_steps['model'].feature_importances_
                
                fi_df = pd.DataFrame({
                    'feature': feature_names[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=False).head(15)
                
                fig_importance = px.bar(fi_df, x='importance', y='feature', orientation='h',
                                       title="Top 15 Most Important Features")
                st.plotly_chart(fig_importance, use_container_width=True)
                
# ===========================
# AI SUGGESTIONS
# ===========================
def generate_ai_suggestions(target_label, prediction_value, feature_impacts=None, language="English"):
    import copy
    feature_impacts = feature_impacts or {}

    # --- Base multilingual suggestions ---
    base_suggestions = {
        "feed_cost": {
            "English": "Your feed costs are significantly impacting profits. Try bulk buying, negotiate better rates with suppliers, or consider alternative feed sources to reduce costs by up to 15-20%.",
            "Marathi": "तुमचा खाद्य खर्च नफ्यावर मोठा परिणाम करत आहे. मोठ्या प्रमाणात खरेदी करा, पुरवठादारांशी चांगल्या दरांसाठी बोला किंवा १५-२०% खर्च कमी करण्यासाठी पर्यायी खाद्य स्रोत विचारात घ्या.",
            "Hindi": "आपका फ़ीड खर्च लाभ को गंभीर रूप से प्रभावित कर रहा है। थोक खरीदारी करें, आपूर्तिकर्ताओं से बेहतर दरों पर बातचीत करें, या 15-20% तक लागत कम करने के लिए वैकल्पिक फ़ीड स्रोतों पर विचार करें।"
        },
        "feed_cost_total": {
            "English": "Your total feed costs are significantly impacting profits. Try bulk buying, negotiate better rates with suppliers, or consider alternative feed sources to reduce costs by up to 15-20%.",
            "Marathi": "तुमचा एकूण खाद्य खर्च नफ्यावर मोठा परिणाम करत आहे. मोठ्या प्रमाणात खरेदी करा, पुरवठादारांशी चांगल्या दरांसाठी बोला किंवा १५-२०% खर्च कमी करण्यासाठी पर्यायी खाद्य स्रोत विचारात घ्या.",
            "Hindi": "आपकी कुल फ़ीड लागत लाभ को गंभीर रूप से प्रभावित कर रही है। थोक खरीदारी करें, आपूर्तिकर्ताओं से बेहतर दरों पर बातचीत करें, या 15-20% तक लागत कम करने के लिए वैकल्पिक फ़ीड स्रोतों पर विचार करें।"
        },
        "total_fixed_costs": {
            "English": "High fixed costs are reducing profitability. Review equipment financing options, consider leasing instead of buying, and optimize facility utilization to spread fixed costs over more birds.",
            "Marathi": "जास्त स्थिर खर्चामुळे नफादायकता कमी होत आहे. उपकरणांच्या वित्तपुरवठा पर्यायांचे पुनरावलोकन करा, खरेदीऐवजी भाड्याने घेण्याचा विचार करा आणि अधिक पक्ष्यांवर स्थिर खर्च वाटण्यासाठी सुविधा वापराचे अनुकूलन करा.",
            "Hindi": "उच्च स्थिर लागत लाभप्रदता को कम कर रही है। उपकरण वित्तपोषण विकल्पों की समीक्षा करें, खरीदने के बजाय लीज़िंग पर विचार करें, और अधिक पक्षियों पर स्थिर लागतों को वितरित करने के लिए सुविधा उपयोग का अनुकूलन करें।"
        },
        "total_costs": {
            "English": "Overall costs are too high for optimal profitability. Conduct detailed cost analysis, identify major cost drivers, and implement systematic cost reduction measures across all operations.",
            "Marathi": "इष्टतम नफादायकतेसाठी एकूण खर्च खूप जास्त आहेत. तपशीलवार खर्च विश्लेषण करा, प्रमुख खर्च चालकांची ओळख करा आणि सर्व कामकाजात पद्धतशीर खर्च कपात उपाय अंमलात आणा.",
            "Hindi": "इष्टतम लाभप्रदता के लिए कुल लागत बहुत अधिक है। विस्तृत लागत विश्लेषण करें, मुख्य लागत चालकों की पहचान करें, और सभी संचालनों में व्यवस्थित लागत कम करने के उपायों को लागू करें।"
        },
        "total_variable_costs": {
            "English": "Variable costs are impacting margins significantly. Focus on feed efficiency, reduce wastage, optimize medicine usage, and negotiate better rates for all variable inputs.",
            "Marathi": "परिवर्तनीय खर्च मार्जिनवर लक्षणीय परिणाम करत आहेत. खाद्य कार्यक्षमतेवर लक्ष द्या, अपव्यय कमी करा, औषधांचा वापर अनुकूलित करा आणि सर्व परिवर्तनीय इनपुटसाठी चांगल्या दरांची वाटाघाटी करा.",
            "Hindi": "परिवर्तनीय लागतें मार्जिन को काफी प्रभावित कर रही हैं। फ़ीड दक्षता पर ध्यान दें, बर्बादी कम करें, दवा के उपयोग को अनुकूलित करें, और सभी परिवर्तनीय इनपुट के लिए बेहतर दरों पर बातचीत करें।"
        },
        "survival_rate": {
            "English": "Low survival rate is severely impacting profits. Implement strict biosecurity measures, improve vaccination schedules, maintain optimal temperature and humidity, and enhance overall bird welfare.",
            "Marathi": "कमी जिवंत राहण्याचा दर नफ्यावर गंभीर परिणाम करत आहे. कठोर जैवसुरक्षा उपाय अंमलात आणा, लसीकरण वेळापत्रक सुधारा, इष्टतम तापमान आणि आर्द्रता राखा आणि एकूण पक्षी कल्याण वाढवा.",
            "Hindi": "कम जीवित रहने की दर मुनाफे को गंभीर रूप से प्रभावित कर रही है। सख्त जैवसुरक्षा उपायों को लागू करें, टीकाकरण कार्यक्रम सुधारें, इष्टतम तापमान और आर्द्रता बनाए रखें, और समग्र पक्षी कल्याण बढ़ाएं।"
        },
        "land_cost": {
            "English": "Land costs are impacting profitability. Consider optimizing land use efficiency, explore partnerships or cooperative farming models, or evaluate relocation to more cost-effective areas.",
            "Marathi": "जमिनीचा खर्च नफादायकतेवर परिणाम करत आहे. जमिनीचा वापर कार्यक्षमता अनुकूलित करण्याचा विचार करा, भागीदारी किंवा सहकारी शेती मॉडेल एक्सप्लोर करा किंवा अधिक किफायतशीर भागात स्थलांतराचे मूल्यांकन करा.",
            "Hindi": "भूमि की लागत लाभप्रदता को प्रभावित कर रही है। भूमि उपयोग दक्षता को अनुकूलित करने पर विचार करें, साझेदारी या सहकारी कृषि मॉडल का अन्वेषण करें, या अधिक लागत-प्रभावी क्षेत्रों में स्थानांतरण का मूल्यांकन करें।"
        },
        "mortality_rate": {
            "English": "High mortality is severely affecting your profits. Focus on strict vaccination schedules, improve housing conditions, and maintain better biosecurity to reduce mortality by 3-5%.",
            "Marathi": "जास्त मृत्यू दर तुमच्या नफ्यावर गंभीर परिणाम करत आहे. कठोर लसीकरण वेळापत्रक, घरांची परिस्थिती सुधारा आणि ३-५% मृत्यू दर कमी करण्यासाठी चांगली जैवसुरक्षा राखा.",
            "Hindi": "उच्च मृत्यु दर आपके मुनाफे को गंभीर रूप से प्रभावित कर रही है। सख्त टीकाकरण कार्यक्रम पर ध्यान दें, आवास की स्थिति सुधारें, और 3-5% मृत्यु दर कम करने के लिए बेहतर जैवसुरक्षा बनाए रखें।"
        },
        "medicine_cost": {
            "English": "Medicine costs are impacting your bottom line. Invest in preventive healthcare practices, maintain cleaner environments, and focus on nutrition to reduce medicine dependency by 30-40%.",
            "Marathi": "औषधांचा खर्च तुमच्या तळाशी प्रभावित करत आहे. प्रतिबंधक आरोग्य पद्धतींमध्ये गुंतवणूक करा, स्वच्छ वातावरण राखा आणि ३०-४०% औषध अवलंबित्व कमी करण्यासाठी पोषणावर लक्ष द्या.",
            "Hindi": "दवा की लागत आपकी मुनाफे को प्रभावित कर रही है। निवारक स्वास्थ्य पद्धतियों में निवेश करें, स्वच्छ वातावरण बनाए रखें, और 30-40% दवा निर्भरता कम करने के लिए पोषण पर ध्यान दें।"
        },
        "electricity": {
            "English": "Electricity costs are eating into profits. Improve shed ventilation naturally, adopt energy-efficient equipment, and consider solar solutions to cut electricity bills by 20-25%.",
            "Marathi": "वीज खर्च नफ्यात कपात करत आहे. शेडचे नैसर्गिक वायुवीजन सुधारा, ऊर्जा-कार्यक्षम उपकरणे वापरा आणि २०-२५% वीज बिले कमी करण्यासाठी सौर उपायांचा विचार करा.",
            "Hindi": "बिजली की लागत मुनाफे को कम कर रही है। शेड वेंटिलेशन को प्राकृतिक रूप से सुधारें, ऊर्जा-कुशल उपकरण अपनाएं, और 20-25% बिजली के बिल कम करने के लिए सोलर समाधान पर विचार करें।"
        },
        "fcr": {
            "English": "Poor Feed Conversion Ratio is hurting profitability. Monitor bird weights weekly, ensure feed quality, and optimize feeding schedules to improve FCR by 0.1-0.2 points.",
            "Marathi": "खराब खाद्य रूपांतरण प्रमाण नफादायकतेला हानी पोहोचवत आहे. आठवड्यातून पक्षी वजनाचे निरीक्षण करा, खाद्य गुणवत्ता सुनिश्चित करा आणि FCR ०.१-०.२ गुणांनी सुधारण्यासाठी आहार वेळापत्रक अनुकूलित करा.",
            "Hindi": "खराब फ़ीड रूपांतरण अनुपात लाभप्रदता को नुकसान पहुंचा रहा है। साप्ताहिक पक्षी वजन की निगरानी करें, फ़ीड गुणवत्ता सुनिश्चित करें, और FCR को 0.1-0.2 अंकों से सुधारने के लिए फीडिंग शेड्यूल को अनुकूलित करें।"
        },
        "avg_weight": {
            "English": "Low average bird weight is reducing revenue potential. Focus on balanced nutrition, ensure adequate feeding space, and monitor growth patterns to increase weight by 100-150g per bird.",
            "Marathi": "कमी सरासरी पक्षी वजनामुळे उत्पन्न क्षमता कमी होत आहे. संतुलित पोषणावर लक्ष द्या, पुरेशी आहार जागा सुनिश्चित करा आणि प्रति पक्षी १००-१५० ग्रॅम वजन वाढवण्यासाठी वाढीच्या पद्धतींचे निरीक्षण करा.",
            "Hindi": "कम औसत पक्षी वजन राजस्व क्षमता को कम कर रहा है। संतुलित पोषण पर ध्यान दें, पर्याप्त फीडिंग स्थान सुनिश्चित करें, और प्रति पक्षी 100-150 ग्राम वजन बढ़ाने के लिए वृद्धि पैटर्न की निगरानी करें।"
        },
        "sale_price": {
            "English": "Sale prices are below potential. Research market trends, consider direct sales to consumers, or negotiate better rates with buyers to increase price by ₹2-5 per kg.",
            "Marathi": "विक्री किमती क्षमतेपेक्षा कमी आहेत. बाजार ट्रेंडचे संशोधन करा, ग्राहकांना थेट विक्री विचारात घ्या किंवा प्रति किलो ₹२-५ किमत वाढवण्यासाठी खरेदीदारांशी चांगल्या दरांची वाटाघाटी करा.",
            "Hindi": "बिक्री मूल्य क्षमता से कम है। बाजार के रुझान का अनुसंधान करें, उपभोक्ताओं को सीधी बिक्री पर विचार करें, या प्रति किलो ₹2-5 मूल्य बढ़ाने के लिए खरीदारों के साथ बेहतर दरों पर बातचीत करें।"
        },
        "chick_cost": {
            "English": "High chick costs are impacting margins. Source from reliable but competitive hatcheries, buy in bulk during off-season, or negotiate annual contracts for better rates.",
            "Marathi": "जास्त चिमुकल्यांचा खर्च मार्जिनवर परिणाम करत आहे. विश्वसनीय परंतु स्पर्धात्मक हॅचरीकडून खरेदी करा, ऑफ-सीझनमध्ये मोठ्या प्रमाणात खरेदी करा किंवा चांगल्या दरांसाठी वार्षिक करारावर वाटाघाटी करा.",
            "Hindi": "उच्च चूजा लागत मार्जिन को प्रभावित कर रही है। विश्वसनीय लेकिन प्रतिस्पर्धी हैचरी से सोर्स करें, ऑफ-सीजन में थोक में खरीदें, या बेहतर दरों के लिए वार्षिक अनुबंधों पर बातचीत करें।"
        },
        "litter_cost": {
            "English": "Litter management costs are adding up. Reuse litter for 2-3 batches with proper treatment, source locally to reduce transport costs, or consider alternative bedding materials.",
            "Marathi": "लिटर व्यवस्थापनाचे खर्च वाढत आहेत. योग्य उपचाराने २-३ तुकड्यांसाठी लिटरचा पुनर्वापर करा, वाहतूक खर्च कमी करण्यासाठी स्थानिक पातळीवर खरेदी करा किंवा पर्यायी बेडिंग सामग्रीचा विचार करा.",
            "Hindi": "लिटर प्रबंधन लागत बढ़ रही है। उचित उपचार के साथ 2-3 बैच के लिए लिटर का पुन: उपयोग करें, परिवहन लागत कम करने के लिए स्थानीय रूप से सोर्स करें, या वैकल्पिक बिस्तर सामग्री पर विचार करें।"
        },
        "transportation": {
            "English": "Transportation costs are significant. Optimize delivery routes, coordinate bulk deliveries with neighbors, or negotiate better rates with transporters to save 10-15%.",
            "Marathi": "वाहतूक खर्च लक्षणीय आहेत. डिलिव्हरी मार्ग अनुकूलित करा, शेजाऱ्यांसह मोठ्या प्रमाणात डिलिव्हरीची समन्वय करा किंवा १०-१५% बचत करण्यासाठी वाहतुकदारांशी चांगल्या दरांची वाटाघाटी करा.",
            "Hindi": "परिवहन लागत महत्वपूर्ण है। डिलीवरी मार्गों को अनुकूलित करें, पड़ोसियों के साथ थोक डिलीवरी का समन्वय करें, या 10-15% बचत के लिए ट्रांसपोर्टरों के साथ बेहतर दरों पर बातचीत करें।"
        },
        "energy_cost": {
            "English": "Energy costs are eating into profits. Improve shed ventilation naturally, adopt energy-efficient equipment, and consider solar solutions to reduce electricity bills by 20-25%.",
            "Marathi": "ऊर्जा खर्च नफ्यात कमी करत आहे. शेड वेंटिलेशन नैसर्गिकरित्या सुधारित करा, ऊर्जा-कार्यक्षम उपकरणे वापरा आणि २०-२५% वीज बिले कमी करण्यासाठी सौर उपायांचा विचार करा.",
            "Hindi": "बिजली की लागत मुनाफे को कम कर रही है। शेड वेंटिलेशन को प्राकृतिक रूप से सुधारें, ऊर्जा-कुशल उपकरण अपनाएं, और 20-25% बिजली के बिल कम करने के लिए सोलर समाधान पर विचार करें।"
        },
        "fcr": {
            "English": "Poor Feed Conversion Ratio is hurting profitability. Monitor bird weights weekly, ensure feed quality, and optimize feeding schedules to improve FCR by 0.1-0.2 points.",
            "Marathi": "खराब खाद्य रूपांतरण प्रमाण नफादायकतेला हानी पोहोचवत आहे. आठवड्यातून पक्षी वजनाचे निरीक्षण करा, खाद्य गुणवत्ता सुनिश्चित करा आणि FCR ०.१-०.२ गुणांनी सुधारण्यासाठी आहार वेळापत्रक अनुकूलित करा.",
            "Hindi": "खराब फ़ीड रूपांतरण अनुपात लाभप्रदता को नुकसान पहुंचा रहा है। साप्ताहिक पक्षी वजन की निगरानी करें, फ़ीड गुणवत्ता सुनिश्चित करें, और FCR को 0.1-0.2 अंकों से सुधारने के लिए फीडिंग शेड्यूल को अनुकूलित करें।"
        },
        "avg_weight": {
            "English": "Low average bird weight is reducing revenue potential. Focus on balanced nutrition, ensure adequate feeding space, and monitor growth patterns to increase weight by 100-150g per bird.",
            "Marathi": "कमी सरासरी पक्षी वजनामुळे उत्पन्न क्षमता कमी होत आहे. संतुलित पोषणावर लक्ष द्या, पुरेशी आहार जागा सुनिश्चित करा आणि प्रति पक्षी १००-१५० ग्रॅम वजन वाढवण्यासाठी वाढीच्या पद्धतींचे निरीक्षण करा.",
            "Hindi": "कम औसत पक्षी वजन राजस्व क्षमता को कम कर रहा है। संतुलित पोषण पर ध्यान दें, पर्याप्त फीडिंग स्थान सुनिश्चित करें, और प्रति पक्षी 100-150 ग्राम वजन बढ़ाने के लिए वृद्धि पैटर्न की निगरानी करें।"
        },
        "sale_price": {
            "English": "Sale prices are below potential. Research market trends, consider direct sales to consumers, or negotiate better rates with buyers to increase price by ₹2-5 per kg.",
            "Marathi": "विक्री किमती क्षमतेपेक्षा कमी आहेत. बाजार ट्रेंडचे संशोधन करा, ग्राहकांना थेट विक्री विचारात घ्या किंवा प्रति किलो ₹२-५ किमत वाढवण्यासाठी खरेदीदारांशी चांगल्या दरांची वाटाघाटी करा.",
            "Hindi": "बिक्री मूल्य क्षमता से कम है। बाजार के रुझान का अनुसंधान करें, उपभोक्ताओं को सीधी बिक्री पर विचार करें, या प्रति किलो ₹2-5 मूल्य बढ़ाने के लिए खरीदारों के साथ बेहतर दरों पर बातचीत करें।"
        },
        "chick_cost": {
            "English": "High chick costs are impacting margins. Source from reliable but competitive hatcheries, buy in bulk during off-season, or negotiate annual contracts for better rates.",
            "Marathi": "जास्त चिमुकल्यांचा खर्च मार्जिनवर परिणाम करत आहे. विश्वसनीय परंतु स्पर्धात्मक हॅचरीकडून खरेदी करा, ऑफ-सीझनमध्ये मोठ्या प्रमाणात खरेदी करा किंवा चांगल्या दरांसाठी वार्षिक करारावर वाटाघाटी करा.",
            "Hindi": "उच्च चूजा लागत मार्जिन को प्रभावित कर रही है। विश्वसनीय लेकिन प्रतिस्पर्धी हैचरी से सोर्स करें, ऑफ-सीजन में थोक में खरीदें, या बेहतर दरों के लिए वार्षिक अनुबंधों पर बातचीत करें।"
        },
        "litter_cost": {
            "English": "Litter management costs are adding up. Reuse litter for 2-3 batches with proper treatment, source locally to reduce transport costs, or consider alternative bedding materials.",
            "Marathi": "लिटर व्यवस्थापनाचे खर्च वाढत आहेत. योग्य उपचाराने २-३ तुकड्यांसाठी लिटरचा पुनर्वापर करा, वाहतूक खर्च कमी करण्यासाठी स्थानिक पातळीवर खरेदी करा किंवा पर्यायी बेडिंग सामग्रीचा विचार करा.",
            "Hindi": "लिटर प्रबंधन लागत बढ़ रही है। उचित उपचार के साथ 2-3 बैच के लिए लिटर का पुन: उपयोग करें, परिवहन लागत कम करने के लिए स्थानीय रूप से सोर्स करें, या वैकल्पिक बिस्तर सामग्री पर विचार करें।"
        },
        "transportation": {
            "English": "Transportation costs are significant. Optimize delivery routes, coordinate bulk deliveries with neighbors, or negotiate better rates with transporters to save 10-15%.",
            "Marathi": "वाहतूक खर्च लक्षणीय आहेत. डिलिव्हरी मार्ग अनुकूलित करा, शेजाऱ्यांसह मोठ्या प्रमाणात डिलिव्हरीची समन्वय करा किंवा १०-१५% बचत करण्यासाठी वाहतुकदारांशी चांगल्या दरांची वाटाघाटी करा.",
            "Hindi": "परिवहन लागत महत्वपूर्ण है। डिलीवरी मार्गों को अनुकूलित करें, पड़ोसियों के साथ थोक डिलीवरी का समन्वय करें, या 10-15% बचत के लिए ट्रांसपोर्टरों के साथ बेहतर दरों पर बातचीत करें।"
        }
    }

    # --- Performance messages ---
    performance_messages = {
        "English": {
            "high_loss": "🔴 CRITICAL: Significant loss predicted! Immediate action required on these key areas:",
            "loss": "⚠️ Loss Alert: Your batch is predicted to make a loss. Focus on these high-impact areas:",
            "low_profit": "📊 Low Profit Warning: These factors need immediate attention to improve margins:",
            "moderate_profit": "📈 Moderate Performance: Good foundation, but these improvements can boost profits:",
            "good_profit": "✅ Strong Performance! These factors are driving your success:"
        },
        "Hindi": {
            "high_loss": "🔴 गंभीर: महत्वपूर्ण नुकसान की भविष्यवाणी! इन मुख्य क्षेत्रों पर तत्काल कार्रवाई आवश्यक:",
            "loss": "⚠️ नुकसान चेतावनी: आपके बैच में नुकसान की संभावना है। इन उच्च प्रभाव वाले क्षेत्रों पर ध्यान दें:",
            "low_profit": "📊 कम लाभ चेतावनी: मार्जिन सुधारने के लिए इन कारकों पर तुरंत ध्यान देना आवश्यक:",
            "moderate_profit": "📈 मध्यम प्रदर्शन: अच्छी नींव है, लेकिन ये सुधार मुनाफा बढ़ा सकते हैं:",
            "good_profit": "✅ मजबूत प्रदर्शन! ये कारक आपकी सफलता को बढ़ा रहे हैं:"
        },
        "Marathi": {
            "high_loss": "🔴 गंभीर: मोठा तोटा अपेक्षित! या मुख्य भागांवर तातडीने कारवाई आवश्यक:",
            "loss": "⚠️ तोटा इशारा: तुमच्या तुकडीत तोटा होण्याची शक्यता आहे. या उच्च प्रभाव असलेल्या भागांवर लक्ष द्या:",
            "low_profit": "📊 कमी नफा इशारा: मार्जिन सुधारण्यासाठी या घटकांवर तातडीने लक्ष देणे आवश्यक:",
            "moderate_profit": "📈 मध्यम कामगिरी: चांगला पाया आहे, परंतु ही सुधारणा नफा वाढवू शकतात:",
            "good_profit": "✅ मजबूत कामगिरी! हे घटक तुमच्या यशाला चालना देत आहेत:"
        }
    }

    # --- Feature name translations ---
    feature_translations = {
        "feed_cost": {
            "English": "Feed Cost",
            "Marathi": "खाद्य खर्च", 
            "Hindi": "फ़ीड लागत"
        },
        "mortality_rate": {
            "English": "Mortality Rate",
            "Marathi": "मृत्यू दर",
            "Hindi": "मृत्यु दर"
        },
        "medicine_cost": {
            "English": "Medicine Cost",
            "Marathi": "औषध खर्च",
            "Hindi": "दवा लागत"
        },
        "electricity": {
            "English": "Electricity",
            "Marathi": "वीज",
            "Hindi": "बिजली"
        },
        "fcr": {
            "English": "Feed Conversion Ratio",
            "Marathi": "खाद्य रूपांतरण प्रमाण",
            "Hindi": "फ़ीड रूपांतरण अनुपात"
        },
        "avg_weight": {
            "English": "Average Weight",
            "Marathi": "सरासरी वजन",
            "Hindi": "औसत वजन"
        },
        "sale_price": {
            "English": "Sale Price",
            "Marathi": "विक्री किमत",
            "Hindi": "बिक्री मूल्य"
        },
        "chick_cost": {
            "English": "Chick Cost",
            "Marathi": "चिमुकल्या खर्च",
            "Hindi": "चूजा लागत"
        },
        "litter_cost": {
            "English": "Litter Cost",
            "Marathi": "लिटर खर्च",
            "Hindi": "लिटर लागत"
        },
        "transportation": {
            "English": "Transportation",
            "Marathi": "वाहतूक",
            "Hindi": "परिवहन"
        }
    }

    # --- Section headers ---
    section_headers = {
        "English": {
            "key_areas": "🔑 **Key Areas Impacting Performance:**",
            "detailed_recommendations": "🛠️ **Detailed Recommendations:**"
        },
        "Hindi": {
            "key_areas": "🔑 **प्रदर्शन को प्रभावित करने वाले मुख्य क्षेत्र:**",
            "detailed_recommendations": "🛠️ **विस्तृत सिफारिशें:**"
        },
        "Marathi": {
            "key_areas": "🔑 **कामगिरीवर परिणाम करणारे मुख्य क्षेत्र:**",
            "detailed_recommendations": "🛠️ **तपशीलवार शिफारसी:**"
        }
    }

    # --- Step 1: Decide performance key based on target ---
    if "ROI" in target_label.upper():
        if prediction_value < -20:
            perf_key = "high_loss"
        elif prediction_value < 0:
            perf_key = "loss"
        elif prediction_value < 15:
            perf_key = "low_profit"
        elif prediction_value < 30:
            perf_key = "moderate_profit"
        else:
            perf_key = "good_profit"
    else:
        if prediction_value < -10000:
            perf_key = "high_loss"
        elif prediction_value < 0:
            perf_key = "loss"
        elif prediction_value < 5000:
            perf_key = "low_profit"
        elif prediction_value < 15000:
            perf_key = "moderate_profit"
        else:
            perf_key = "good_profit"

    # --- Step 2: Build report ---
    report = []
    report.append(f"📊 **Performance Summary:** {performance_messages[language][perf_key]}")

    if feature_impacts:
        # Sort by impact and remove duplicates
        sorted_features = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:6]

        # Key areas - now translated
        translated_feature_names = []
        for feature_name, impact in top_features:
            feature_key = feature_name.lower().replace(" ", "_")
            translated_name = feature_translations.get(feature_key, {}).get(language, 
                                                     feature_name.replace("_", " "))
            translated_feature_names.append(translated_name)
        
        key_areas_text = ", ".join(translated_feature_names)
        report.append(f"{section_headers[language]['key_areas']} {key_areas_text}")

        # Detailed recommendations - now with translated feature names
        detailed = [section_headers[language]['detailed_recommendations']]
        for feature, impact in top_features:
            feature_key = feature.lower().replace(" ", "_")
            suggestion_text = base_suggestions.get(feature_key, {}).get(language, 
                                "This factor strongly impacts results—review management strategies.")
            translated_feature_name = feature_translations.get(feature_key, {}).get(language, 
                                                             feature.replace("_", " "))
            detailed.append(f"- **{translated_feature_name}**: {suggestion_text}")

        report.extend(detailed)

    # --- Step 3: Return as single Markdown block ---
    final_report = "\n".join(report)
    return [final_report]

    return suggestions

if st.session_state.prediction_made:
    st.subheader("💡 AI-Powered Farming Intelligence")
    
    with st.expander("Click here for personalized recommendations based on your data", expanded=True):
        
        # Generate suggestions
        suggestions = generate_ai_suggestions(
            target_label=st.session_state.target_label,
            prediction_value=st.session_state.pred_result,
            feature_impacts=st.session_state.feat_shap,
            language=language_choice  
        )
        
        if suggestions:
            st.markdown("### 📊 Performance Assessment")
            
            st.markdown(suggestions[0], unsafe_allow_html=True) 
            
        else:
            st.info("Detailed recommendations will appear after generating a prediction with feature analysis.")

def convert_shap_to_farmer_friendly(feature_impacts, target_label):
    """Convert SHAP values to farmer-friendly explanations with percentages."""
    friendly_impacts = {}
    if not feature_impacts:
        return friendly_impacts
    
    # Calculate max absolute impact for scaling to a 0-100 range
    max_impact = max(abs(v) for v in feature_impacts.values()) if feature_impacts else 1
    
    for feature, impact in feature_impacts.items():
        # Normalize impact to a 0-100 percentage score
        normalized_impact = (abs(impact) / max_impact * 100) if max_impact > 0 else 0
        
        friendly_impacts[feature] = {
            'score': normalized_impact,
            'direction': 'Positive' if impact > 0 else 'Negative',
            'raw_impact': impact
        }
    return friendly_impacts

# ... OPTIMIZED: This function now displays percentages instead of fuzzy labels ...
def display_farmer_friendly_impacts_streamlit(feature_impacts, target_label):
    """Display farmer-friendly impacts in Streamlit using percentages."""
    if not feature_impacts:
        st.warning("No feature impact data available.")
        return
    
    friendly_impacts = convert_shap_to_farmer_friendly(feature_impacts, target_label)
    
    st.subheader("🎯 Impact Analysis (by Percentage)")
    
    sorted_impacts = sorted(friendly_impacts.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Display top impactful features
    for feature, impact_data in sorted_impacts[:8]:
        col1, col2, col3 = st.columns([4, 2, 2])
        
        with col1:
            st.write(f"**{feature.replace('_', ' ').title()}**")
        
        with col2:
            # Impact percentage with color coding
            score = impact_data['score']
            if score >= 70:
                st.write(f"🔴 **{score:.0f}%**")
            elif score >= 30:
                st.write(f"🟡 **{score:.0f}%**")
            else:
                st.write(f"🟢 **{score:.0f}%**")
        
        with col3:
            # Direction with arrows
            direction = impact_data['direction']
            st.write("📈 **Increases Profit**" if direction == 'Positive' else "📉 **Decreases Profit**")
    
    st.info("""
    **How to read this:**
    - **Percentage (%)**: Shows how strongly a factor affects your result compared to others. Higher is more important.
    - 🔴 **Critical Impact (70%+)**: Pay close attention to this.
    - 🟡 **Moderate Impact (30-69%)**: Important for improving results.
    - 🟢 **Lower Impact (<30%)**: Less critical but still relevant.
    """)
    
    # Visual bar chart
    st.subheader("📊 Visual Impact Summary")
    chart_data = [{'feature': item[0], **item[1]} for item in sorted_impacts[:8]]
    df_chart = pd.DataFrame(chart_data)
    df_chart['signed_score'] = df_chart.apply(lambda row: row['score'] if row['direction'] == 'Positive' else -row['score'], axis=1)
    df_chart['color'] = df_chart['direction'].apply(lambda d: 'green' if d == 'Positive' else 'red')
    
    fig_simple = go.Figure(go.Bar(
        y=df_chart['feature'].str.replace('_', ' ').str.title(),
        x=df_chart['signed_score'],
        orientation='h',
        marker_color=df_chart['color'],
        text=df_chart['score'].apply(lambda s: f"{s:.0f}%"),
        textposition='outside'
    ))
    fig_simple.update_layout(
        title="Key Factors Affecting Your Business",
        xaxis_title="Impact Score (%) (Positive = Good, Negative = Bad)",
        yaxis_title="Farm Factors",
        yaxis={'autorange': 'reversed'},
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_simple, use_container_width=True)
    
def setup_matplotlib_fonts():
    """Setup matplotlib to handle multilingual text"""
    try:
        # Try to use a Unicode font for better multilingual support
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        unicode_fonts = ['DejaVu Sans', 'Arial Unicode MS', 'Noto Sans', 'Liberation Sans']
        
        selected_font = 'sans-serif'  # fallback
        for font in unicode_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        rcParams['font.family'] = selected_font
        rcParams['axes.unicode_minus'] = False
        return selected_font
    except Exception:
        return 'sans-serif'

def create_shap_chart_matplotlib(feature_impacts, target_label):
    """Create SHAP chart using matplotlib for better PDF compatibility"""
    if not feature_impacts:
        return None
    
    # Setup font
    font_name = setup_matplotlib_fonts()
    
    # Sort features by absolute impact
    sorted_items = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    features = [item[0].replace('_', ' ').title() for item in sorted_items]
    impacts = [item[1] for item in sorted_items]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars based on positive/negative impact
    colors = ['green' if x > 0 else 'red' for x in impacts]
    
    # Create horizontal bar chart
    bars = ax.barh(features, impacts, color=colors, alpha=0.7)
    
    # Customize chart
    ax.set_xlabel('Impact on Prediction', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Impact Analysis - {target_label}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, impacts):
        width = bar.get_width()
        ax.text(width + (0.01 * max(abs(min(impacts)), abs(max(impacts)))),
                bar.get_y() + bar.get_height()/2,
                f'{value:.2f}', ha='left' if width > 0 else 'right',
                va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    # Save to BytesIO
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def create_roi_trend_matplotlib(data):
    """Create ROI trend chart using matplotlib"""
    if data.empty or 'ROI_Percentage' not in data.columns:
        return None
    
    setup_matplotlib_fonts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort data by Batch_ID for proper trend line
    trend_data = data.sort_values('Batch_ID')
    
    ax.plot(trend_data['Batch_ID'], trend_data['ROI_Percentage'], 
            marker='o', linewidth=2, markersize=6, color='blue')
    
    ax.set_xlabel('Batch ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROI Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('ROI Trend Across Batches', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at 0% ROI
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax.legend()
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def download_fonts():
    """Download required fonts for multilingual support"""
    font_dir = "fonts"
    os.makedirs(font_dir, exist_ok=True)
    
    font_urls = {
        'NotoSans-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf',
        'NotoSans-Bold.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf',
        'NotoSansDevanagari-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf'
    }
    
    for filename, url in font_urls.items():
        filepath = os.path.join(font_dir, filename)
        if not os.path.exists(filepath):
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                st.warning(f"Could not download {filename}: {str(e)}")
    
    return font_dir

def register_pdf_fonts():
    """Register fonts for ReportLab PDF generation"""
    font_dir = download_fonts()
    
    fonts_registered = {}
    
    # Try to register fonts
    font_files = {
        'NotoSans': 'NotoSans-Regular.ttf',
        'NotoSans-Bold': 'NotoSans-Bold.ttf',
        'NotoSansDevanagari': 'NotoSansDevanagari-Regular.ttf'
    }
    
    for font_name, filename in font_files.items():
        filepath = os.path.join(font_dir, filename)
        if os.path.exists(filepath):
            try:
                pdfmetrics.registerFont(TTFont(font_name, filepath))
                fonts_registered[font_name] = True
            except Exception as e:
                fonts_registered[font_name] = False
        else:
            fonts_registered[font_name] = False
    
    return fonts_registered

def get_content_translations():
    """Get all text content and feature translations in multiple languages."""
    return {
        'en': {
            'title': 'Poultry Batch Analysis Report',
            'prediction_section': 'Prediction Results',
            'shap_section': 'Feature Impact Analysis', 
            'positive_impacts': 'Factors Helping Your Profit',
            'negative_impacts': 'Factors Hurting Your Profit',
            'suggestions_section': 'AI Recommendations',
            'feature_name': 'Farm Factor',
            'impact_percentage': 'Impact (%)',
            'no_positive': 'No significant positive factors found.',
            'no_negative': 'No significant negative factors found.',
            'generated_on': 'Report generated on',
            # English doesn't need translations, but keys are here for consistency
            'feature_translations': {} 
        },
        'hi': {
            'title': 'मुर्गी पालन बैच विश्लेषण रिपोर्ट',
            'prediction_section': 'पूर्वानुमान परिणाम',
            'shap_section': 'विशेषताओं का प्रभाव विश्लेषण',
            'positive_impacts': 'आपके लाभ में मदद करने वाले कारक',
            'negative_impacts': 'आपके लाभ को नुकसान पहुंचाने वाले कारक',
            'suggestions_section': 'AI सुझाव',
            'feature_name': 'फार्म कारक',
            'impact_percentage': 'प्रभाव (%)',
            'no_positive': 'कोई महत्वपूर्ण सकारात्मक कारक नहीं मिला।',
            'no_negative': 'कोई महत्वपूर्ण नकारात्मक कारक नहीं मिला।',
            'generated_on': 'रिपोर्ट तैयार की गई',
            'feature_translations': {
                'Number Of Birds': 'पक्षियों की संख्या', 'Cost Per Chick': 'चूजे की लागत', 'Feed Cost Total': 'कुल फीड लागत',
                'Labor Cost': 'श्रम लागत', 'Rent Cost': 'किराया लागत', 'Medicine Cost': 'दवा लागत', 'Land Cost': 'भूमि लागत',
                'Infrastructure Cost': 'ढांचा लागत', 'Equipment Cost': 'उपकरण लागत', 'Utilities Cost': 'उपयोगिताएं लागत',
                'Feed Conversion Ratio': 'फीड रूपांतरण अनुपात', 'Mortality Rate': 'मृत्यु दर', 'Manure Sales': 'खाद बिक्री',
                'Age Of Birds At Sale': 'बिक्री के समय पक्षियों की आयु', 'Sale Price Per Bird': 'प्रति पक्षी बिक्री मूल्य',
                'Total Revenue': 'कुल आय', 'Revenue Per Bird Alive': 'जीवित पक्षी प्रति आय', 'Feed Cost Per Bird': 'प्रति पक्षी फीड लागत',
                'Total Fixed Costs': 'कुल निश्चित लागत', 'Total Variable Costs': 'कुल परिवर्तनीय लागत', 'Total Costs': 'कुल लागत',
                'Cost Per Bird': 'प्रति पक्षी लागत', 'Survival Rate': 'उत्तरजीविता दर', 'Revenue Cost Ratio': 'आय लागत अनुपात',
                'Feed Efficiency': 'फीड दक्षता', 'Profit Margin': 'लाभ मार्जिन'
            }
        },
        'mr': {
            'title': 'कुक्कुट तुकडींचा विश्लेषण अहवाल',
            'prediction_section': 'अंदाजित निकाल',
            'shap_section': 'वैशिष्ट्यांचा प्रभाव विश्लेषण',
            'positive_impacts': 'तुमच्या नफ्यात मदत करणारे घटक',
            'negative_impacts': 'तुमच्या नफ्याला हानी पोहोचवणारे घटक',
            'suggestions_section': 'AI शिफारसी',
            'feature_name': 'फार्म घटक',
            'impact_percentage': 'प्रभाव (%)',
            'no_positive': 'कोणताही महत्त्वपूर्ण सकारात्मक घटक आढळला नाही।',
            'no_negative': 'कोणताही महत्त्वपूर्ण नकारात्मक घटक आढळला नाही।',
            'generated_on': 'अहवाल तयार केला',
            'feature_translations': {
                'Number Of Birds': 'पक्ष्यांची संख्या', 'Cost Per Chick': 'पिल्लाची किंमत', 'Feed Cost Total': 'एकूण खाद्य खर्च',
                'Labor Cost': 'कामगार खर्च', 'Rent Cost': 'भाडे खर्च', 'Medicine Cost': 'औषध खर्च', 'Land Cost': 'जमिनीचा खर्च',
                'Infrastructure Cost': 'पायाभूत सुविधा खर्च', 'Equipment Cost': 'उपकरण खर्च', 'Utilities Cost': 'उपयोगिता खर्च',
                'Feed Conversion Ratio': 'खाद्य रूपांतरण प्रमाण', 'Mortality Rate': 'मृत्यू दर', 'Manure Sales': 'खत विक्री',
                'Age Of Birds At Sale': 'विक्रीच्या वेळी पक्ष्यांचे वय', 'Sale Price Per Bird': 'प्रति पक्षी विक्री किंमत',
                'Total Revenue': 'एकूण उत्पन्न', 'Revenue Per Bird Alive': 'जिवंत पक्षी प्रति उत्पन्न', 'Feed Cost Per Bird': 'प्रति पक्षी खाद्य खर्च',
                'Total Fixed Costs': 'एकूण स्थिर खर्च', 'Total Variable Costs': 'एकूण बदलणारा खर्च', 'Total Costs': 'एकूण खर्च',
                'Cost Per Bird': 'प्रति पक्षी खर्च', 'Survival Rate': 'जगण्याचे प्रमाण', 'Revenue Cost Ratio': 'उत्पन्न खर्च प्रमाण',
                'Feed Efficiency': 'खाद्य कार्यक्षमता', 'Profit Margin': 'नफा मार्जिन'
            }
        }
    }
def create_enhanced_multilingual_pdf(lang_code, prediction_result, target_label, feature_impacts, 
                                   train_r2=None, test_r2=None, model_name=None):

    fonts_registered = register_pdf_fonts()
    
    if lang_code in ['hi', 'mr']:
        if fonts_registered.get('NotoSansDevanagari', False):
            base_font = 'NotoSansDevanagari'
            bold_font = 'NotoSansDevanagari'
        else:
            base_font = 'Helvetica'
            bold_font = 'Helvetica-Bold'
    else:
        if fonts_registered.get('NotoSans', False):
            base_font = 'NotoSans'
            bold_font = 'NotoSans-Bold' if fonts_registered.get('NotoSans-Bold', False) else 'NotoSans'
        else:
            base_font = 'Helvetica'
            bold_font = 'Helvetica-Bold'
    
    translations = get_content_translations()
    content = translations.get(lang_code, translations['en'])
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                          topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1,  # Center
        fontName=bold_font,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'], 
        fontSize=14,
        spaceAfter=15,
        fontName=bold_font,
        textColor=colors.darkgreen
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        fontName=base_font
    )
    
    raw_impacts = {}
    friendly_impacts = {}
    if feature_impacts:
        # Check the format of the incoming dictionary
        first_value = next(iter(feature_impacts.values()))
        is_friendly_format = isinstance(first_value, dict)

        if is_friendly_format:
            # If data is already friendly, derive the raw version from it
            friendly_impacts = feature_impacts
            raw_impacts = {k: v.get('raw_impact', 0) for k, v in friendly_impacts.items()}
        else:
            # If data is raw, derive the friendly version
            raw_impacts = feature_impacts
            friendly_impacts = convert_shap_to_farmer_friendly(raw_impacts, target_label)
    
    story = []
    
    # Title
    story.append(Paragraph(content['title'], title_style))
    story.append(Spacer(1, 20))
    
    # Prediction Results Section
    story.append(Paragraph(content['prediction_section'], heading_style))
    pred_text = f"Predicted {target_label}: {prediction_result:,.2f}"
    story.append(Paragraph(pred_text, normal_style))
    
    # Model performance if available
    if train_r2 is not None and test_r2 is not None:
        perf_text = f"Model: {model_name} | Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}"
        story.append(Paragraph(perf_text, normal_style))
    
    story.append(Spacer(1, 20))
    
    # Feature Impact Chart
    story.append(Paragraph(content['shap_section'], heading_style))
    
    # Create and add SHAP chart
    shap_chart = create_shap_chart_matplotlib(feature_impacts, target_label)
    if shap_chart:
        try:
            chart_img = Image(shap_chart, width=6*inch, height=3.6*inch)
            story.append(chart_img)
        except Exception as e:
            story.append(Paragraph(f"Chart generation error: {str(e)[:100]}", normal_style))
    
    story.append(Spacer(1, 20))
    
    # ROI Trend Chart (if applicable)
    if "ROI" in target_label and 'data' in globals():
        story.append(Paragraph(content['roi_trend_section'], heading_style))
        roi_chart = create_roi_trend_matplotlib(data)
        if roi_chart:
            try:
                roi_img = Image(roi_chart, width=6*inch, height=3.6*inch)
                story.append(roi_img)
            except Exception as e:
                story.append(Paragraph(f"ROI chart error: {str(e)[:100]}", normal_style))
        story.append(Spacer(1, 20))
    
    if feature_impacts:
        feature_trans_dict = content.get('feature_translations', {})
        def translate_feature_name(name):
            english_name = name.replace('_', ' ').title()
            return feature_trans_dict.get(english_name, english_name)
        friendly_impacts = convert_shap_to_farmer_friendly(feature_impacts, target_label)
        feature_translations = {
            'en': {k: translate_feature_name(k) for k in friendly_impacts.keys()},
            'hi': {
                'Number of Birds': 'पक्षियों की संख्या',
                'Cost per Chick': 'चूजे की लागत',
                'Feed Cost Total': 'कुल फीड लागत', 
                'Labor Cost': 'श्रम लागत',
                'Rent Cost': 'किराया लागत',
                'Medicine Cost': 'दवा लागत',
                'Land Cost': 'भूमि लागत',
                'Infrastructure Cost': 'ढांचा लागत',
                'Equipment Cost': 'उपकरण लागत',
                'Utilities Cost': 'उपयोगिताएं लागत',
                'Feed Conversion Ratio': 'फीड रूपांतरण अनुपात',
                'Mortality Rate': 'मृत्यु दर',
                'Manure Sales': 'खाद बिक्री',
                'Age of Birds at Sale': 'बिक्री के समय पक्षियों की आयु',
                'Sale Price per Bird': 'प्रति पक्षी बिक्री मूल्य',
                'Total Revenue': 'कुल आय',
                'Revenue per Bird Alive': 'जीवित पक्षी प्रति आय',
                'Feed Cost per Bird': 'प्रति पक्षी फीड लागत',
                'Total Fixed Costs': 'कुल निश्चित लागत',
                'Total Variable Costs': 'कुल परिवर्तनीय लागत',
                'Total Costs': 'कुल लागत',
                'Cost per Bird': 'प्रति पक्षी लागत',
                'Survival Rate': 'उत्तरजीविता दर',
                'Revenue Cost Ratio': 'आय लागत अनुपात',
                'Feed Efficiency': 'फीड दक्षता',
                'Profit Margin': 'लाभ मार्जिन'
            },
            'mr': {
                'Number of Birds': 'पक्ष्यांची संख्या',
                'Cost per Chick': 'चिमुकल्याची किंमत',
                'Feed Cost Total': 'एकूण खाद्य खर्च',
                'Labor Cost': 'कामगार खर्च', 
                'Rent Cost': 'भाडे खर्च',
                'Medicine Cost': 'औषध खर्च',
                'Land Cost': 'जमिनीचा खर्च',
                'Infrastructure Cost': 'पायाभूत सुविधा खर्च',
                'Equipment Cost': 'उपकरण खर्च',
                'Utilities Cost': 'उपयोगिता खर्च',
                'Feed Conversion Ratio': 'खाद्य रूपांतरण प्रमाण',
                'Mortality Rate': 'मृत्यू दर',
                'Manure Sales': 'खत विक्री',
                'Age of Birds at Sale': 'विक्रीच्या वेळी पक्ष्यांचे वय',
                'Sale Price per Bird': 'प्रति पक्षी विक्री किंमत',
                'Total Revenue': 'एकूण उत्पन्न',
                'Revenue per Bird Alive': 'जिवंत पक्षी प्रति उत्पन्न',
                'Feed Cost per Bird': 'प्रति पक्षी खाद्य खर्च',
                'Total Fixed Costs': 'एकूण स्थिर खर्च',
                'Total Variable Costs': 'एकूण बदलणारे खर्च',
                'Total Costs': 'एकूण खर्च',
                'Cost per Bird': 'प्रति पक्षी खर्च',
                'Survival Rate': 'जगण्याचे प्रमाण',
                'Revenue Cost Ratio': 'उत्पन्न खर्च प्रमाण',
                'Feed Efficiency': 'खाद्य कार्यक्षमता',
                'Profit Margin': 'नफा मार्जिन'
            }
        }
        
        feat_trans = feature_translations.get(lang_code, {})
        
        # Positive impacts
        story.append(Paragraph(content['positive_impacts'], heading_style))
        positive_features = sorted([(f, d) for f, d in friendly_impacts.items() if d['direction'] == 'Positive'], key=lambda item: item[1]['score'], reverse=True)
        if positive_features:
            pos_data = [[content['feature_name'], content['impact_percentage']]]
            # This loop now correctly accesses the 'score' from the 'data' dictionary
            for feat, data in positive_features[:8]:
                pos_data.append([translate_feature_name(feat), f"{data['score']:.0f}%"])
            
            pos_table = Table(pos_data, colWidths=[3.5*inch, 2*inch])
            pos_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), base_font),
                ('FONTNAME', (0, 0), (-1, 0), bold_font),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            story.append(pos_table)
        else:
            story.append(Paragraph(content['no_positive'], normal_style))
        story.append(Spacer(1, 15))

        # Negative impacts table
        story.append(Paragraph(content['negative_impacts'], heading_style))
        negative_features = sorted([(f, d) for f, d in friendly_impacts.items() if d['direction'] == 'Negative'], key=lambda item: item[1]['score'], reverse=True)
        if negative_features:
            neg_data = [[content['feature_name'], content['impact_percentage']]]
            # This loop now correctly accesses the 'score' from the 'data' dictionary
            for feat, data in negative_features[:8]:
                neg_data.append([translate_feature_name(feat), f"{data['score']:.0f}%"])

            neg_table = Table(neg_data, colWidths=[3.5*inch, 2*inch])
            neg_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightcoral),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), base_font),
                ('FONTNAME', (0, 0), (-1, 0), bold_font),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            story.append(neg_table)
        else:
            story.append(Paragraph(content['no_negative'], normal_style))
        story.append(Spacer(1, 20))
    
    # AI Suggestions Section
    story.append(Paragraph(content['suggestions_section'], heading_style))
    
    # Get AI suggestions
    lang_map = {'en': 'English', 'hi': 'Hindi', 'mr': 'Marathi'}
    suggestions = generate_ai_suggestions(
        target_label, prediction_result, feature_impacts,
        lang_map.get(lang_code, 'English')
    )
    
    # Display suggestions with proper encoding
    for suggestion in suggestions:
        # Clean suggestion for PDF (remove emojis and markdown)
        clean_suggestion = suggestion
        # Remove common emojis
        emoji_replacements = {
            '🎉': '✓', '⚠️': '!', '✅': '✓', '🔴': '●', '🟢': '●', '🟡': '●',
            '📉': '', '📈': '', '**': '', '●●': '●'
        }
        
        for emoji, replacement in emoji_replacements.items():
            clean_suggestion = clean_suggestion.replace(emoji, replacement)
        
        # Create bullet point with proper spacing
        bullet_para = Paragraph(f"• {clean_suggestion}", normal_style)
        story.append(bullet_para)
        story.append(Spacer(1, 8))
    
    # If no suggestions generated, add a fallback
    if not suggestions:
        fallback_suggestions = {
            'en': ["Focus on reducing major cost drivers", "Monitor feed conversion efficiency", "Improve biosecurity measures"],
            'hi': ["प्रमुख लागत कारकों को कम करने पर ध्यान दें", "फीड रूपांतरण दक्षता की निगरानी करें", "जैव सुरक्षा उपायों में सुधार करें"],
            'mr': ["मुख्य खर्च कारणांवर लक्ष द्या", "खाद्य रूपांतरण कार्यक्षमतेवर लक्ष ठेवा", "जैवसुरक्षा उपायांमध्ये सुधारणा करा"]
        }
        
        fallback_list = fallback_suggestions.get(lang_code, fallback_suggestions['en'])
        for suggestion in fallback_list:
            story.append(Paragraph(f"• {suggestion}", normal_style))
            story.append(Spacer(1, 8))
    
    # Footer
    story.append(Spacer(1, 30))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer_text = f"{content['generated_on']}: {timestamp}"
    story.append(Paragraph(footer_text, normal_style))
    
    # Build PDF
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return create_simple_fallback_pdf(prediction_result, target_label, lang_code)

def create_simple_fallback_pdf(prediction_result, target_label, lang_code):
    """Simple fallback PDF if main generation fails"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    lang_names = {'en': 'English', 'hi': 'Hindi', 'mr': 'Marathi'}
    lang_name = lang_names.get(lang_code, 'English')
    
    story.append(Paragraph(f"Poultry Analysis Report ({lang_name})", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Predicted {target_label}: {prediction_result:,.2f}", styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Simplified report generated due to technical limitations.", styles['Normal']))
    story.append(Paragraph("Please contact support for full report generation.", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def add_pdf_download_section():
    """Add PDF download section to Streamlit app"""
    if st.session_state.get('prediction_made', False):
        st.markdown("---")
        st.subheader("📄 Download Comprehensive Report")
        
        if st.button("Generate PDF Report", key="generate_pdf"):
            lang_code = {'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr'}[language_choice]
            
            with st.spinner("Generating comprehensive PDF report..."):
                try:
                    pdf_bytes = create_enhanced_multilingual_pdf(
                        lang_code=lang_code,
                        prediction_result=st.session_state.pred_result,
                        target_label=st.session_state.target_label,
                        feature_impacts=st.session_state.feat_shap,
                        train_r2=st.session_state.get('train_r2'),
                        test_r2=st.session_state.get('test_r2'),
                        model_name=st.session_state.get('model_name', 'Unknown')
                    )
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=pdf_bytes,
                        file_name=f"poultry_analysis_{lang_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )
                    
                    st.success("✅ Report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("Please try again or contact support if the issue persists.")
add_pdf_download_section()