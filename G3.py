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
import os
import google.generativeai as genai
import warnings
import joblib
import pickle
import time
from plotly.subplots import make_subplots
import plotly.express as px

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("GEMINI_API_KEY not found. Please add it to your .streamlit/secrets.toml file.")
    st.stop()
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
if 'data_engineered' not in st.session_state:
    st.session_state.data_engineered = None
if 'kpi_data' not in st.session_state:
    st.session_state.kpi_data = None
if 'pretrained_models' not in st.session_state:
    st.session_state.pretrained_models = None
if 'model_training_complete' not in st.session_state:
    st.session_state.model_training_complete = False
if 'best_models' not in st.session_state:
    st.session_state.best_models = {}
if 'whatif_scenarios' not in st.session_state:
    st.session_state.whatif_scenarios = {}
if 'whatif_baseline' not in st.session_state:
    st.session_state.whatif_baseline = None
if 'whatif_results' not in st.session_state:
    st.session_state.whatif_results = {}
    
# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="GainSight AI – Smarter Business Predictions", 
    page_icon="📊", 
    layout="wide"
)

st.title("📊 GainSight AI – Smarter Business Predictions")
st.write("**Advanced ML predictions with feature engineering and comprehensive model evaluation for any business domain.**")

class MultilingualPDFHandler:
    def __init__(self, language):
        self.language = language
        self.font_registered = False
        self.base_font = 'Helvetica'
        self.bold_font = 'Helvetica-Bold'
        self.styles = getSampleStyleSheet()

        self.register_devanagari_fonts()
        self.create_multilingual_styles()

# ===========================
# PRE-TRAINING SECTION
# ===========================
def create_dummy_dataset():
    """Create a comprehensive dummy business dataset for training"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate dummy business data
    data = {
        'Revenue': np.random.normal(100000, 30000, n_samples),
        'Marketing_Spend': np.random.normal(15000, 5000, n_samples),
        'R_D_Investment': np.random.normal(8000, 3000, n_samples),
        'Employee_Count': np.random.randint(10, 500, n_samples),
        'Units_Sold': np.random.randint(100, 5000, n_samples),
        'Customer_Satisfaction': np.random.uniform(1, 5, n_samples),
        'Market_Share': np.random.uniform(0.01, 0.3, n_samples),
        'Product_Price': np.random.uniform(10, 200, n_samples),
        'Competition_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Season': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], n_samples),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'Year': np.random.choice([2021, 2022, 2023, 2024], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic relationships
    df['Operating_Costs'] = (df['Revenue'] * 0.6 + 
                            df['Marketing_Spend'] * 0.8 + 
                            df['Employee_Count'] * 100 + 
                            np.random.normal(0, 5000, n_samples))
    
    df['Net_Profit'] = (df['Revenue'] - df['Operating_Costs'] + 
                       np.random.normal(0, 8000, n_samples))
    
    # Ensure some realistic constraints
    df['Operating_Costs'] = np.clip(df['Operating_Costs'], 0, df['Revenue'] * 0.9)
    df['Revenue'] = np.clip(df['Revenue'], 10000, 500000)
    df['Marketing_Spend'] = np.clip(df['Marketing_Spend'], 1000, 50000)
    
    return df

def get_hyperparameter_grids():
    """Define hyperparameter grids for different models"""
    param_grids = {
        'LinearRegression': {
            'model__fit_intercept': [True, False],
            'model__positive': [False, True]
        },
        'RandomForest': {
            'model__n_estimators': [50, 100, 200, 300],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None]
        },
        'XGBoost': {
            'model__n_estimators': [50, 100, 200, 300],
            'model__max_depth': [3, 6, 9, 12],
            'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'model__subsample': [0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.8, 0.9, 1.0],
            'model__reg_alpha': [0, 0.1, 1],
            'model__reg_lambda': [0, 0.1, 1]
        }
    }
    return param_grids

def train_models_with_hyperparameter_tuning(X, y, cv_folds=5, n_iter=20):
    """Train multiple models with hyperparameter tuning"""
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    }
    
    param_grids = get_hyperparameter_grids()
    best_models = {}
    model_results = {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (model_name, base_model) in enumerate(models.items()):
        status_text.text(f"Training {model_name} with hyperparameter tuning...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])
        
        # Hyperparameter tuning
        if model_name in param_grids:
            search = RandomizedSearchCV(
                pipeline,
                param_grids[model_name],
                n_iter=n_iter,
                cv=cv_folds,
                scoring='r2',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            cv_score = search.best_score_
        else:
            # For models without hyperparameters
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
            cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='r2').mean()
        
        # Evaluate on test set
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        metrics = {
            'cv_score': cv_score,
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'best_params': best_params
        }
        metrics['overfitting_score'] = metrics['train_r2'] - metrics['test_r2']
        
        best_models[model_name] = best_model
        model_results[model_name] = metrics
        
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.text("Model training completed! ✅")
    return best_models, model_results

def display_model_comparison(model_results):
    """Display comparison of trained models"""
    st.subheader("🏆 Pre-trained Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'CV Score': f"{metrics['cv_score']:.4f}",
            'Test R²': f"{metrics['test_r2']:.4f}",
            'Test MAE': f"{metrics['test_mae']:.2f}",
            'Test RMSE': f"{metrics['test_rmse']:.2f}",
            'Overfitting': f"{metrics['overfitting_score']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualize model performance
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score comparison
        r2_scores = [model_results[model]['test_r2'] for model in model_results.keys()]
        model_names = list(model_results.keys())
        
        fig_r2 = go.Figure(data=[
            go.Bar(x=model_names, y=r2_scores, 
                  marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ])
        fig_r2.update_layout(
            title="📈 Test R² Score Comparison",
            xaxis_title="Models",
            yaxis_title="R² Score",
            height=400
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        # MAE comparison
        mae_scores = [model_results[model]['test_mae'] for model in model_results.keys()]
        
        fig_mae = go.Figure(data=[
            go.Bar(x=model_names, y=mae_scores,
                  marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ])
        fig_mae.update_layout(
            title="📉 Mean Absolute Error Comparison",
            xaxis_title="Models",
            yaxis_title="MAE",
            height=400
        )
        st.plotly_chart(fig_mae, use_container_width=True)

# ===========================
# PRE-TRAINING EXECUTION
# ===========================
def train_models_with_hyperparameter_tuning(X, y, cv_folds=5, n_iter=20):
    """Train multiple models with hyperparameter tuning (silent version)"""
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    }
    
    param_grids = get_hyperparameter_grids()
    best_models = {}
    model_results = {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Remove progress indicators - train silently
    for model_name, base_model in models.items():
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])
        
        # Hyperparameter tuning
        if model_name in param_grids:
            search = RandomizedSearchCV(
                pipeline,
                param_grids[model_name],
                n_iter=n_iter,
                cv=cv_folds,
                scoring='r2',
                random_state=42,
                n_jobs=-1,
                verbose=0  # Silent operation
            )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            cv_score = search.best_score_
        else:
            # For models without hyperparameters
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
            cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='r2').mean()
        
        # Evaluate on test set
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        metrics = {
            'cv_score': cv_score,
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'best_params': best_params
        }
        metrics['overfitting_score'] = metrics['train_r2'] - metrics['test_r2']
        
        best_models[model_name] = best_model
        model_results[model_name] = metrics
    
    return best_models, model_results

# ===========================
# MULTILINGUAL TRANSLATION SYSTEM
# ===========================
# Global TRANSLATIONS dictionary for multilingual support
TRANSLATIONS = {
    "English": {
        # Report titles and headers
        "business_intelligence_report": "Business Intelligence Analysis Report",
        "executive_summary": "Executive Summary", 
        "detailed_analysis": "Detailed Analysis",
        "technical_report": "Technical Report",
        "visual_analysis_dashboard": "Visual Analysis Dashboard",
        "ai_powered_recommendations": "AI-Powered Strategic Recommendations",
        "expected_financial_impact": "Expected Financial Impact",
        "key_performance_indicators": "Key Performance Indicators",
        "critical_success_factors": "Critical Success Factors",
        "analysis_focus": "Analysis Focus",
        "key_insight": "Key Insight",
        "recommendation": "Recommendation",
        "generated_in_ai": "Generated in {language} by AI, displayed in English",
    
        # Chart titles
        "financial_waterfall_analysis": "Financial Waterfall Analysis",
            "profitability_trend_analysis": "Profitability Trend Analysis", 
        "breakeven_analysis": "Break-Even Analysis",
        "segment_profitability_analysis": "Segment Profitability Analysis",
        "roi_performance_distribution": "ROI Performance Distribution",
        "cumulative_roi_analysis": "Cumulative ROI Analysis",
        "roi_by_project_analysis": "ROI by Project Analysis",
        "roi_vs_risk_analysis": "ROI vs Risk Analysis",
    
        # Table headers
        "metric": "Metric", "value": "Value", "interpretation": "Interpretation",
        "rank": "Rank", "success_factor": "Success Factor", "impact_score": "Impact Score", "priority": "Priority",
        "current": "Current", "projected": "Projected", "improvement": "Improvement",
    
        # FIXED: Metric Names - using exact keys from KPI calculation
        "totalrevenue": "Total Revenue", 
        "totalprofit": "Total Profit", 
        "totalloss": "Total Loss",
        "totalebit": "Total EBIT", 
        "totalgrossprofit": "Total Gross Profit", 
        "averageroi": "Average ROI",
        "profitmargin": "Profit Margin", 
        "revenueimpact": "Revenue Impact", 
        "additionalprofit": "Additional Profit",
        
        # Add to Marathi section:
        "revenue": "महसूल",
        "marketing_spend": "मार्केटिंग खर्च",
        "employee_count": "कर्मचारी संख्या",
        "units_sold": "विकली गेलेली युनिट्स", 
        "customer_satisfaction": "ग्राहक समाधान",
        "market_share": "बाजार वाटा",
        "product_price": "उत्पाद किंमत",
        "revenue_per_unit": "प्रति युनिट महसूल",
        "cost_per_unit": "प्रति युनिट किंमत", 
        "roi": "आरओआई",
        "operating_costs": "ऑपरेटिंग खर्च",

        # Status and interpretations
        "above_average": "Above Average", "needs_improvement": "Needs Improvement", "strong": "Strong",
        "moderate": "Moderate", "weak": "Weak", "positive": "Positive", "negative": "Negative",
        "high": "High", "medium": "Medium", "normal": "Normal",
    
        # Metadata labels
        "business_context": "Business Context", "analysis_type": "Analysis Type", "target_variable": "Target Variable",
        "report_language": "Report Language", "generated_on": "Generated On",
    
        # Notes and messages
        "multilingual_note": "Note: This report was generated for {language} language. Due to PDF font limitations, content is displayed in English with {language} AI insights included.",
        "no_insight_available": "No specific insight available."
    },

    "Hindi": {
        # Report titles and headers  
        "business_intelligence_report": "व्यावसायिक बुद्धिमत्ता विश्लेषण रिपोर्ट",
        "executive_summary": "कार्यकारी सारांश", "detailed_analysis": "विस्तृत विश्लेषण", 
        "technical_report": "तकनीकी रिपोर्ट", "visual_analysis_dashboard": "दृश्य विश्लेषण डैशबोर्ड",
        "ai_powered_recommendations": "AI-संचालित रणनीतिक सिफारिशें", "expected_financial_impact": "अपेक्षित वित्तीय प्रभाव",
        "key_performance_indicators": "मुख्य प्रदर्शन संकेतक", "critical_success_factors": "महत्वपूर्ण सफलता कारक",
        "analysis_focus": "विश्लेषण फोकस", "key_insight": "मुख्य अंतर्दृष्टि", "recommendation": "सिफारिश",
        "generated_in_ai": "{language} में AI द्वारा जेनरेट किया गया, अंग्रेजी में प्रदर्शित",
    
        # Chart titles
        "financial_waterfall_analysis": "वित्तीय वॉटरफॉल विश्लेषण", "profitability_trend_analysis": "लाभप्रदता रुझान विश्लेषण",
        "breakeven_analysis": "ब्रेक-ईवन विश्लेषण", "segment_profitability_analysis": "खंड लाभप्रदता विश्लेषण",
        "roi_performance_distribution": "ROI प्रदर्शन वितरण", "cumulative_roi_analysis": "संचयी ROI विश्लेषण",
        "roi_by_project_analysis": "परियोजना द्वारा ROI विश्लेषण", "roi_vs_risk_analysis": "ROI बनाम जोखिम विश्लेषण",
    
        # Table headers
        "metric": "मेट्रिक", "value": "मूल्य", "interpretation": "व्याख्या", 
        "rank": "रैंक", "success_factor": "सफलता कारक", "impact_score": "प्रभाव स्कोर", "priority": "प्राथमिकता",
        "current": "वर्तमान", "projected": "अनुमानित", "improvement": "सुधार",
    
        # FIXED: Metric Names - Hindi translations
        "totalrevenue": "कुल राजस्व", 
        "totalprofit": "कुल लाभ", 
        "totalloss": "कुल हानि",
        "totalebit": "कुल ईबिट", 
        "totalgrossprofit": "कुल सकल लाभ", 
        "averageroi": "औसत आरओआई",
        "profitmargin": "लाभ मार्जिन", 
        "revenueimpact": "राजस्व प्रभाव", 
        "additionalprofit": "अतिरिक्त लाभ",
        
        # Add to Hindi section:
        "revenue": "राजस्व",
        "marketing_spend": "मार्केटिंग खर्च",
        "employee_count": "कर्मचारी संख्या", 
        "units_sold": "बेची गई इकाइयां",
        "customer_satisfaction": "ग्राहक संतुष्टि",
        "market_share": "बाजार हिस्सा",
        "product_price": "उत्पाद मूल्य",
        "revenue_per_unit": "प्रति यूनिट राजस्व",
        "cost_per_unit": "प्रति यूनिट लागत",
        "roi": "आरओआई",
        "operating_costs": "परिचालन लागत",

        # Status and interpretations
        "above_average": "औसत से ऊपर", "needs_improvement": "सुधार की आवश्यकता", "strong": "मजबूत",
        "moderate": "मध्यम", "weak": "कमजोर", "positive": "सकारात्मक", "negative": "नकारात्मक",
        "high": "उच्च", "medium": "मध्यम", "normal": "सामान्य",
    
        # Metadata labels
        "business_context": "व्यावसायिक संदर्भ", "analysis_type": "विश्लेषण प्रकार", "target_variable": "लक्ष्य चर",
        "report_language": "रिपोर्ट भाषा", "generated_on": "जेनरेट किया गया",
    
        # Notes and messages
        "multilingual_note": "नोट: यह रिपोर्ट {language} भाषा के लिए जेनरेट की गई थी। PDF फॉन्ट सीमाओं के कारण, सामग्री {language} AI अंतर्दृष्टि के साथ अंग्रेजी में प्रदर्शित की जाती है।",
        "no_insight_available": "कोई विशिष्ट अंतर्दृष्टि उपलब्ध नहीं।"
    },

    "Marathi": {
        # Report titles and headers
        "business_intelligence_report": "व्यावसायिक बुद्धिमत्ता विश्लेषण अहवाल",
        "executive_summary": "कार्यकारी सारांश", "detailed_analysis": "तपशीलवार विश्लेषण",
        "technical_report": "तांत्रिक अहवाल", "visual_analysis_dashboard": "दृश्य विश्लेषण डॅशबोर्ड",
        "ai_powered_recommendations": "AI-चालित धोरणात्मक शिफारसी", "expected_financial_impact": "अपेक्षित आर्थिक प्रभाव",
        "key_performance_indicators": "मुख्य कामगिरी निर्देशक", "critical_success_factors": "गंभीर यश घटक", 
        "analysis_focus": "विश्लेषण फोकस", "key_insight": "मुख्य अंतर्दृष्टी", "recommendation": "शिफारस",
        "generated_in_ai": "{language} मध्ये AI द्वारे व्युत्पन्न, इंग्रजीमध्ये प्रदर्शित",
    
        # Chart titles
        "financial_waterfall_analysis": "आर्थिक वॉटरफॉल विश्लेषण", "profitability_trend_analysis": "नफा ट्रेंड विश्लेषण",
        "breakeven_analysis": "ब्रेक-इव्हन विश्लेषण", "segment_profitability_analysis": "विभाग नफा विश्लेषण", 
        "roi_performance_distribution": "ROI कामगिरी वितरण", "cumulative_roi_analysis": "संचयी ROI विश्लेषण",
        "roi_by_project_analysis": "प्रकल्पानुसार ROI विश्लेषण", "roi_vs_risk_analysis": "ROI विरुद्ध जोखीम विश्लेषण",
    
        # Table headers
        "metric": "मेट्रिक", "value": "मूल्य", "interpretation": "व्याख्या",
        "rank": "रँक", "success_factor": "यश घटक", "impact_score": "प्रभाव स्कोअर", "priority": "प्राथमिकता",
        "current": "सध्याचे", "projected": "अंदाजित", "improvement": "सुधारणा",
    
        # FIXED: Metric Names - Marathi translations
        "totalrevenue": "एकूण महसूल", 
        "totalprofit": "एकूण नफा", 
        "totalloss": "एकूण तोटा",
        "totalebit": "एकूण ईबिट", 
        "totalgrossprofit": "एकूण सकल नफा", 
        "averageroi": "सरासरी ROI",
        "profitmargin": "नफा मार्जिन", 
        "revenueimpact": "महसूल प्रभाव", 
        "additionalprofit": "अतिरिक्त नफा",
        
        # Status and interpretations
        "above_average": "सरासरीपेक्षा जास्त", "needs_improvement": "सुधारणेची गरज", "strong": "मजबूत",
        "moderate": "मध्यम", "weak": "कमकुवत", "positive": "सकारात्मक", "negative": "नकारात्मक",
        "high": "उच्च", "medium": "मध्यम", "normal": "सामान्य",
    
        # Metadata labels
        "business_context": "व्यावसायिक संदर्भ", "analysis_type": "विश्लेषण प्रकार", "target_variable": "लक्ष्य चल",
        "report_language": "अहवाल भाषा", "generated_on": "व्युत्पन्न केले",
    
        # Notes and messages
        "multilingual_note": "टीप: हा अहवाल {language} भाषेसाठी व्युत्पन्न करण्यात आला होता। PDF फॉन्ट मर्यादांमुळे, सामग्री {language} AI अंतर्दृष्टीसह इंग्रजीमध्ये प्रदर्शित केली जाते.",
        "no_insight_available": "कोणतीही विशिष्ट अंतर्दृष्टी उपलब्ध नाही."
    }
}

def get_metric_name_translation(key, language):
    """Get metric name in specified language"""
    return TRANSLATIONS.get(language, {}).get(key, key.replace('_', ' '))

def get_translation(key, language="English", **kwargs):
    """Get translated text for a given key and language"""
    try:
        if language in TRANSLATIONS and key in TRANSLATIONS[language]:
            text = TRANSLATIONS[language][key]
            # Handle string formatting if kwargs provided
            if kwargs:
                return text.format(**kwargs)
            return text
        else:
            # Fallback to English
            text = TRANSLATIONS["English"].get(key, key)
            if kwargs:
                return text.format(**kwargs)
            return text
    except Exception as e:
        # Emergency fallback
        return TRANSLATIONS["English"].get(key, key)

def translate_chart_title(english_title, language):
    """Translate chart titles to the specified language"""
    
    title_mappings = {
        "Financial Waterfall Analysis": "financial_waterfall_analysis",
        "Profitability Trend Analysis": "profitability_trend_analysis", 
        "Break-Even Analysis": "breakeven_analysis",
        "Segment Profitability Analysis": "segment_profitability_analysis",
        "ROI Performance Distribution": "roi_performance_distribution",
        "Cumulative ROI Analysis": "cumulative_roi_analysis", 
        "ROI by Project Analysis": "roi_by_project_analysis",
        "ROI vs Risk Analysis": "roi_vs_risk_analysis"
    }
    
    # Find the mapping key
    for title, mapping_key in title_mappings.items():
        if title.lower() in english_title.lower():
            return get_translation(mapping_key, language)
    
    # If no mapping found, return original
    return english_title

def translate_analysis_type(analysis_type, language):
    """Translate analysis type labels"""
    if "Profit & Loss" in analysis_type or "P/L" in analysis_type:
        if language == "Hindi":
            return "लाभ और हानि (P/L)"
        elif language == "Marathi": 
            return "नफा आणि तोटा (P/L)"
    elif "Return on Investment" in analysis_type or "ROI" in analysis_type:
        if language == "Hindi":
            return "निवेश पर रिटर्न (ROI)"
        elif language == "Marathi":
            return "गुंतवणुकीवरील परतावा (ROI)"
    
    return analysis_type

def translate_business_context(business_context, language):
    """Translate business context"""
    business_translations = {
        "English": {
            "General Business Analysis": "General Business Analysis",
            "Sales & Revenue Analysis": "Sales & Revenue Analysis", 
            "Marketing ROI Analysis": "Marketing ROI Analysis",
            "Financial Performance": "Financial Performance",
            "Investment Analysis": "Investment Analysis"
        },
        "Hindi": {
            "General Business Analysis": "सामान्य व्यावसायिक विश्लेषण",
            "Sales & Revenue Analysis": "बिक्री और राजस्व विश्लेषण",
            "Marketing ROI Analysis": "मार्केटिंग ROI विश्लेषण", 
            "Financial Performance": "वित्तीय प्रदर्शन",
            "Investment Analysis": "निवेश विश्लेषण"
        },
        "Marathi": {
            "General Business Analysis": "सामान्य व्यावसायिक विश्लेषण",
            "Sales & Revenue Analysis": "विक्री आणि महसूल विश्लेषण",
            "Marketing ROI Analysis": "मार्केटिंग ROI विश्लेषण",
            "Financial Performance": "आर्थिक कामगिरी", 
            "Investment Analysis": "गुंतवणूक विश्लेषण"
        }
    }
    
    return business_translations.get(language, {}).get(business_context, business_context)

def get_interpretation_text(key, value, language):
    """Get interpretation text in the specified language - FIXED VERSION"""
    
    # Debug print to check inputs
    print(f"DEBUG: Interpreting key='{key}', value={value}, language='{language}'")
    
    try:
        if "profit" in key.lower() or "revenue" in key.lower():
            if value > 100000:  # Adjust threshold for large numbers
                interpretation_key = "above_average"
            elif value > 0:
                interpretation_key = "positive"
            else:
                interpretation_key = "needs_improvement"
        elif "roi" in key.lower() or "margin" in key.lower():
            if value > 15:
                interpretation_key = "strong"
            elif value > 5:
                interpretation_key = "moderate"
            elif value > 0:
                interpretation_key = "weak"
            else:
                interpretation_key = "negative"
        elif "loss" in key.lower():
            if value > 0:
                interpretation_key = "negative"  # Loss is bad
            else:
                interpretation_key = "positive"  # No loss is good
        else:
            interpretation_key = "positive" if value > 0 else "negative"
        
        result = get_translation(interpretation_key, language)
        print(f"DEBUG: Interpretation result: '{result}'")
        return result
        
    except Exception as e:
        print(f"ERROR in get_interpretation_text: {e}")
        # Fallback
        return get_translation("normal", language)

def get_priority_text(rank, language):
    """Get priority text in specified language"""
    if rank < 2:
        return get_translation("high", language)
    elif rank < 4:
        return get_translation("medium", language)
    else:
        return get_translation("normal", language)

# Function to use in main PDF generation
def create_translated_metadata_table(business_context, analysis_type, target_variable, language):
    """Create metadata table with complete translations - FINAL FIX"""
    
    print(f"DEBUG METADATA: Inputs - context: {business_context}, analysis: {analysis_type}, target: {target_variable}, lang: {language}")
    
    # Ensure all inputs are valid strings with fallbacks
    safe_business_context = str(business_context) if business_context else "General Business Analysis"
    safe_analysis_type = str(analysis_type) if analysis_type else "Business Analysis"
    safe_target_variable = str(target_variable).replace('_', ' ').title() if target_variable else "Business Metric"
    safe_language = str(language) if language else "English"
    
    # FIXED: Translate business context properly
    if safe_business_context == "General Business Analysis":
        if language == "Hindi":
            translated_business_context = "सामान्य व्यावसायिक विश्लेषण"
        elif language == "Marathi":
            translated_business_context = "सामान्य व्यावसायिक विश्लेषण"
        else:
            translated_business_context = safe_business_context
    else:
        translated_business_context = translate_business_context(safe_business_context, language)
    
    # FIXED: Translate analysis type properly
    if "Profit & Loss" in safe_analysis_type or "P/L" in safe_analysis_type:
        if language == "Hindi":
            translated_analysis_type = "लाभ और हानि (P/L) विश्लेषण"
        elif language == "Marathi":
            translated_analysis_type = "नफा आणि तोटा (P/L) विश्लेषण"
        else:
            translated_analysis_type = safe_analysis_type
    elif "ROI" in safe_analysis_type or "Return on Investment" in safe_analysis_type:
        if language == "Hindi":
            translated_analysis_type = "निवेश पर रिटर्न (ROI) विश्लेषण"
        elif language == "Marathi":
            translated_analysis_type = "गुंतवणुकीवरील परतावा (ROI) विश्लेषण"
        else:
            translated_analysis_type = safe_analysis_type
    else:
        translated_analysis_type = safe_analysis_type
    
    # FIXED: Translate target variable properly
    target_translations = {
        "Net Profit": {"Hindi": "शुद्ध लाभ", "Marathi": "निव्वळ नफा"},
        "Revenue": {"Hindi": "राजस्व", "Marathi": "महसूल"},
        "ROI": {"Hindi": "आरओआई", "Marathi": "आरओआई"},
        "Total Profit": {"Hindi": "कुल लाभ", "Marathi": "एकूण नफा"},
        "Profit Margin": {"Hindi": "लाभ मार्जिन", "Marathi": "नफा मार्जिन"}
    }
    
    if safe_target_variable in target_translations and language in target_translations[safe_target_variable]:
        translated_target_variable = target_translations[safe_target_variable][language]
    else:
        translated_target_variable = safe_target_variable
    
    # FIXED: Translate language name
    language_translations = {
        "English": {"Hindi": "अंग्रेजी", "Marathi": "इंग्रजी"},
        "Hindi": {"Hindi": "हिंदी", "Marathi": "हिंदी"},
        "Marathi": {"Hindi": "मराठी", "Marathi": "मराठी"}
    }
    
    if safe_language in language_translations and language in language_translations[safe_language]:
        translated_language = language_translations[safe_language][language]
    else:
        translated_language = safe_language
    
    # Generate current timestamp - FIXED for Hindi/Marathi
    if language == "Hindi":
        current_time = datetime.now().strftime('%d %B, %Y को %H:%M')
    elif language == "Marathi":
        current_time = datetime.now().strftime('%d %B, %Y रोजी %H:%M')
    else:
        current_time = datetime.now().strftime('%B %d, %Y at %H:%M')
    
    # Create metadata with guaranteed translated content
    metadata = [
        [get_translation("business_context", language), translated_business_context],
        [get_translation("analysis_type", language), translated_analysis_type], 
        [get_translation("target_variable", language), translated_target_variable],
        [get_translation("report_language", language), translated_language],
        [get_translation("generated_on", language), current_time]
    ]
    
    print(f"DEBUG METADATA: Final metadata with translations: {metadata}")
    return metadata

def format_currency_value(value):
    """Format currency with guaranteed $M/$K notation."""
    try:
        # Step 1: Clean the value if it's a string
        if isinstance(value, str):
            # Remove any commas, spaces, or currency symbols
            cleaned_value = value.replace(",", "").replace(" ", "").replace("$", "")
            # Convert to float
            val = float(cleaned_value)
        else:
            val = float(value)

        abs_val = abs(val)

        if abs_val >= 1_000_000:
            return f"${val/1_000_000:.1f}M"
        elif abs_val >= 1_000:
            return f"${val/1_000:.1f}K" 
        else:
            return f"${val:,.0f}"
    except (ValueError, TypeError):
        # Fallback for invalid inputs
        return "$0"

def create_translated_kpi_table(kpis, language):
    """Create KPI table with translations and correct currency formatting - FIXED VERSION"""
    
    kpi_data = [[
        get_translation("metric", language), 
        get_translation("value", language), 
        get_translation("interpretation", language)
    ]]
    
    # Fixed mapping dictionary
    kpi_translation_mapping = {
        'Total_Revenue': 'totalrevenue',
        'Total_Profit': 'totalprofit', 
        'Total_Loss': 'totalloss',
        'Total_EBIT': 'totalebit',
        'Total_Gross_Profit': 'totalgrossprofit',
        'Average_ROI': 'averageroi',
        'Profit_Margin': 'profitmargin',
        'Revenue_Impact': 'revenueimpact',
        'Additional_Profit': 'additionalprofit'
    }

    for key, value in kpis.items():
        translation_key = kpi_translation_mapping.get(key, key.lower().replace('_', ''))
        metric_name = get_translation(translation_key, language)
        
        if "roi" in key.lower() or "margin" in key.lower():
            formatted_value = f"{value:.1f}%"
        else:
            formatted_value = format_currency_value(value)

        interpretation = get_interpretation_text(key, value, language)
        kpi_data.append([metric_name, formatted_value, interpretation])
    
    return kpi_data

def create_working_metadata_table(business_context, analysis_type, target_variable, language):
    """Create metadata table with translations and correct date format."""
    
    translated_business_context = translate_business_context(business_context, language)
    translated_analysis_type = translate_analysis_type(analysis_type, language)
    
    months = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    now = datetime.now()
    month_name = months.get(now.month, "")
    formatted_date = now.strftime(f"{month_name} %d, %Y at %H:%M")
    
    # FIX: Ensure all values are correctly formatted as strings before passing to paragraph
    metadata = [
        [get_translation("business_context", language), translated_business_context],
        [get_translation("analysis_type", language), translated_analysis_type], 
        [get_translation("target_variable", language), target_variable.replace('_', ' ')],
        [get_translation("report_language", language), language],
        [get_translation("generated_on", language), formatted_date]
    ]
    
    return metadata

def create_translated_factor_table(feature_df, language):
    """Create success factors table with complete translations - FINAL FIX"""
    
    print(f"DEBUG FACTORS: Input feature_df: {feature_df}")
    print(f"DEBUG FACTORS: Language: {language}")
    
    # Handle empty or None feature_df
    if feature_df is None or feature_df.empty or len(feature_df) == 0:
        no_data_message = get_translation("no_insight_available", language)
        return [[no_data_message, "", "", ""]]
    
    # Create header row
    factor_data = [[
        get_translation("rank", language),
        get_translation("success_factor", language), 
        get_translation("impact_score", language),
        get_translation("priority", language)
    ]]
    
    # COMPREHENSIVE factor translation dictionary
    factor_translations = {
        "English": {
            "Revenue": "Revenue",
            "Marketing Spend": "Marketing Spend",
            "Marketing Investment": "Marketing Investment",
            "Employee Count": "Employee Count",
            "Team Size": "Team Size",
            "Units Sold": "Sales Volume",
            "Sales Volume": "Sales Volume",
            "Customer Satisfaction": "Customer Satisfaction",
            "Market Share": "Market Position",
            "Product Price": "Pricing Strategy",
            "Revenue Per Unit": "Unit Revenue",
            "Cost Per Unit": "Unit Cost",
            "Profit Margin": "Profit Margins",
            "ROI": "Return on Investment",
            "Operating Costs": "Operating Expenses",
            "Net Profit": "Net Profit",
            "R D Investment": "R&D Investment"
        },
        "Hindi": {
            "Revenue": "राजस्व",
            "Marketing Spend": "मार्केटिंग खर्च",
            "Marketing Investment": "मार्केटिंग निवेश",
            "Employee Count": "कर्मचारी संख्या",
            "Team Size": "टीम का आकार",
            "Units Sold": "बेची गई इकाइयाँ",
            "Sales Volume": "बिक्री मात्रा",
            "Customer Satisfaction": "ग्राहक संतुष्टि",
            "Market Share": "बाजार हिस्सा",
            "Product Price": "उत्पाद मूल्य",
            "Revenue Per Unit": "प्रति यूनिट राजस्व",
            "Cost Per Unit": "प्रति यूनिट लागत",
            "Profit Margin": "लाभ मार्जिन",
            "ROI": "आरओआई",
            "Operating Costs": "परिचालन लागत",
            "Net Profit": "शुद्ध लाभ",
            "R D Investment": "अनुसंधान विकास निवेश"
        },
        "Marathi": {
            "Revenue": "महसूल",
            "Marketing Spend": "मार्केटिंग खर्च",
            "Marketing Investment": "मार्केटिंग गुंतवणूक",
            "Employee Count": "कर्मचारी संख्या",
            "Team Size": "टीम आकार",
            "Units Sold": "विकली गेलेली युनिट्स",
            "Sales Volume": "विक्री प्रमाण",
            "Customer Satisfaction": "ग्राहक समाधान",
            "Market Share": "बाजार वाटा",
            "Product Price": "उत्पाद किंमत",
            "Revenue Per Unit": "प्रति युनिट महसूल",
            "Cost Per Unit": "प्रति युनिट किंमत",
            "Profit Margin": "नफा मार्जिन",
            "ROI": "आरओआई",
            "Operating Costs": "ऑपरेटिंग खर्च",
            "Net Profit": "निव्वळ नफा",
            "R D Investment": "संशोधन विकास गुंतवणूक"
        }
    }
    
    # Process each factor
    for i in range(min(5, len(feature_df))):
        try:
            row = feature_df.iloc[i]
            
            # --- FIX STARTS HERE ---
            # Get factor name and handle missing/invalid data
            factor_name = row.get('Factor', 'Unknown Factor')
            if pd.isna(factor_name) or not isinstance(factor_name, str) or not factor_name.strip():
                factor_name = "Unknown Factor"
            
            # --- FIX ENDS HERE ---
            
            # Get impact score
            impact_score = float(row.get('Impact_Score', 0.0))
            
            # Translate the factor name
            # Check for the key with proper spacing first
            translated_factor = factor_translations.get(language, {}).get(
                factor_name.replace("_", " "), 
                factor_name.replace("_", " ").title()
            )
            
            # Get priority text
            priority = get_priority_text(i, language)
            
            # Add row to table
            factor_row = [
                f"{i+1}",
                translated_factor, 
                f"{impact_score:.3f}",
                priority
            ]
            
            factor_data.append(factor_row)
            
        except Exception as e:
            print(f"ERROR processing factor {i}: {e}")
            error_factor = get_translation(f"Factor", language) + f" {i+1}"
            factor_data.append([
                f"{i+1}",
                error_factor,
                "0.000",
                get_translation("normal", language)
            ])
    
    return factor_data

# ===========================
# ORIGINAL FUNCTIONS (keeping all existing functions)
# ===========================

def analyze_dataset_features(df):
    """Analyze dataset features and provide insights"""
    st.subheader("🔍 Dataset Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📋 Total Rows", f"{df.shape[0]:,}")
        st.metric("📊 Total Columns", f"{df.shape[1]:,}")
    
    with col2:
        missing_values = df.isnull().sum().sum()
        st.metric("⌀ Missing Values", f"{missing_values:,}")
        duplicates = df.duplicated().sum()
        st.metric("🔄 Duplicate Rows", f"{duplicates:,}")
    
    with col3:
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        st.metric("🔢 Numeric Columns", len(numeric_cols))
        st.metric("🏷 Categorical Columns", len(categorical_cols))
    
    # Display column types and missing values
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    #st.subheader("📋 Column Information")
    #st.dataframe(col_info)
    
    return numeric_cols.tolist(), categorical_cols.tolist()

def auto_clean_dataset(df):
    """Perform automatic cleaning: fill missing values and remove outliers."""
    #st.subheader("🧹 Auto-Cleaning Dataset")
    
    df_clean = df.copy()
    cleaning_log = []

    categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()

    # Fill missing values
    for col in numeric_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            cleaning_log.append(f"✅ Filled {missing_count} missing values in '{col}' with median")
    
    for col in categorical_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(mode_value)
            cleaning_log.append(f"✅ Filled {missing_count} missing values in '{col}' with mode/Unknown")

    # Remove outliers using IQR for numeric features
    original_rows = len(df_clean)
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:  # Only remove outliers if there's variation
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            before_outlier_removal = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            outliers_removed = before_outlier_removal - len(df_clean)
            #if outliers_removed > 0:
                #cleaning_log.append(f"🗑️ Removed {outliers_removed} outliers from '{col}'")

    total_outliers_removed = original_rows - len(df_clean)
    #if total_outliers_removed > 0:
        #cleaning_log.append(f"📊 Total rows removed due to outliers: {total_outliers_removed}")

    # Display cleaning log
    for log in cleaning_log:
        st.write(log)

    return df_clean, categorical_cols, numeric_cols

def engineer_features_with_metrics(df, num_cols, cat_cols):
    """Engineer comprehensive financial and business features"""
    #st.subheader("🧪 Feature Engineering")
    
    df_eng = df.copy()
    new_features = []

    # --- Revenue-based metrics ---
    revenue_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'income'])]
    cost_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['cost', 'expense', 'expenditure'])]
    
    if revenue_cols and cost_cols:
        revenue_col = revenue_cols[0]
        cost_col = cost_cols[0]
        
        # Net Profit/Loss
        df_eng["Net_Profit"] = df_eng[revenue_col] - df_eng[cost_col]
        df_eng["Net_Loss"] = df_eng["Net_Profit"].apply(lambda x: abs(x) if x < 0 else 0)
        df_eng["Total_Profit"] = df_eng["Net_Profit"].apply(lambda x: x if x > 0 else 0)
        new_features.extend(["Net_Profit", "Net_Loss", "Total_Profit"])
        
        # Profit Margin
        df_eng["Profit_Margin"] = (df_eng["Net_Profit"] / df_eng[revenue_col]).replace([np.inf, -np.inf], 0) * 100
        new_features.append("Profit_Margin")

    # --- COGS and Gross Profit ---
    cogs_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['cogs', 'cost_of_goods', 'direct_cost'])]
    if revenue_cols and cogs_cols:
        revenue_col = revenue_cols[0]
        cogs_col = cogs_cols[0]
        df_eng["Gross_Profit"] = df_eng[revenue_col] - df_eng[cogs_col]
        df_eng["Gross_Profit_Margin"] = (df_eng["Gross_Profit"] / df_eng[revenue_col]).replace([np.inf, -np.inf], 0) * 100
        new_features.extend(["Gross_Profit", "Gross_Profit_Margin"])

    # --- Operating metrics ---
    operating_cost_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['operating_cost', 'opex', 'operational'])]
    if revenue_cols and operating_cost_cols:
        revenue_col = revenue_cols[0]
        op_cost_col = operating_cost_cols[0]
        df_eng["EBIT"] = df_eng[revenue_col] - df_eng[op_cost_col]
        df_eng["EBIT_Margin"] = (df_eng["EBIT"] / df_eng[revenue_col]).replace([np.inf, -np.inf], 0) * 100
        new_features.extend(["EBIT", "EBIT_Margin"])

    # --- Investment and ROI ---
    investment_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['investment', 'capital', 'asset'])]
    if "Net_Profit" in df_eng.columns and investment_cols:
        investment_col = investment_cols[0]
        df_eng["ROI"] = (df_eng["Net_Profit"] / df_eng[investment_col]).replace([np.inf, -np.inf], 0) * 100
        new_features.append("ROI")

    # --- Units and Break-even analysis ---
    units_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['units', 'quantity', 'volume'])]
    if units_cols and revenue_cols:
        units_col = units_cols[0]
        revenue_col = revenue_cols[0]
        df_eng["Revenue_Per_Unit"] = (df_eng[revenue_col] / df_eng[units_col]).replace([np.inf, -np.inf], 0)
        new_features.append("Revenue_Per_Unit")
        
        if cost_cols:
            cost_col = cost_cols[0]
            df_eng["Cost_Per_Unit"] = (df_eng[cost_col] / df_eng[units_col]).replace([np.inf, -np.inf], 0)
            df_eng["Profit_Per_Unit"] = df_eng["Revenue_Per_Unit"] - df_eng["Cost_Per_Unit"]
            new_features.extend(["Cost_Per_Unit", "Profit_Per_Unit"])

    # --- Time-based features (if date columns exist) ---
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if not date_cols:
        # Try to identify date columns by name
        potential_date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month'])]
        for col in potential_date_cols:
            try:
                df_eng[col] = pd.to_datetime(df_eng[col])
                date_cols.append(col)
            except:
                pass

    if date_cols:
        date_col = date_cols[0]
        df_eng['Year'] = df_eng[date_col].dt.year
        df_eng['Month'] = df_eng[date_col].dt.month
        df_eng['Quarter'] = df_eng[date_col].dt.quarter
        new_features.extend(['Year', 'Month', 'Quarter'])

    # One-hot encode remaining categoricals
    remaining_cat_cols = [col for col in cat_cols if col in df_eng.columns]
    if remaining_cat_cols:
        df_eng = pd.get_dummies(df_eng, columns=remaining_cat_cols, drop_first=True)
        #st.info(f"🔄 One-hot encoded {len(remaining_cat_cols)} categorical columns")

    return df_eng, new_features

def calculate_kpis(df):
    """Calculate comprehensive KPIs with proper naming - FIXED VERSION"""
    kpis = {}
    
    # Revenue metrics
    revenue_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'income'])]
    if revenue_cols:
        kpis['Total_Revenue'] = float(df[revenue_cols[0]].sum())
    elif 'Revenue' in df.columns:
        kpis['Total_Revenue'] = float(df['Revenue'].sum())
    else:
        kpis['Total_Revenue'] = 0.0
    
    # Profit/Loss metrics
    if 'Total_Profit' in df.columns:
        kpis['Total_Profit'] = float(df['Total_Profit'].sum())
    elif 'Net_Profit' in df.columns:
        kpis['Total_Profit'] = float(df[df['Net_Profit'] > 0]['Net_Profit'].sum())
    else:
        kpis['Total_Profit'] = 0.0
        
    if 'Net_Loss' in df.columns:
        kpis['Total_Loss'] = float(df['Net_Loss'].sum())
    elif 'Net_Profit' in df.columns:
        kpis['Total_Loss'] = float(abs(df[df['Net_Profit'] < 0]['Net_Profit'].sum()))
    else:
        kpis['Total_Loss'] = 0.0
    
    # EBIT
    if 'EBIT' in df.columns:
        kpis['Total_EBIT'] = float(df['EBIT'].sum())
    else:
        kpis['Total_EBIT'] = 0.0
    
    # Gross Profit
    if 'Gross_Profit' in df.columns:
        kpis['Total_Gross_Profit'] = float(df['Gross_Profit'].sum())
    else:
        kpis['Total_Gross_Profit'] = 0.0
    
    # ROI
    if 'ROI' in df.columns:
        kpis['Average_ROI'] = float(df['ROI'].mean())
    else:
        kpis['Average_ROI'] = 0.0
    
    print(f"DEBUG KPIs calculated: {kpis}")
    return kpis

def get_kpi_translation_key(kpi_key):
    """Convert KPI keys to translation keys with explicit mapping"""
    mapping = {
        'Total_Revenue': 'totalrevenue',
        'Total_Profit': 'totalprofit', 
        'Total_Loss': 'totalloss',
        'Total_EBIT': 'totalebit',
        'Total_Gross_Profit': 'totalgrossprofit',
        'Average_ROI': 'averageroi'
    }
    return mapping.get(kpi_key, kpi_key.lower().replace('_', ''))

def format_large_number(value):
    """Format large numbers with appropriate suffixes"""
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.2f}"

def create_kpi_cards(kpis):
    """Create beautiful KPI cards with properly formatted numbers"""
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        profit_loss_diff = kpis['Total_Profit'] - kpis['Total_Loss']
        color = "normal" if profit_loss_diff >= 0 else "inverse"
        st.metric(
            label="💰 Total Profit/Loss", 
            value=format_large_number(profit_loss_diff),
            delta=f"Profit: {format_large_number(kpis['Total_Profit'])}"
        )
    
    with col2:
        st.metric(
            label="📈 Total Revenue", 
            value=format_large_number(kpis['Total_Revenue']),
            delta="Primary income source"
        )
    
    with col3:
        st.metric(
            label="🏢 EBIT", 
            value=format_large_number(kpis['Total_EBIT']),
            delta="Earnings before interest & tax"
        )
    
    with col4:
        st.metric(
            label="📊 Gross Profit", 
            value=format_large_number(kpis['Total_Gross_Profit']),
            delta="Revenue - COGS"
        )
    
    with col5:
        roi_color = "normal" if kpis['Average_ROI'] >= 0 else "inverse"
        st.metric(
            label="🎯 Average ROI", 
            value=f"{kpis['Average_ROI']:.2f}%",
            delta="Return on Investment"
        )

def create_profitability_trend(df):
    """Create profitability trend lines over time"""
    st.subheader("📈 Profitability Trend Analysis")
    
    # Check for time-based columns
    time_cols = []
    for col in ['Year', 'Month', 'Quarter', 'Date']:
        if col in df.columns:
            time_cols.append(col)
    
    if not time_cols:
        # Try to find date-like columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month']):
                time_cols.append(col)
                break
    
    if time_cols and any(col in df.columns for col in ['Gross_Profit_Margin', 'EBIT_Margin', 'Profit_Margin']):
        time_col = time_cols[0]
        
        # Aggregate by time period
        margin_cols = [col for col in ['Gross_Profit_Margin', 'EBIT_Margin', 'Profit_Margin'] if col in df.columns]
        
        if margin_cols:
            trend_data = df.groupby(time_col)[margin_cols].mean().reset_index()
            
            fig = go.Figure()
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for i, col in enumerate(margin_cols):
                fig.add_trace(go.Scatter(
                    x=trend_data[time_col],
                    y=trend_data[col],
                    mode='lines+markers',
                    name=col.replace('_', ' '),
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="📈 Profitability Margins Trend Over Time",
                xaxis_title=time_col,
                yaxis_title="Margin (%)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            # Calculate average margin for the insight
            avg_margin = trend_data[margin_cols].mean().mean()
            return f"The average profitability margin across all periods is {avg_margin:.2f}%, with notable fluctuations over time."
    else:
        st.info("📅 No time-based data or margin columns found for trend analysis")

def create_waterfall_chart(df):
    """Create waterfall chart showing Revenue → Gross Profit → EBIT → Net Profit"""
    st.subheader("💧 Financial Waterfall Analysis")
    
    # Calculate totals
    revenue = df['Revenue'].sum() if 'Revenue' in df.columns else 0
    gross_profit = df['Gross_Profit'].sum() if 'Gross_Profit' in df.columns else 0
    ebit = df['EBIT'].sum() if 'EBIT' in df.columns else 0
    net_profit = df['Net_Profit'].sum() if 'Net_Profit' in df.columns else 0
    
    if revenue > 0:
        # Calculate deductions
        cogs = revenue - gross_profit if gross_profit > 0 else 0
        operating_expenses = gross_profit - ebit if ebit > 0 and gross_profit > 0 else 0
        other_expenses = ebit - net_profit if net_profit > 0 and ebit > 0 else 0
        
        # Waterfall data
        categories = ['Revenue', 'COGS', 'Gross Profit', 'OpEx', 'EBIT', 'Other', 'Net Profit']
        values = [revenue, -cogs, gross_profit, -operating_expenses, ebit, -other_expenses, net_profit]
        
        fig = go.Figure(go.Waterfall(
            name="Financial Flow",
            orientation="v",
            measure=["absolute", "relative", "total", "relative", "total", "relative", "total"],
            x=categories,
            textposition="outside",
            text=[f"${v:,.0f}" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="💧 Revenue to Net Profit Waterfall",
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return f"From ${revenue:,.0f} revenue, net profit is ${net_profit:,.0f} after COGS and operating costs."
    else:
        st.info("💧 No revenue data found for waterfall chart")

def create_breakeven_chart(df):
    """Create break-even analysis chart"""
    st.subheader("⚖️ Break-Even Analysis")
    
    if all(col in df.columns for col in ['Revenue_Per_Unit', 'Cost_Per_Unit']) or \
       ('Revenue' in df.columns and any('units' in col.lower() for col in df.columns)):
        
        # Get units data
        units_col = None
        for col in df.columns:
            if 'units' in col.lower() or 'quantity' in col.lower():
                units_col = col
                break
        
        if units_col and 'Revenue' in df.columns:
            max_units = int(df[units_col].max() * 1.2)
            units_range = np.arange(0, max_units, max_units//50)
            
            avg_revenue_per_unit = df['Revenue_Per_Unit'].mean() if 'Revenue_Per_Unit' in df.columns else df['Revenue'].sum() / df[units_col].sum()
            avg_cost_per_unit = df['Cost_Per_Unit'].mean() if 'Cost_Per_Unit' in df.columns else 0
            
            if avg_cost_per_unit == 0 and 'Total_Costs' in df.columns:
                avg_cost_per_unit = df['Total_Costs'].sum() / df[units_col].sum()
            
            revenue_line = units_range * avg_revenue_per_unit
            cost_line = units_range * avg_cost_per_unit
            
            # Calculate break-even point
            if avg_revenue_per_unit > avg_cost_per_unit:
                breakeven_units = 0  # Already profitable from first unit
            else:
                fixed_costs = df['Fixed_Costs'].sum() if 'Fixed_Costs' in df.columns else 0
                if fixed_costs > 0 and avg_revenue_per_unit > avg_cost_per_unit:
                    breakeven_units = fixed_costs / (avg_revenue_per_unit - avg_cost_per_unit)
                else:
                    breakeven_units = 0
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=units_range, y=revenue_line,
                mode='lines', name='Revenue',
                line=dict(color='green', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=units_range, y=cost_line,
                mode='lines', name='Costs',
                line=dict(color='red', width=3)
            ))
            
            if breakeven_units > 0 and breakeven_units < max_units:
                fig.add_vline(x=breakeven_units, line_dash="dash", line_color="orange",
                            annotation_text=f"Break-even: {breakeven_units:.0f} units")
            
            fig.update_layout(
                title="⚖️ Break-Even Analysis: Units vs Revenue/Costs",
                xaxis_title="Units Sold",
                yaxis_title="Amount ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return f"The average revenue per unit is ${avg_revenue_per_unit:.2f} and the cost is ${avg_cost_per_unit:.2f}, resulting in a profit margin of ${(avg_revenue_per_unit - avg_cost_per_unit):.2f} per unit."
        else:
            st.info("⚖️ No units/quantity data found for break-even analysis")
    else:
        st.info("⚖️ Insufficient data for break-even analysis")

def create_segment_profitability(df):
    """Create profitability analysis by segments"""
    st.subheader("🎯 Profitability by Business Segments")
    
    # Look for segment columns
    segment_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['product', 'segment', 'category', 'region', 'department']):
            if df[col].dtype == 'object' and df[col].nunique() < 20:  # Reasonable number of categories
                segment_cols.append(col)
    
    if segment_cols and 'Net_Profit' in df.columns:
        for segment_col in segment_cols[:2]:  # Limit to 2 segments to avoid clutter
            segment_profit = df.groupby(segment_col)['Net_Profit'].sum().reset_index()
            segment_profit = segment_profit.sort_values('Net_Profit', ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                segment_profit, 
                x='Net_Profit', 
                y=segment_col,
                orientation='h',
                title=f"🎯 Profitability by {segment_col}",
                color='Net_Profit',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Also create pie chart if reasonable number of segments
            if len(segment_profit) <= 8:
                positive_profits = segment_profit[segment_profit['Net_Profit'] > 0]
                if len(positive_profits) > 0:
                    fig_pie = px.pie(
                        positive_profits, 
                        values='Net_Profit', 
                        names=segment_col,
                        title=f"🥧 Profit Distribution by {segment_col}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    # Define seg_data for insight
                    seg_data = df.groupby(segment_col)['Net_Profit'].sum()
                    return f"'{seg_data.idxmax()}' is the most profitable segment, while '{seg_data.idxmin()}' contributes the least profit."
    else:
        st.info("🎯 No segment columns or profit data found for segment analysis")

def create_roi_funnel(df):
    """Create ROI funnel analysis"""
    st.subheader("🎯 ROI Investment Funnel")
    
    if 'ROI' in df.columns:
        # Create ROI distribution
        roi_ranges = [
            ('🔴 Loss (< 0%)', (df['ROI'] < 0).sum()),
            ('🟡 Low (0-10%)', ((df['ROI'] >= 0) & (df['ROI'] <= 10)).sum()),
            ('🟢 Good (10-25%)', ((df['ROI'] > 10) & (df['ROI'] <= 25)).sum()),
            ('💚 Excellent (> 25%)', (df['ROI'] > 25).sum())
        ]
        
        categories, counts = zip(*roi_ranges)
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=counts, 
                  marker_color=['red', 'orange', 'lightgreen', 'green'])
        ])
        
        fig.update_layout(
            title="🎯 ROI Distribution Across Investments",
            xaxis_title="ROI Range",
            yaxis_title="Number of Investments",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate average ROI and most common range for insight
        avg_roi = df['ROI'].mean()
        ranges = dict(roi_ranges)
        most_common_range = max(ranges, key=ranges.get)

        # Investment vs Returns scatter
        investment_cols = [col for col in df.columns if 'investment' in col.lower() or 'capital' in col.lower()]
        if investment_cols and 'Net_Profit' in df.columns:
            investment_col = investment_cols[0]
            
            fig_scatter = px.scatter(
                df, x=investment_col, y='Net_Profit',
                size=df['ROI'].abs(), color='ROI', # Corrected line
                title="💰 Investment vs Returns Analysis",
                labels={investment_col: "Investment Amount", 'Net_Profit': 'Returns'},
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        return f"The average ROI across all investments is {avg_roi:.2f}%. Most investments fall into the '{most_common_range}' performance category."
    else:
        st.info("🎯 No ROI data found for funnel analysis")

def create_profitability_roi_heatmap(df):
    """Create heatmap comparing profitability vs ROI efficiency"""
    st.subheader("🔥 Profitability vs ROI Efficiency Heatmap")
    
    # Look for segment columns for heatmap dimensions
    segment_cols = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['product', 'segment', 'category', 'region']) 
                   and df[col].dtype == 'object' and df[col].nunique() < 10]
    
    if len(segment_cols) >= 1 and 'Net_Profit' in df.columns and 'ROI' in df.columns:
        segment_col = segment_cols[0]
        
        # Create pivot table
        heatmap_data = df.groupby(segment_col).agg({
            'Net_Profit': 'sum',
            'ROI': 'mean'
        }).reset_index()
        
        # Create bins for better visualization
        heatmap_data['Profit_Level'] = pd.cut(heatmap_data['Net_Profit'], 
                                            bins=3, labels=['Low', 'Medium', 'High'])
        heatmap_data['ROI_Level'] = pd.cut(heatmap_data['ROI'], 
                                         bins=3, labels=['Low', 'Medium', 'High'])
        
        # Create heatmap matrix
        pivot_data = heatmap_data.pivot_table(values='Net_Profit', 
                                             index='Profit_Level', 
                                             columns='ROI_Level', 
                                             aggfunc='count', 
                                             fill_value=0)
        
        fig = px.imshow(pivot_data, 
                       title="🔥 Profitability vs ROI Efficiency Matrix",
                       labels=dict(x="ROI Level", y="Profit Level", color="Count"),
                       color_continuous_scale='Viridis')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Alternative: Scatter plot with color coding   
        fig_scatter = px.scatter(
            heatmap_data, x='ROI', y='Net_Profit',
            color=segment_col, size=heatmap_data['Net_Profit'].abs(), # Corrected line
            title="🎯 Profitability vs ROI by Segment",
            labels={'ROI': 'Average ROI (%)', 'Net_Profit': 'Total Profit ($)'}
        )
        
        # Add quadrant lines
        median_roi = heatmap_data['ROI'].median()
        median_profit = heatmap_data['Net_Profit'].median()
        
        fig_scatter.add_hline(y=median_profit, line_dash="dash", line_color="gray", 
                            annotation_text="Median Profit")
        fig_scatter.add_vline(x=median_roi, line_dash="dash", line_color="gray", 
                            annotation_text="Median ROI")
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        # Calculate averages for insight
        avg_roi = heatmap_data['ROI'].mean()
        avg_margin = (heatmap_data['Net_Profit'] / heatmap_data['Net_Profit'].abs().sum() * 100).mean() if heatmap_data['Net_Profit'].abs().sum() != 0 else 0
        return f"Segments show varied performance, with an average ROI of {avg_roi:.2f}% and an average profit contribution of {avg_margin:.2f}%."
    else:
        st.info("🔥 Insufficient data for profitability vs ROI heatmap analysis")

def create_roi_by_project_year(df):
    """Create ROI analysis by project/category and year"""
    st.subheader("📊 ROI Analysis by Project & Year")
    
    # Look for project/category columns
    project_cols = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['project', 'product', 'category', 'segment', 'division']) 
                   and df[col].dtype == 'object' and df[col].nunique() < 20]
    
    # Look for year column
    year_cols = [col for col in df.columns if any(keyword in col.lower() 
                for keyword in ['year', 'date']) and df[col].dtype in ['int64', 'datetime64[ns]', 'object']]
    
    if 'ROI' in df.columns:
        col1, col2 = st.columns(2)
        
        # ROI by Project/Category (Bar Chart)
        if project_cols:
            with col1:
                project_col = project_cols[0]
                roi_by_project = df.groupby(project_col)['ROI'].mean().reset_index()
                roi_by_project = roi_by_project.sort_values('ROI', ascending=True)
                
                fig_project = px.bar(
                    roi_by_project, 
                    x='ROI', 
                    y=project_col,
                    orientation='h',
                    title=f"📈 Average ROI by {project_col}",
                    color='ROI',
                    color_continuous_scale='RdYlGn',
                    labels={'ROI': 'Average ROI (%)', project_col: project_col}
                )
                fig_project.update_layout(height=400)
                st.plotly_chart(fig_project, use_container_width=True)
        
        # ROI by Year (Line Chart)
        if year_cols:
            with col2:
                year_col = year_cols[0]
                
                # Handle different year formats
                if df[year_col].dtype == 'datetime64[ns]':
                    df_temp = df.copy()
                    df_temp['Year'] = df_temp[year_col].dt.year
                    year_col = 'Year'
                elif df[year_col].dtype == 'object':
                    try:
                        df_temp = df.copy()
                        df_temp[year_col] = pd.to_datetime(df_temp[year_col]).dt.year
                    except:
                        df_temp = df.copy()
                else:
                    df_temp = df.copy()
                
                roi_by_year = df_temp.groupby(year_col)['ROI'].agg(['mean', 'count']).reset_index()
                roi_by_year.columns = [year_col, 'Average_ROI', 'Investment_Count']
                roi_by_year = roi_by_year.sort_values(year_col)
                
                fig_year = go.Figure()
                
                # ROI trend line
                fig_year.add_trace(go.Scatter(
                    x=roi_by_year[year_col],
                    y=roi_by_year['Average_ROI'],
                    mode='lines+markers',
                    name='Average ROI',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8),
                    yaxis='y'
                ))
                
                # Investment count bars (secondary axis)
                fig_year.add_trace(go.Bar(
                    x=roi_by_year[year_col],
                    y=roi_by_year['Investment_Count'],
                    name='Investment Count',
                    opacity=0.3,
                    yaxis='y2',
                    marker_color='lightblue'
                ))
                
                fig_year.update_layout(
                    title=f"📈 ROI Trend Over {year_col}",
                    xaxis_title=year_col,
                    yaxis=dict(
                        title="Average ROI (%)",
                        side="left"
                    ),
                    yaxis2=dict(
                        title="Number of Investments",
                        side="right",
                        overlaying="y"
                    ),
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_year, use_container_width=True)
        
        # Combined analysis if both project and year exist
        if project_cols and year_cols:
            st.subheader("🔥 ROI Heatmap: Project vs Year")
            
            project_col = project_cols[0]
            year_col = year_cols[0] if year_cols[0] != 'Year' else 'Year'
            
            # Create heatmap data
            heatmap_data = df_temp.groupby([project_col, year_col])['ROI'].mean().reset_index()
            pivot_data = heatmap_data.pivot(index=project_col, columns=year_col, values='ROI')
            
            fig_heatmap = px.imshow(
                pivot_data,
                title="🔥 ROI Performance Heatmap: Projects Over Time",
                labels=dict(x="Year", y="Project/Category", color="ROI (%)"),
                color_continuous_scale='RdYlGn',
                aspect="auto"
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            best_year_data = roi_by_year.loc[roi_by_year['Average_ROI'].idxmax()]
            best_year = int(best_year_data[year_col])
            best_roi = best_year_data['Average_ROI']
            return f"ROI performance peaked in {best_year}, reaching an average of {best_roi:.2f}%."
    else:
        st.info("📊 No ROI data found for project/year analysis")

def create_cumulative_roi_curves(df):
    """Create cumulative ROI curves over time (clean version without debug output)"""
    st.subheader("📈 Cumulative ROI Performance")
    
    if 'ROI' not in df.columns:
        st.error("⚠️ ROI column not found in the data")
        return False
    
    # Look for date/time columns
    date_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'year', 'month', 'day']):
            date_cols.append(col)
    
    if not date_cols:
        st.warning("⚠️ No date/time column found. Creating index-based analysis.")
        df_temp = df.copy().reset_index()
        df_temp['Index'] = range(len(df_temp))
        date_col = 'Index'
    else:
        date_col = date_cols[0]  # Use the first date column found
        df_temp = df.copy()
        
        # Convert to datetime if needed
        try:
            if df_temp[date_col].dtype not in ['datetime64[ns]']:
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            
            # Remove rows with invalid dates
            df_temp = df_temp.dropna(subset=[date_col])
            
            if df_temp.empty:
                st.error("⚠️ No valid dates found after processing")
                return False
                
        except Exception as e:
            st.error(f"⚠️ Could not process date column: {str(e)}")
            # Fallback to index
            df_temp['Index'] = range(len(df_temp))
            date_col = 'Index'
    
    # Sort by date and reset index
    df_temp = df_temp.sort_values(date_col).reset_index(drop=True)
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Overall cumulative ROI
            df_temp['Cumulative_ROI'] = df_temp['ROI'].cumsum()
            
            fig_cum = go.Figure()
            
            fig_cum.add_trace(go.Scatter(
                x=df_temp[date_col],
                y=df_temp['Cumulative_ROI'],
                mode='lines',
                name='Cumulative ROI',
                line=dict(color='green', width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative ROI:</b> %{y:.2f}%<extra></extra>'
            ))
            
            # Add zero line
            fig_cum.add_hline(y=0, line_dash="dash", line_color="red", 
                            annotation_text="Break-even Line")
            
            # Add current performance annotation
            final_roi = df_temp['Cumulative_ROI'].iloc[-1]
            fig_cum.add_annotation(
                x=df_temp[date_col].iloc[-1],
                y=final_roi,
                text=f"Final: {final_roi:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green" if final_roi > 0 else "red",
                bgcolor="lightgreen" if final_roi > 0 else "lightcoral",
                bordercolor="green" if final_roi > 0 else "red"
            )
            
            fig_cum.update_layout(
                title="📈 Cumulative ROI Over Time",
                xaxis_title="Time Period",
                yaxis_title="Cumulative ROI (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_cum, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating cumulative chart: {str(e)}")
    
    with col2:
        try:
            # Rolling average ROI with volatility bands
            window_size = max(1, min(30, len(df_temp) // 10))  # Adaptive window size
            df_temp['Rolling_ROI'] = df_temp['ROI'].rolling(window=window_size, min_periods=1).mean()
            df_temp['Rolling_Std'] = df_temp['ROI'].rolling(window=window_size, min_periods=1).std()
            
            fig_rolling = go.Figure()
            
            # Add volatility bands
            df_temp['Upper_Band'] = df_temp['Rolling_ROI'] + df_temp['Rolling_Std']
            df_temp['Lower_Band'] = df_temp['Rolling_ROI'] - df_temp['Rolling_Std']
            
            # Volatility bands
            fig_rolling.add_trace(go.Scatter(
                x=df_temp[date_col],
                y=df_temp['Upper_Band'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0)',
                showlegend=False
            ))
            
            fig_rolling.add_trace(go.Scatter(
                x=df_temp[date_col],
                y=df_temp['Lower_Band'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0)',
                name='Volatility Band',
                fillcolor='rgba(0,100,80,0.2)'
            ))
            
            # Rolling average
            fig_rolling.add_trace(go.Scatter(
                x=df_temp[date_col],
                y=df_temp['Rolling_ROI'],
                mode='lines',
                name=f'{window_size}-Period Average',
                line=dict(color='blue', width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>Avg ROI:</b> %{y:.2f}%<extra></extra>'
            ))
            
            # Add zero line
            fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig_rolling.update_layout(
                title="📊 ROI Trend Analysis with Volatility",
                xaxis_title="Time Period",
                yaxis_title="ROI (%)",
                height=400
            )
            
            st.plotly_chart(fig_rolling, use_container_width=True)
            final_roi = df_temp['Cumulative_ROI'].iloc[-1]
            return f"The cumulative ROI over the entire period has reached {final_roi:.2f}%, indicating the overall long-term performance."
            
        except Exception as e:
            st.error(f"Error creating trend chart: {str(e)}")
    
    return True

def display_cumulative_performance_metrics(df):
    """Display cumulative performance metrics (clean version)"""
    st.subheader("📊 Performance Summary")
    
    if 'ROI' not in df.columns:
        st.error("⚠️ ROI column not found for metrics calculation")
        return False
    
    try:
        # Calculate metrics
        roi_data = df['ROI'].dropna()
        
        if len(roi_data) == 0:
            st.error("⚠️ No valid ROI data found")
            return False
        
        # Core metrics
        avg_roi = float(roi_data.mean())
        positive_roi_pct = float((roi_data > 0).mean() * 100)
        roi_volatility = float(roi_data.std())
        best_roi = float(roi_data.max())
        worst_roi = float(roi_data.min())
        total_investments = len(roi_data)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📈 Average ROI", 
                value=f"{avg_roi:.1f}%",
                delta=f"Best: {best_roi:.1f}%"
            )
        
        with col2:
            st.metric(
                label="✅ Success Rate", 
                value=f"{positive_roi_pct:.0f}%",
                delta=f"{total_investments} investments"
            )
        
        with col3:
            volatility_level = "Low" if roi_volatility < 50 else "Medium" if roi_volatility < 150 else "High"
            st.metric(
                label="📊 Volatility", 
                value=f"{roi_volatility:.1f}%",
                delta=f"{volatility_level} Risk"
            )
        
        with col4:
            cumulative_roi = float(roi_data.sum())
            st.metric(
                label="💰 Total ROI", 
                value=f"{cumulative_roi:.1f}%",
                delta="All investments combined"
            )
        
        # Performance insights
        st.markdown("### 💡 Key Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            if avg_roi > 20:
                st.success("🟢 **Strong Performance**: Average ROI exceeds 20%")
            elif avg_roi > 0:
                st.info("🟡 **Positive Performance**: Average ROI is positive but could improve")
            else:
                st.warning("🔴 **Underperforming**: Average ROI is negative")
                
            if positive_roi_pct > 70:
                st.success(f"🎯 **High Success Rate**: {positive_roi_pct:.0f}% of investments are profitable")
            else:
                st.warning(f"⚠️ **Mixed Results**: Only {positive_roi_pct:.0f}% of investments are profitable")
        
        with insight_col2:
            # Risk assessment
            if roi_volatility < 50:
                st.success("🛡️ **Low Risk Portfolio**: Consistent returns with low volatility")
            elif roi_volatility < 150:
                st.info("⚖️ **Balanced Risk**: Moderate volatility with reasonable returns")
            else:
                st.warning("⚡ **High Risk Portfolio**: High volatility - review risk management")
                
            # Recent trend analysis
            if len(roi_data) >= 10:
                recent_data = roi_data.tail(min(100, len(roi_data)))
                recent_avg = recent_data.mean()
                if recent_avg > avg_roi * 1.1:
                    st.success("📈 **Improving Trend**: Recent performance is stronger")
                elif recent_avg < avg_roi * 0.9:
                    st.warning("📉 **Declining Trend**: Recent performance is weaker")
                else:
                    st.info("➡️ **Stable Trend**: Consistent performance over time")
        
        return True
        
    except Exception as e:
        st.error(f"⚠️ Error calculating metrics: {str(e)}")
        return False

# Additional helper function to debug data issues
def debug_roi_data(df):
    """Debug function to check data quality"""
    st.subheader("🔍 Data Debug Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**DataFrame Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        
    with col2:
        if 'ROI' in df.columns:
            st.write("**ROI Column Stats:**")
            st.write(f"- Count: {df['ROI'].count()}")
            st.write(f"- Null values: {df['ROI'].isnull().sum()}")
            st.write(f"- Data type: {df['ROI'].dtype}")
            
    # Show sample data
    st.write("**Sample Data:**")
    st.dataframe(df.head())
    
    # Check for date columns
    date_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month']):
            date_cols.append(col)
    
    if date_cols:
        st.write("**Date Columns Found:**", date_cols)
        for col in date_cols[:2]:  # Show info for first 2 date columns
            st.write(f"**{col} Column:**")
            st.write(f"- Data type: {df[col].dtype}")
            st.write(f"- Sample values: {df[col].head().tolist()}")
    else:
        st.warning("No date columns detected")

def create_focused_analytics_by_type(df, analysis_type):
    """Create visualizations based on selected analysis type with grid layout"""
    
    if analysis_type == "💰 Profit & Loss (P/L)":
        st.header("💰 Profit & Loss Analysis Dashboard")
        
        # Create a 2x2 grid for P/L visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            create_waterfall_chart(df)
            create_breakeven_chart(df)
        
        with col2:
            create_profitability_trend(df)
            create_segment_profitability(df)
        
        # Full width visualization
        create_profitability_roi_heatmap(df)
        
    elif analysis_type == "📈 Return on Investment (ROI)":
        st.header("📈 Return on Investment Analysis Dashboard")
        
        # Create a 2x2 grid for ROI visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            create_roi_funnel(df)
        
        with col2:
            create_cumulative_roi_curves(df)
        
        # Full width ROI analysis (includes cumulative metrics at the end)
        create_roi_by_project_year(df)
        
        # Additional ROI analysis
        create_roi_vs_risk_scatter(df)
        display_cumulative_performance_metrics(df)
        
def create_roi_vs_risk_scatter(df):
    """Create ROI vs Risk scatter plots"""
    st.subheader("⚖️ ROI vs Risk Analysis")
    
    if 'ROI' in df.columns:
        # Calculate risk metrics
        risk_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['risk', 'volatility', 'variance', 'std', 'deviation'])]
        
        investment_cols = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['investment', 'capital', 'amount', 'cost'])]
        
        # If no explicit risk column, calculate risk as ROI standard deviation by groups
        if not risk_cols:
            # Look for grouping columns to calculate risk
            group_cols = [col for col in df.columns if any(keyword in col.lower() 
                         for keyword in ['project', 'product', 'category', 'segment']) 
                         and df[col].dtype == 'object' and df[col].nunique() < 20]
            
            if group_cols:
                group_col = group_cols[0]
                
                # Calculate risk as standard deviation of ROI within each group
                risk_data = df.groupby(group_col)['ROI'].agg(['mean', 'std', 'count']).reset_index()
                risk_data.columns = [group_col, 'Average_ROI', 'ROI_Risk', 'Count']
                risk_data['ROI_Risk'] = risk_data['ROI_Risk'].fillna(0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk-Return scatter by category
                    fig_risk = px.scatter(
                        risk_data,
                        x='ROI_Risk',
                        y='Average_ROI',
                        size='Count',
                        color=group_col,
                        title="⚖️ Risk-Return Profile by Category",
                        labels={
                            'ROI_Risk': 'Risk (ROI Std Dev %)',
                            'Average_ROI': 'Average ROI (%)',
                            'Count': 'Number of Investments'
                        },
                        hover_data=[group_col, 'Count']
                    )
                    
                    # Add quadrant lines
                    median_risk = risk_data['ROI_Risk'].median()
                    median_return = risk_data['Average_ROI'].median()
                    
                    fig_risk.add_hline(y=median_return, line_dash="dash", line_color="gray",
                                     annotation_text="Median Return")
                    fig_risk.add_vline(x=median_risk, line_dash="dash", line_color="gray",
                                     annotation_text="Median Risk")
                    
                    # Add quadrant annotations
                    max_x = risk_data['ROI_Risk'].max()
                    max_y = risk_data['Average_ROI'].max()
                    min_y = risk_data['Average_ROI'].min()
                    
                    # High Return, Low Risk (ideal quadrant)
                    fig_risk.add_annotation(
                        x=median_risk * 0.3,
                        y=median_return + (max_y - median_return) * 0.7,
                        text="🟢 Low Risk<br>High Return",
                        showarrow=False,
                        bgcolor="lightgreen",
                        opacity=0.7
                    )
                    
                    # High Return, High Risk
                    fig_risk.add_annotation(
                        x=median_risk + (max_x - median_risk) * 0.7,
                        y=median_return + (max_y - median_return) * 0.7,
                        text="🟡 High Risk<br>High Return",
                        showarrow=False,
                        bgcolor="lightyellow",
                        opacity=0.7
                    )
                    
                    fig_risk.update_layout(height=500)
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                with col2:
                    # Investment size vs ROI with risk coloring
                    if investment_cols:
                        investment_col = investment_cols[0]
                        
                        # Merge with original data for investment amounts
                        detailed_data = df.merge(
                            risk_data[[group_col, 'ROI_Risk']], 
                            on=group_col, 
                            how='left'
                        )
                        
                        fig_investment = px.scatter(
                            detailed_data,
                            x=investment_col,
                            y='ROI',
                            color='ROI_Risk',
                            size=detailed_data[investment_col].abs(),
                            title="💰 Investment Amount vs ROI (Colored by Risk)",
                            labels={
                                investment_col: f'Investment Amount ({investment_col})',
                                'ROI': 'ROI (%)',
                                'ROI_Risk': 'Risk Level'
                            },
                            color_continuous_scale='RdYlGn_r'  # Red for high risk, Green for low risk
                        )
                        
                        fig_investment.update_layout(height=500)
                        st.plotly_chart(fig_investment, use_container_width=True)
                    else:
                        # Alternative: ROI distribution by risk level
                        risk_data['Risk_Category'] = pd.cut(
                            risk_data['ROI_Risk'], 
                            bins=3, 
                            labels=['Low Risk', 'Medium Risk', 'High Risk']
                        )
                        
                        fig_box = px.box(
                            risk_data,
                            x='Risk_Category',
                            y='Average_ROI',
                            title="📊 ROI Distribution by Risk Category",
                            color='Risk_Category',
                            color_discrete_map={
                                'Low Risk': 'green',
                                'Medium Risk': 'orange', 
                                'High Risk': 'red'
                            }
                        )
                        
                        fig_box.update_layout(height=500)
                        st.plotly_chart(fig_box, use_container_width=True)
                
                # Risk-Return summary table
                st.subheader("📋 Risk-Return Summary")
                
                # Add risk-return ratios
                risk_data['Sharpe_Ratio'] = risk_data['Average_ROI'] / risk_data['ROI_Risk']
                risk_data['Sharpe_Ratio'] = risk_data['Sharpe_Ratio'].replace([np.inf, -np.inf], 0)
                
                summary_data = risk_data.copy()
                summary_data = summary_data.round(2)
                summary_data = summary_data.sort_values('Average_ROI', ascending=False)
                
                st.dataframe(
                    summary_data[[group_col, 'Average_ROI', 'ROI_Risk', 'Sharpe_Ratio', 'Count']],
                    use_container_width=True
                )
                
            else:
                st.info("⚖️ No grouping columns found to calculate risk metrics")
        
        else:
            # Use existing risk column
            risk_col = risk_cols[0]
            
            fig_existing_risk = px.scatter(
                df,
                x=risk_col,
                y='ROI',
                title="⚖️ ROI vs Risk Analysis",
                labels={risk_col: 'Risk', 'ROI': 'ROI (%)'},
                opacity=0.7
            )
            
            st.plotly_chart(fig_existing_risk, use_container_width=True)
            avg_roi = df['ROI'].mean()
            roi_std = df['ROI'].std()
        return f"The portfolio shows an average ROI of {avg_roi:.2f}% with a risk (volatility) of {roi_std:.2f}%."  
    else:
        st.info("⚖️ No ROI data found for risk analysis")

# ===========================
# HELPER FUNCTIONS FOR SIDEBAR
# ===========================
def get_suggested_targets(business_type, available_columns):
    """Suggest target variables based on business context"""
    
    suggestions = {
        "Sales & Revenue Analysis": ["Revenue", "Sales", "Total_Sales", "Net_Sales"],
        "Marketing ROI Analysis": ["ROI", "Marketing_ROI", "ROAS", "Conversion_Rate"],
        "Financial Performance": ["Net_Profit", "Profit_Margin", "EBITDA", "Operating_Income"],
        "Investment Analysis": ["ROI", "Net_Present_Value", "IRR", "Payback_Period"],
        "Project Performance": ["Project_Profit", "Completion_Rate", "Efficiency_Score"],
        "Cost Analysis": ["Total_Costs", "Cost_Per_Unit", "Operating_Costs"],
        "General Business Analysis": ["Net_Profit", "Revenue", "ROI", "Growth_Rate"]
    }
    
    target_keywords = suggestions.get(business_type, suggestions["General Business Analysis"])
    
    # Find matching columns
    matched_targets = []
    for keyword in target_keywords:
        for col in available_columns:
            if keyword.lower() in col.lower() or col.lower() in keyword.lower():
                if col not in matched_targets:
                    matched_targets.append(col)
    
    # If no matches, return all numeric columns
    return matched_targets if matched_targets else available_columns

def apply_time_filter(df, time_filter, date_col, start_period=None, end_period=None):
    """Apply time period filtering"""
    if time_filter == "All Time":
        return df
    
    df_filtered = df.copy()
    
    # Convert date column if needed
    if df_filtered[date_col].dtype not in ['datetime64[ns]']:
        try:
            df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
        except:
            return df  # Return original if conversion fails
    
    current_year = datetime.now().year
    
    if time_filter == "Last Year":
        start_date = pd.to_datetime(f"{current_year-1}-01-01")
        df_filtered = df_filtered[df_filtered[date_col] >= start_date]
    elif time_filter == "Last 2 Years":
        start_date = pd.to_datetime(f"{current_year-2}-01-01") 
        df_filtered = df_filtered[df_filtered[date_col] >= start_date]
    elif time_filter == "Custom Range" and start_period and end_period:
        try:
            start_date = pd.to_datetime(f"{start_period}-01-01")
            end_date = pd.to_datetime(f"{end_period}-12-31")
            df_filtered = df_filtered[
                (df_filtered[date_col] >= start_date) & 
                (df_filtered[date_col] <= end_date)
            ]
        except:
            pass  # Keep original data if date parsing fails
    
    return df_filtered

def create_performance_alerts(df, profit_threshold, roi_threshold):
    """Generate performance alerts based on thresholds"""
    alerts = []
    
    if 'Profit_Margin' in df.columns:
        low_profit_count = (df['Profit_Margin'] < profit_threshold).sum()
        if low_profit_count > 0:
            alerts.append({
                'type': 'warning',
                'title': 'Low Profit Margin Alert',
                'message': f'{low_profit_count} entries have profit margin below {profit_threshold}%',
                'count': low_profit_count
            })
    
    if 'ROI' in df.columns:
        low_roi_count = (df['ROI'] < roi_threshold).sum()
        if low_roi_count > 0:
            alerts.append({
                'type': 'danger',
                'title': 'Low ROI Alert', 
                'message': f'{low_roi_count} investments have ROI below {roi_threshold}%',
                'count': low_roi_count
            })
    
    return alerts

def display_alerts(alerts):
    """Display performance alerts"""
    if alerts:
        st.subheader("🚨 Performance Alerts")
        for alert in alerts:
            if alert['type'] == 'warning':
                st.warning(f"⚠️ {alert['title']}: {alert['message']}")
            elif alert['type'] == 'danger':
                st.error(f"🚨 {alert['title']}: {alert['message']}")
            else:
                st.info(f"ℹ️ {alert['title']}: {alert['message']}")

def save_analysis_state():
    """Save current analysis configuration"""
    st.success("✅ Analysis view saved! You can restore this configuration later.")
    # In a real app, this would save to database or file

def detect_business_issues():
    """Auto-detect potential business issues"""
    if st.session_state.get("data_engineered") is not None:
        df = st.session_state["data_engineered"]
        issues = []
        
        # Check for negative profits
        if 'Net_Profit' in df.columns:
            loss_rate = (df['Net_Profit'] < 0).mean() * 100
            if loss_rate > 20:
                issues.append(f"🚨 High loss rate: {loss_rate:.1f}% of entries are unprofitable")
        
        # Check for declining trends
        if 'Year' in df.columns and 'Revenue' in df.columns:
            yearly_revenue = df.groupby('Year')['Revenue'].mean()
            if len(yearly_revenue) >= 2:
                growth_rate = (yearly_revenue.iloc[-1] / yearly_revenue.iloc[-2] - 1) * 100
                if growth_rate < -10:
                    issues.append(f"📉 Revenue decline: {abs(growth_rate):.1f}% decrease from previous period")
        
        # Display issues
        if issues:
            st.subheader("🔍 Auto-Detected Issues")
            for issue in issues:
                st.warning(issue)
        else:
            st.success("✅ No major issues detected in your data!")

def create_industry_benchmarks(business_type, current_metrics):
    """Create industry benchmark comparisons"""
    # Simplified benchmark data (in real app, this would come from external API)
    benchmarks = {
        "Sales & Revenue Analysis": {
            "Profit_Margin": {"good": 15, "average": 8, "poor": 3},
            "Growth_Rate": {"good": 20, "average": 10, "poor": 0}
        },
        "Marketing ROI Analysis": {
            "ROI": {"good": 400, "average": 200, "poor": 100},
            "Conversion_Rate": {"good": 5, "average": 2.5, "poor": 1}
        },
        "Financial Performance": {
            "Net_Profit_Margin": {"good": 12, "average": 6, "poor": 2},
            "ROI": {"good": 25, "average": 15, "poor": 5}
        }
    }
    
    return benchmarks.get(business_type, {})

def create_whatif_interface():
    """Create the What-If scenario analysis interface"""
    st.header("🔮 What-If Scenario Analysis")
    
    if not st.session_state.get("analysis_run_successfully"):
        st.warning("⚠️ Please run the main analysis first to enable What-If scenarios.")
        return
    
    # Get the current data and model
    data_eng = st.session_state.get("data_engineered")
    final_model = st.session_state.get("final_model")
    X_data = st.session_state.get("X_data")
    
    if data_eng is None or final_model is None or X_data is None:
        st.error("❌ Required data not found. Please run the analysis first.")
        return
    
    # Create baseline if not exists
    if st.session_state.whatif_baseline is None:
        st.session_state.whatif_baseline = create_baseline_scenario(data_eng, X_data)
    
    st.info("💡 Adjust parameters below to see how changes affect your target metric predictions.")
    
    # Scenario tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Create Scenarios", "📈 Compare Results", "💾 Saved Scenarios", "📋 Summary Report"])
    
    with tab1:
        create_scenario_builder(data_eng, X_data, final_model)
    
    with tab2:
        display_scenario_comparison()
    
    with tab3:
        manage_saved_scenarios()
    
    with tab4:
        generate_whatif_report()

def create_baseline_scenario(data_eng, X_data):
    """Create baseline scenario from current data"""
    baseline = {}
    
    # Get numerical columns that are adjustable
    adjustable_cols = get_adjustable_columns(data_eng)
    
    for col in adjustable_cols:
        if col in data_eng.columns:
            baseline[col] = {
                'current_value': float(data_eng[col].mean()),
                'min_value': float(data_eng[col].min()),
                'max_value': float(data_eng[col].max()),
                'std_value': float(data_eng[col].std())
            }
    
    return baseline

def get_adjustable_columns(data_eng):
    """Get columns that can be adjusted in what-if scenarios"""
    # Business-relevant columns that users can typically control
    priority_columns = [
        'Revenue', 'Marketing_Spend', 'R_D_Investment', 'Employee_Count',
        'Units_Sold', 'Product_Price', 'Operating_Costs', 'Investment',
        'Marketing_Budget', 'Advertising_Spend', 'Sales_Team_Size',
        'Customer_Acquisition_Cost', 'Pricing', 'Capacity', 'Volume'
    ]
    
    numeric_cols = data_eng.select_dtypes(include=['number']).columns.tolist()
    
    # Find matching columns (case insensitive)
    adjustable = []
    for priority in priority_columns:
        for col in numeric_cols:
            if priority.lower() in col.lower():
                adjustable.append(col)
    
    # Add other numeric columns that seem business-relevant
    business_keywords = ['cost', 'price', 'revenue', 'sales', 'profit', 'spend', 'investment', 'budget']
    for col in numeric_cols:
        if col not in adjustable:
            if any(keyword in col.lower() for keyword in business_keywords):
                adjustable.append(col)
    
    return list(set(adjustable))[:15]  # Limit to 15 most relevant columns

def create_scenario_builder(data_eng, X_data, final_model):
    """Build the scenario creation interface"""
    st.subheader("🎯 Scenario Builder")
    
    # Scenario naming
    col1, col2 = st.columns([3, 1])
    with col1:
        scenario_name = st.text_input("Scenario Name:", value=f"Scenario_{len(st.session_state.whatif_scenarios)+1}")
    with col2:
        scenario_type = st.selectbox("Type:", ["Optimistic", "Pessimistic", "Realistic", "Custom"])
    
    adjustable_cols = get_adjustable_columns(data_eng)
    baseline = st.session_state.whatif_baseline
    
    if not adjustable_cols:
        st.warning("No adjustable parameters found in your dataset.")
        return
    
    st.markdown("### 📊 Adjust Parameters")
    
    # Quick scenario templates
    if st.button("🚀 Apply Quick Scenario Template"):
        apply_scenario_template(scenario_type, adjustable_cols, baseline)
    
    # Parameter adjustment interface
    scenario_values = {}
    
    # Organize parameters into columns
    num_cols = min(3, len(adjustable_cols))
    cols = st.columns(num_cols)
    
    for i, col in enumerate(adjustable_cols):
        if col in baseline:
            with cols[i % num_cols]:
                current = baseline[col]['current_value']
                min_val = baseline[col]['min_value']
                max_val = baseline[col]['max_value']
                std_val = baseline[col]['std_value']
                
                # Create reasonable bounds for slider
                slider_min = max(min_val, current - 3 * std_val)
                slider_max = min(max_val, current + 3 * std_val)
                
                # Percentage change input
                pct_change = st.slider(
                    f"{col.replace('_', ' ')}",
                    min_value=-50.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    help=f"Current: {current:.2f}"
                )
                
                new_value = current * (1 + pct_change / 100)
                scenario_values[col] = new_value
                
                # Show the impact
                st.caption(f"📊 {current:.2f} → {new_value:.2f} ({pct_change:+.0f}%)")
    
    # Predict scenario outcome
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Predict Scenario Impact", type="primary"):
            prediction_result = predict_scenario_impact(scenario_values, X_data, final_model, data_eng)
            if prediction_result:
                st.session_state.current_prediction = prediction_result
                display_scenario_prediction(prediction_result, scenario_name)
    
    with col2:
        if st.button("💾 Save Scenario") and 'current_prediction' in st.session_state:
            save_scenario(scenario_name, scenario_values, st.session_state.current_prediction)
            st.success(f"✅ Scenario '{scenario_name}' saved!")

def apply_scenario_template(scenario_type, adjustable_cols, baseline):
    """Apply predefined scenario templates"""
    templates = {
        "Optimistic": {
            "revenue_boost": 20,
            "cost_reduction": -10,
            "efficiency_gain": 15
        },
        "Pessimistic": {
            "revenue_decline": -15,
            "cost_increase": 10,
            "efficiency_loss": -10
        },
        "Realistic": {
            "moderate_growth": 5,
            "slight_cost_increase": 3,
            "steady_improvement": 2
        }
    }
    
    template = templates.get(scenario_type, {})
    
    # Apply template changes to relevant columns
    for col in adjustable_cols:
        col_lower = col.lower()
        current = baseline[col]['current_value']
        
        if 'revenue' in col_lower or 'sales' in col_lower:
            if scenario_type == "Optimistic":
                baseline[col]['template_value'] = current * 1.20
            elif scenario_type == "Pessimistic":
                baseline[col]['template_value'] = current * 0.85
            else:
                baseline[col]['template_value'] = current * 1.05
                
        elif 'cost' in col_lower or 'expense' in col_lower:
            if scenario_type == "Optimistic":
                baseline[col]['template_value'] = current * 0.90
            elif scenario_type == "Pessimistic":
                baseline[col]['template_value'] = current * 1.10
            else:
                baseline[col]['template_value'] = current * 1.03
        else:
            # General efficiency changes
            if scenario_type == "Optimistic":
                baseline[col]['template_value'] = current * 1.15
            elif scenario_type == "Pessimistic":
                baseline[col]['template_value'] = current * 0.90
            else:
                baseline[col]['template_value'] = current * 1.02

def predict_scenario_impact(scenario_values, X_data, final_model, data_eng):
    """Predict the impact of scenario changes"""
    try:
        # Create a sample record for prediction
        baseline_record = X_data.mean().to_dict()
        
        # Update with scenario values
        for col, new_value in scenario_values.items():
            # Handle column name mapping (original vs engineered)
            if col in baseline_record:
                baseline_record[col] = new_value
            else:
                # Try to find similar column names
                for x_col in baseline_record.keys():
                    if col.lower().replace('_', '') in x_col.lower().replace('_', ''):
                        baseline_record[x_col] = new_value
                        break
        
        # Create DataFrame for prediction
        prediction_df = pd.DataFrame([baseline_record])
        
        # Make prediction
        predicted_value = final_model.predict(prediction_df)[0]
        
        # Calculate baseline prediction for comparison
        baseline_df = pd.DataFrame([X_data.mean().to_dict()])
        baseline_prediction = final_model.predict(baseline_df)[0]
        
        # Calculate impact metrics
        impact = predicted_value - baseline_prediction
        impact_percentage = (impact / baseline_prediction * 100) if baseline_prediction != 0 else 0
        
        return {
            'scenario_values': scenario_values,
            'predicted_value': predicted_value,
            'baseline_prediction': baseline_prediction,
            'impact': impact,
            'impact_percentage': impact_percentage,
            'prediction_record': baseline_record
        }
        
    except Exception as e:
        st.error(f"Error predicting scenario impact: {str(e)}")
        return None

def display_scenario_prediction(prediction_result, scenario_name):
    """Display the prediction results with visualizations"""
    st.subheader(f"🎯 Prediction Results: {scenario_name}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Predicted Value",
            f"{prediction_result['predicted_value']:,.2f}",
            f"{prediction_result['impact']:+,.2f}"
        )
    
    with col2:
        st.metric(
            "Baseline Value", 
            f"{prediction_result['baseline_prediction']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Impact",
            f"{prediction_result['impact']:+,.2f}",
            f"{prediction_result['impact_percentage']:+.1f}%"
        )
    
    with col4:
        impact_status = "Positive" if prediction_result['impact'] > 0 else "Negative"
        st.metric("Impact Status", impact_status)
    
    # Visualization
    create_scenario_impact_chart(prediction_result)

def create_scenario_impact_chart(prediction_result):
    """Create visualization for scenario impact"""
    
    # Impact waterfall chart
    categories = ['Baseline', 'Impact', 'Predicted']
    values = [
        prediction_result['baseline_prediction'],
        prediction_result['impact'],
        prediction_result['predicted_value']
    ]
    
    fig = go.Figure(go.Waterfall(
        name="Scenario Impact",
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=categories,
        textposition="outside",
        text=[f"{v:,.1f}" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="🔮 Scenario Impact Analysis",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_scenario_comparison():
    """Display comparison of multiple scenarios"""
    st.subheader("📊 Scenario Comparison")
    
    saved_scenarios = st.session_state.whatif_scenarios
    
    if len(saved_scenarios) < 2:
        st.info("💡 Save at least 2 scenarios to enable comparison.")
        return
    
    # Select scenarios to compare
    selected_scenarios = st.multiselect(
        "Select scenarios to compare:",
        options=list(saved_scenarios.keys()),
        default=list(saved_scenarios.keys())[:3]  # Default to first 3
    )
    
    if len(selected_scenarios) < 2:
        st.warning("Please select at least 2 scenarios for comparison.")
        return
    
    # Comparison chart
    create_scenario_comparison_chart(selected_scenarios, saved_scenarios)
    
    # Comparison table
    create_scenario_comparison_table(selected_scenarios, saved_scenarios)

def create_scenario_comparison_chart(selected_scenarios, saved_scenarios):
    """Create comparison charts for multiple scenarios"""
    
    # Prepare data for comparison
    scenario_names = selected_scenarios
    predicted_values = [saved_scenarios[name]['prediction']['predicted_value'] for name in scenario_names]
    impacts = [saved_scenarios[name]['prediction']['impact'] for name in scenario_names]
    impact_percentages = [saved_scenarios[name]['prediction']['impact_percentage'] for name in scenario_names]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Predicted Values', 'Impact Amount', 'Impact Percentage', 'Risk vs Return'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Predicted values
    fig.add_trace(
        go.Bar(x=scenario_names, y=predicted_values, name="Predicted Values", showlegend=False),
        row=1, col=1
    )
    
    # Impact amounts
    colors = ['green' if impact > 0 else 'red' for impact in impacts]
    fig.add_trace(
        go.Bar(x=scenario_names, y=impacts, name="Impact", marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    # Impact percentages
    fig.add_trace(
        go.Bar(x=scenario_names, y=impact_percentages, name="Impact %", marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # Risk vs return (simplified)
    risk_scores = [abs(impact) for impact in impacts]  # Simple risk measure
    fig.add_trace(
        go.Scatter(x=risk_scores, y=predicted_values, mode='markers+text',
                  text=scenario_names, textposition="top center", 
                  marker=dict(size=12, color=colors), showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="📊 Multi-Scenario Analysis Dashboard")
    st.plotly_chart(fig, use_container_width=True)

def create_scenario_comparison_table(selected_scenarios, saved_scenarios):
    """Create detailed comparison table"""
    
    comparison_data = []
    for name in selected_scenarios:
        scenario = saved_scenarios[name]
        comparison_data.append({
            'Scenario': name,
            'Predicted Value': f"{scenario['prediction']['predicted_value']:,.2f}",
            'Impact': f"{scenario['prediction']['impact']:+,.2f}",
            'Impact %': f"{scenario['prediction']['impact_percentage']:+.1f}%",
            'Status': "✅ Positive" if scenario['prediction']['impact'] > 0 else "❌ Negative"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

def save_scenario(scenario_name, scenario_values, prediction_result):
    """Save a scenario for later comparison"""
    st.session_state.whatif_scenarios[scenario_name] = {
        'values': scenario_values,
        'prediction': prediction_result,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def manage_saved_scenarios():
    """Manage saved scenarios"""
    st.subheader("💾 Saved Scenarios Management")
    
    saved_scenarios = st.session_state.whatif_scenarios
    
    if not saved_scenarios:
        st.info("No saved scenarios yet. Create and save scenarios in the builder tab.")
        return
    
    # Display saved scenarios
    saved_scenarios = st.session_state.get("whatif_scenarios", {})
    saved_scenarios = st.session_state.get("whatif_scenarios", {})
    for name, scenario in saved_scenarios.items():
      with st.expander(f"📋 {name} (Created: {scenario['created_at']})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Parameter Changes:**")
                for param, value in scenario['values'].items():
                    baseline_val = st.session_state.whatif_baseline.get(param, {}).get('current_value', 0)
                    change_pct = ((value - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
                    st.write(f"• {param}: {baseline_val:.2f} → {value:.2f} ({change_pct:+.1f}%)")
                
                st.write("**Prediction Results:**")
                pred = scenario['prediction']
                st.write(f"• Predicted Value: {pred['predicted_value']:,.2f}")
                st.write(f"• Impact: {pred['impact']:+,.2f} ({pred['impact_percentage']:+.1f}%)")
            
            with col2:
                if st.button(f"🗑️ Delete", key=f"delete_{name}"):
                    del st.session_state.whatif_scenarios[name]
                    st.rerun()
def generate_whatif_report():
    """Generate What-If analysis report"""
    st.subheader("📋 What-If Analysis Summary")
    
    saved_scenarios = st.session_state.whatif_scenarios
    
    if not saved_scenarios:
        st.info("No scenarios to report. Create scenarios first.")
        return
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    impacts = [s['prediction']['impact'] for s in saved_scenarios.values()]
    
    with col1:
        st.metric("Total Scenarios", len(saved_scenarios))
    
    with col2:
        positive_scenarios = sum(1 for impact in impacts if impact > 0)
        st.metric("Positive Scenarios", f"{positive_scenarios}/{len(impacts)}")
    
    with col3:
        if impacts:
            avg_impact = np.mean(impacts)
            st.metric("Average Impact", f"{avg_impact:+,.2f}")
    
    # Best and worst scenarios
    if impacts:
        best_scenario = max(saved_scenarios.items(), key=lambda x: x[1]['prediction']['impact'])
        worst_scenario = min(saved_scenarios.items(), key=lambda x: x[1]['prediction']['impact'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🏆 **Best Scenario:** {best_scenario[0]}")
            st.write(f"Impact: {best_scenario[1]['prediction']['impact']:+,.2f}")
            st.write(f"Predicted Value: {best_scenario[1]['prediction']['predicted_value']:,.2f}")
        
        with col2:
            st.error(f"⚠️ **Worst Scenario:** {worst_scenario[0]}")
            st.write(f"Impact: {worst_scenario[1]['prediction']['impact']:+,.2f}")
            st.write(f"Predicted Value: {worst_scenario[1]['prediction']['predicted_value']:,.2f}")
    
    # Export scenarios
    if st.button("📤 Export Scenarios to CSV"):
        export_scenarios_to_csv(saved_scenarios)

def export_scenarios_to_csv(saved_scenarios):
    """Export scenarios to CSV for download"""
    export_data = []
    
    for name, scenario in saved_scenarios.items():
        row = {
            'Scenario_Name': name,
            'Created_At': scenario['created_at'],
            'Predicted_Value': scenario['prediction']['predicted_value'],
            'Impact': scenario['prediction']['impact'],
            'Impact_Percentage': scenario['prediction']['impact_percentage']
        }
        
        # Add parameter values
        for param, value in scenario['values'].items():
            row[f'Param_{param}'] = value
        
        export_data.append(row)
    
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"whatif_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


# ===========================
# FOCUSED ANALYTICS FUNCTIONS
# ===========================
def create_focused_analytics_by_type_enhanced(df, analysis_type):
    """Enhanced version that collects comprehensive insights"""
    insights_collected = {}
    
    st.session_state["current_analysis_type"] = analysis_type

    if analysis_type == "💰 Profit & Loss (P/L)":
        st.header("💰 Profit & Loss Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Waterfall Chart
            try:
                revenue = df['Revenue'].sum() if 'Revenue' in df.columns else 0
                net_profit = df['Net_Profit'].sum() if 'Net_Profit' in df.columns else 0
                if revenue > 0:
                    create_waterfall_chart(df)
                    profit_margin = (net_profit / revenue * 100) if revenue > 0 else 0
                    insights_collected['Financial Waterfall'] = f"Revenue of ${revenue:,.0f} converts to net profit of ${net_profit:,.0f}, representing a {profit_margin:.1f}% profit margin."
            except:
                pass
            
            # Break-even Analysis
            try:
                create_breakeven_chart(df)
                if 'Revenue_Per_Unit' in df.columns and 'Cost_Per_Unit' in df.columns:
                    avg_revenue_per_unit = df['Revenue_Per_Unit'].mean()
                    avg_cost_per_unit = df['Cost_Per_Unit'].mean()
                    profit_per_unit = avg_revenue_per_unit - avg_cost_per_unit
                    insights_collected['Break-Even Analysis'] = f"Each unit generates ${avg_revenue_per_unit:.2f} revenue with ${avg_cost_per_unit:.2f} costs, yielding ${profit_per_unit:.2f} profit per unit."
            except:
                pass
        
        with col2:
            # Profitability Trend
            try:
                create_profitability_trend(df)
                time_cols = [col for col in ['Year', 'Quarter', 'Month'] if col in df.columns]
                if time_cols and 'Net_Profit' in df.columns:
                    time_col = time_cols[0]
                    trend_data = df.groupby(time_col)['Net_Profit'].mean()
                    if len(trend_data) >= 2:
                        latest_profit = trend_data.iloc[-1]
                        previous_profit = trend_data.iloc[-2]
                        growth_rate = ((latest_profit - previous_profit) / abs(previous_profit) * 100) if previous_profit != 0 else 0
                        insights_collected['Profitability Trend'] = f"Profitability has {'increased' if growth_rate > 0 else 'decreased'} by {abs(growth_rate):.1f}% in the latest period."
            except:
                pass
            
            # Segment Profitability
            try:
                create_segment_profitability(df)
                segment_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['product', 'segment', 'category', 'region']) and df[col].dtype == 'object' and df[col].nunique() < 20]
                if segment_cols and 'Net_Profit' in df.columns:
                    segment_col = segment_cols[0]
                    seg_data = df.groupby(segment_col)['Net_Profit'].sum()
                    insights_collected['Segment Analysis'] = f"'{seg_data.idxmax()}' is the most profitable segment, while '{seg_data.idxmin()}' contributes the least profit."
            except:
                pass
        
        try:
            create_profitability_roi_heatmap(df)
        except:
            pass
            
    elif analysis_type == "📈 Return on Investment (ROI)":
        st.header("📈 Return on Investment Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                create_roi_funnel(df)
                if 'ROI' in df.columns:
                    roi_data = df['ROI']
                    avg_roi = roi_data.mean()
                    positive_roi_pct = (roi_data > 0).mean() * 100
                    insights_collected['ROI Distribution'] = f"Average ROI across all investments is {avg_roi:.1f}% with {positive_roi_pct:.0f}% of investments showing positive returns."
            except:
                pass
        
        with col2:
            try:
                create_cumulative_roi_curves(df)
                if 'ROI' in df.columns:
                    cumulative_roi = df['ROI'].sum()
                    insights_collected['Cumulative Performance'] = f"Total cumulative ROI across all investments is {cumulative_roi:.1f}%, demonstrating {'strong' if cumulative_roi > 100 else 'moderate' if cumulative_roi > 0 else 'poor'} overall performance."
            except:
                pass
        
        try:
            create_roi_by_project_year(df)
        except:
            pass
        
        try:
            create_roi_vs_risk_scatter(df)
            if 'ROI' in df.columns:
                roi_std = df['ROI'].std()
                insights_collected['Investment Risk'] = f"ROI volatility is {roi_std:.1f}%, indicating {'high' if roi_std > 50 else 'moderate' if roi_std > 25 else 'low'} investment risk."
        except:
            pass
    
    # Store insights in session state for PDF generation
    st.session_state.chart_insights = {k: v for k, v in insights_collected.items() if v is not None and v.strip()}
       
def create_growth_analysis(df):
    st.subheader("📈 Growth Trends")
    if 'Year' in df.columns and 'Revenue' in df.columns:
        growth_data = df.groupby('Year')['Revenue'].sum().pct_change().fillna(0) * 100
        st.metric("Average Annual Growth", f"{growth_data.mean():.2f}%")
        st.line_chart(df.groupby('Year')['Revenue'].sum())
    else:
        st.info("Growth analysis requires 'Year' and 'Revenue' columns.")

def create_top_performers_analysis(df):
    st.subheader("🏆 Top Performers")
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 10]
    if cat_cols and 'Net_Profit' in df.columns:
        performer_col = st.selectbox("Select category for performance analysis", cat_cols)
        top_5 = df.groupby(performer_col)['Net_Profit'].sum().nlargest(5)
        st.bar_chart(top_5)
    else:
        st.info("Top performer analysis requires categorical columns and 'Net_Profit'.")

def create_predictive_insights_summary(df):
    st.subheader("💡 Predictive Insights Summary")
    st.info("A summary of predictive insights and model findings would appear here.")

def create_seasonal_analysis(df):
    st.subheader("🔄 Seasonal Patterns")
    if 'Season' in df.columns and 'Revenue' in df.columns:
        seasonal_data = df.groupby('Season')['Revenue'].mean()
        st.bar_chart(seasonal_data)
    else:
        st.info("Seasonal analysis requires a 'Season' column and a numeric column like 'Revenue'.")

def get_visualization_categories():
    """Define which visualizations belong to P/L vs ROI categories"""
    return {
        'P/L': [
            'create_waterfall_chart',
            'create_profitability_trend', 
            'create_breakeven_chart',
            'create_segment_profitability',
            'create_profitability_roi_heatmap'
        ],
        'ROI': [
            'create_roi_funnel',
            'create_roi_by_project_year',
            'create_cumulative_roi_curves', 
            'create_roi_vs_risk_scatter'
        ],
        'Misc': [
            'create_growth_analysis',
            'create_top_performers_analysis',
            'create_seasonal_analysis',
            'detect_business_issues'
        ]
    }

def create_comparison_analysis(df, dimension):
    st.subheader(f"🔄 Comparison Analysis by {dimension}")
    if 'Net_Profit' in df.columns:
        comparison_data = df.groupby(dimension)['Net_Profit'].mean()
        st.bar_chart(comparison_data)
    else:
        st.info(f"Comparison analysis requires the '{dimension}' and 'Net_Profit' columns.")
        
def create_comprehensive_analytics(df):
    """Default comprehensive analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        create_profitability_trend(df)
        create_breakeven_chart(df)
        create_roi_funnel(df)
    
    with col2:
        create_waterfall_chart(df)
        create_segment_profitability(df)
        create_profitability_roi_heatmap(df)
    
    # Advanced ROI Analysis
    st.markdown("---")
    st.header("📊 Advanced ROI Analytics")
    
    create_roi_by_project_year(df)
    
    col3, col4 = st.columns(2)
    with col3:
        create_cumulative_roi_curves(df)
    with col4:
        create_roi_vs_risk_scatter(df)

def create_benchmark_comparison(business_type, current_metrics):
    """Create industry benchmark comparison"""
    st.subheader("🏭 Industry Benchmark Comparison")
    
    benchmarks = create_industry_benchmarks(business_type, current_metrics)
    
    if benchmarks:
        col1, col2, col3 = st.columns(3)
        
        for i, (metric, benchmark_data) in enumerate(benchmarks.items()):
            current_value = current_metrics.get(metric, 0)
            
            col = [col1, col2, col3][i % 3]
            with col:
                # Determine performance level
                if current_value >= benchmark_data['good']:
                    status = "🟢 Excellent"
                    color = "green"
                elif current_value >= benchmark_data['average']:
                    status = "🟡 Good"
                    color = "orange"
                else:
                    status = "🔴 Needs Improvement"
                    color = "red"
                
                st.metric(
                    f"{metric} vs Industry",
                    f"{current_value:.1f}%",
                    f"{status} (Avg: {benchmark_data['average']}%)"
                )

def create_business_feature_insights(model, X, target_name, business_type, language_choice, analysis_type=None):
    """Business-friendly feature importance and AI recommendations."""
    st.subheader("🔍 Key Factors Driving Your Results")

    try:
        # Determine feature importances from the model pipeline
        if hasattr(model.named_steps["model"], "feature_importances_"):
            importances = model.named_steps["model"].feature_importances_
        elif hasattr(model.named_steps["model"], "coef_"):
            importances = np.abs(model.named_steps["model"].coef_)
        else:
            st.info("Feature importance analysis not available for this model type.")
            return None  # Return None if no feature importance available

        # Create and display the feature importance dataframe and chart
        fi_df = pd.DataFrame({
            "Factor": [name.replace("_", " ").title() for name in X.columns[:len(importances)]],
            "Impact_Score": importances
        }).sort_values("Impact_Score", ascending=False).head(10)

        # Store feature_df in session state for PDF generation
        st.session_state["feature_df"] = fi_df

        # --- Enhanced: Call the AI recommendations function with chart insights ---
        if st.session_state.get("kpi_data"):
            # Get analysis type and chart insights from session state
            current_analysis_type = st.session_state.get("current_analysis_type")
            chart_insights = st.session_state.get("chart_insights", {})
            
            generate_ai_recommendations(
                kpis=st.session_state["kpi_data"],
                feature_df=fi_df,
                business_context=business_type,
                target_variable=target_name,
                language=language_choice,
                analysis_type=current_analysis_type,
                chart_insights=chart_insights
            )
        
        return fi_df  # Return the dataframe for PDF use

    except Exception as e:
        st.error(f"Could not analyze key factors: {str(e)}")
        return None

def translate_feature_name(technical_name):
    """Convert technical feature names to business-friendly names"""
    translations = {
        "Marketing_Spend": "Marketing Investment",
        "R_D_Investment": "R&D Investment", 
        "Employee_Count": "Team Size",
        "Units_Sold": "Sales Volume",
        "Customer_Satisfaction": "Customer Satisfaction",
        "Market_Share": "Market Position",
        "Product_Price": "Pricing Strategy",
        "Revenue_Per_Unit": "Unit Revenue",
        "Cost_Per_Unit": "Unit Cost",
        "Profit_Margin": "Profit Margins",
        "ROI": "Return on Investment",
        "Operating_Costs": "Operating Expenses"
    }
    
    for tech, business in translations.items():
        if tech.lower() in technical_name.lower():
            return business
    
    # Clean up the name if no translation found
    return technical_name.replace("_", " ").title()

def generate_chart_insights(df, analysis_type="💰 Profit & Loss (P/L)"):
    """
    Collect insights from all visualization functions based on analysis type.
    Returns a dictionary of {chart_name: insight}.
    """
    insights = {}

    if analysis_type == "💰 Profit & Loss (P/L)":
        functions = [
            create_waterfall_chart,
            create_profitability_trend,
            create_breakeven_chart,
            create_segment_profitability,
            create_profitability_roi_heatmap
        ]
    elif analysis_type == "📈 Return on Investment (ROI)":
        functions = [
            create_roi_funnel,
            create_roi_by_project_year,
            create_cumulative_roi_curves,
            create_roi_vs_risk_scatter
        ]
    else:
        functions = []

    for func in functions:
        try:
            # Capture the output without displaying
            result = func(df)
            if result and isinstance(result, dict):
                insights[result["chart"]] = result["insight"]
        except Exception as e:
            # Don't show warnings for missing data - this is expected
            pass

    return insights

def format_chart_insights_for_display(chart_insights):
    """Format chart insights for better readability in AI prompt"""
    formatted_insights = {}
    
    for chart_name, insight in chart_insights.items():
        # Clean up the insight text and make it more descriptive
        if "Waterfall Chart" in chart_name:
            formatted_insights["Financial Waterfall Analysis"] = f"Your revenue flow analysis shows: {insight}"
        elif "Break-even" in chart_name:
            formatted_insights["Break-even Performance"] = f"Unit economics analysis reveals: {insight}"
        elif "Profitability Trend" in chart_name:
            formatted_insights["Profit Trend Analysis"] = f"Time-based profitability shows: {insight}"
        elif "Segment Profitability" in chart_name:
            formatted_insights["Business Segment Performance"] = f"Segment comparison indicates: {insight}"
        elif "ROI Funnel" in chart_name:
            formatted_insights["Investment Performance Distribution"] = f"ROI analysis shows: {insight}"
        elif "Cumulative ROI" in chart_name:
            formatted_insights["Overall Investment Returns"] = f"Cumulative performance indicates: {insight}"
        elif "ROI by Project" in chart_name:
            formatted_insights["Project/Category Performance"] = f"Performance by category shows: {insight}"
        else:
            formatted_insights[chart_name] = insight
    
    return formatted_insights

def generate_ai_recommendations(kpis, feature_df, business_context, target_variable, language, analysis_type=None, chart_insights=None):
    """Generates AI-powered business recommendations using Gemini with visualization explanations."""
    st.subheader("💡 AI-Powered Actionable Insights")
    
    chart_insights = st.session_state.get("chart_insights", {})

    # Get analysis type from session state or parameter
    if not analysis_type:
        analysis_type = st.session_state.get("analysis_type", "💰 Profit & Loss (P/L)")

    # Define visualization explanations for different analysis types
    visualization_explanations = {
        "💰 Profit & Loss (P/L)": {
            "Waterfall Chart": "Shows the step-by-step breakdown from Revenue to Net Profit, helping identify where money is gained or lost in your business process.",
            "Break-even Analysis": "Displays the point where total revenue equals total costs, indicating the minimum sales volume needed to avoid losses.",
            "Profitability Trend": "Tracks how profit margins change over time periods, revealing seasonal patterns and long-term business health trends.",
            "Segment Profitability": "Compares profit performance across different business segments, products, or regions to identify your most and least profitable areas.",
            "Profitability vs ROI Heatmap": "Visualizes the relationship between profitability levels and ROI efficiency, helping identify optimal investment opportunities."
        },
        "📈 Return on Investment (ROI)": {
            "ROI Funnel": "Categorizes investments by ROI performance levels (Loss, Low, Good, Excellent), showing the distribution of your investment success rates.",
            "ROI by Project & Year": "Analyzes ROI trends across different projects and time periods, identifying which investments perform best over time.",
            "Cumulative ROI Curves": "Tracks the cumulative return performance over time, showing whether your overall investment strategy is improving or declining.",
            "ROI vs Risk Scatter": "Maps the relationship between investment risk and returns, helping identify high-reward, low-risk opportunities in your portfolio."
        }
    }

    # Format the data for the prompt
    try:
        kpi_summary = "\n".join([f"- {key.replace('_', ' ')}: {value:,.2f}" for key, value in kpis.items()])
        top_factors = "\n".join([f"- {row['Factor']} (Impact Score: {row['Impact_Score']:.2f})" for index, row in feature_df.head(3).iterrows()])
        viz_explanation_text = "\n".join([f"- **{viz_name}**: {insight}" for viz_name, insight in chart_insights.items()]) if chart_insights else "No specific insights from charts."
    except Exception as e:
        st.error(f"Error preparing data for AI prompt: {e}")
        return

    # Get visualization explanations for current analysis type
    formatted_insights = format_chart_insights_for_display(chart_insights)
    viz_explanation_text = "\n".join([f"- **{viz_name}**: {insight}" for viz_name, insight in formatted_insights.items()])

    prompt = f"""
    As an expert business analyst, your task is to provide actionable recommendations based on the data below.
    Analyze the KPIs, the key factors influencing the business outcome, and the visualizations displayed.
    Provide 4-5 concise, actionable bullet-point recommendations focused on what is lacking and what should be improved.
    Tailor your advice specifically to the business context provided.

    **IMPORTANT: Generate the entire response ONLY in the {language} language.**

    ---
    **Business Data:**
    - **Business Context:** {business_context}
    - **Analysis Focus:** {analysis_type}
    - **Primary Goal:** Improve {target_variable.replace('_', ' ')}

    **Key Performance Indicators (KPIs):**
    {kpi_summary}

    **Top 3 Factors Influencing {target_variable.replace('_', ' ')}:**
    {top_factors}

    **Visualizations Displayed ({analysis_type}):**
    {viz_explanation_text}

    ---

    **Format your response with two sections in {language}:**

    **📊 Visualization Insights:**
    Briefly explain what each visualization reveals about the business performance (1-2 lines per chart).

    **🎯 Actionable Recommendations:**
    Provide 4-5 specific, actionable recommendations based on the data analysis.

    **Response in {language}:**
    """

    try:
        with st.spinner(f"🤖 Generating AI recommendations in {language}..."):
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Add generation configuration for better results
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1500,
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                st.markdown(response.text)
            else:
                st.warning("⚠️ No response generated. Please try again.")
                
    except Exception as e:
        st.error(f"❌ Error generating AI insights: {str(e)}")

def generate_pdf_report(language, style):
    """Generate PDF report in selected language and style"""
    with st.spinner(f"Generating {style} report in {language}..."):
        # Placeholder for PDF generation
        time.sleep(2)  # Simulate processing
        st.success(f"✅ {style} report generated in {language}!")
        st.download_button(
            "📥 Download PDF Report",
            data="dummy_pdf_content",  # Replace with actual PDF bytes
            file_name=f"business_analysis_{style.lower().replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
    
def generate_ai_content_for_pdf(kpis, feature_df, business_context, target_variable, language, analysis_type, report_style):
    """Generate AI content for different PDF report sections using Gemini"""
    
    # Format data for AI processing
    kpi_summary = "\n".join([f"- {key.replace('_', ' ')}: {value:,.2f}" for key, value in kpis.items()])
    top_factors = "\n".join([f"- {row['Factor']} (Impact Score: {row['Impact_Score']:.2f})" 
                            for index, row in feature_df.head(5).iterrows()])
    
    prompts = {
        "Executive Summary": f"""
        As a senior business consultant, create an EXECUTIVE SUMMARY in {language} for a {business_context} analysis.
        
        **Data Overview:**
        - Business Focus: {analysis_type}
        - Target Variable: {target_variable}
        - Key Metrics: {kpi_summary}
        - Top Success Factors: {top_factors}
        
        **Required Sections (in {language}):**
        1. **Business Performance Overview** (2-3 sentences)
        2. **Key Findings** (3-4 bullet points)
        3. **Critical Success Factors** (top 3 factors)
        4. **Strategic Recommendations** (3-4 action items)
        5. **Financial Impact** (expected outcomes)
        
        Keep it concise, executive-level, and actionable. Focus on high-level strategic insights.
        """,
        
        "Detailed Analysis": f"""
        As a business analyst, create a DETAILED ANALYSIS report in {language} for {business_context}.
        
        **Data Context:**
        - Analysis Type: {analysis_type}
        - Primary Goal: Optimize {target_variable}
        - Performance Metrics: {kpi_summary}
        - Key Drivers: {top_factors}
        
        **Required Sections (in {language}):**
        1. **Executive Summary** (brief overview)
        2. **Current Performance Analysis** (detailed KPI breakdown)
        3. **Factor Analysis** (detailed explanation of top 5 factors)
        4. **Trend Analysis** (patterns and insights)
        5. **Risk Assessment** (potential challenges)
        6. **Detailed Recommendations** (specific action plans with timelines)
        7. **Implementation Roadmap** (step-by-step approach)
        8. **Expected ROI** (quantified benefits)
        
        Provide comprehensive analysis with specific examples and metrics.
        """,
        
        "Technical Report": f"""
        As a data scientist, create a TECHNICAL REPORT in {language} for {business_context} analysis.
        
        **Technical Context:**
        - Model Target: {target_variable}
        - Business Domain: {business_context}
        - Analysis Focus: {analysis_type}
        - Performance Metrics: {kpi_summary}
        - Feature Importance: {top_factors}
        
        **Required Technical Sections (in {language}):**
        1. **Data Overview** (dataset characteristics, preprocessing steps)
        2. **Feature Engineering** (new variables created, transformations)
        3. **Model Performance** (accuracy metrics, validation results)
        4. **Feature Importance Analysis** (statistical significance, impact scores)
        5. **Model Interpretation** (how predictions are made)
        6. **Statistical Insights** (correlations, patterns, anomalies)
        7. **Technical Recommendations** (data quality improvements, model enhancements)
        8. **Limitations & Assumptions** (model constraints, data limitations)
        9. **Future Enhancements** (suggested improvements)
        
        Include technical details while keeping it business-relevant.
        """
    }
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2500,
        }
        
        response = model.generate_content(
            prompts[report_style],
            generation_config=generation_config
        )
        
        return response.text if response.text else "Content generation failed."
        
    except Exception as e:
        return f"Error generating content: {str(e)}"

def save_plotly_as_image(fig, filename="chart.png", width=800, height=600):
    """Save Plotly figure as image for PDF inclusion with error handling"""
    try:
        # Save as static image
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.write(img_bytes)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        print(f"Error saving chart: {e}")
        # Try with kaleido backend
        try:
            import plotly.io as pio
            pio.kaleido.scope.default_width = width
            pio.kaleido.scope.default_height = height
            img_bytes = pio.to_image(fig, format="png")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.write(img_bytes)
            temp_file.close()
            
            return temp_file.name
        except Exception as e2:
            print(f"Fallback image saving also failed: {e2}")
            return None

def save_matplotlib_as_image(fig, filename="chart.png"):
    """Save matplotlib figure as image"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, dpi=300, bbox_inches='tight')
        temp_file.close()
        return temp_file.name
    except Exception as e:
        st.error(f"Error saving matplotlib chart: {e}")
        return None

def create_summary_charts_for_pdf(df, kpis):
    """Create summary charts specifically for PDF inclusion"""
    chart_files = []
    
    try:
        # 1. KPI Summary Chart
        fig_kpi = go.Figure()
        
        kpi_names = []
        kpi_values = []
        
        for key, value in kpis.items():
            clean_name = key.replace('_', ' ').title()
            if 'Total' in clean_name or 'Average' in clean_name:
                kpi_names.append(clean_name)
                kpi_values.append(abs(value))  # Use absolute value for visualization
        
        if kpi_names:
            fig_kpi.add_trace(go.Bar(
                x=kpi_names,
                y=kpi_values,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(kpi_names)],
                text=[f"{v:,.0f}" for v in kpi_values],
                textposition='auto',
            ))
            
            fig_kpi.update_layout(
                title="Key Performance Indicators Overview",
                xaxis_title="Metrics",
                yaxis_title="Value",
                height=400,
                showlegend=False
            )
            
            kpi_chart_file = save_plotly_as_image(fig_kpi, "kpi_summary.png")
            if kpi_chart_file:
                chart_files.append(("KPI Summary", kpi_chart_file))
        
        # 2. Profitability Analysis (if available) - FIXED VERSION
        if 'Net_Profit' in df.columns:
            # Profit distribution
            profit_data = df['Net_Profit']
            positive_profit = (profit_data > 0).sum()
            negative_profit = (profit_data <= 0).sum()
            
            fig_profit = go.Figure(data=[
                go.Pie(
                    labels=['Profitable', 'Loss/Break-even'], 
                    values=[positive_profit, negative_profit],
                    marker=dict(colors=['#2ECC71', '#E74C3C'])  # Fixed: Use marker dict with colors
                )
            ])
            
            fig_profit.update_layout(
                title="Profitability Distribution",
                height=400
            )
            
            profit_chart_file = save_plotly_as_image(fig_profit, "profitability.png")
            if profit_chart_file:
                chart_files.append(("Profitability Analysis", profit_chart_file))
        
        # 3. Performance Trend (if time data available)
        time_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['year', 'month', 'quarter', 'date'])]
        
        if time_cols and 'Revenue' in df.columns:
            time_col = time_cols[0]
            trend_data = df.groupby(time_col)['Revenue'].sum().reset_index()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_data[time_col],
                y=trend_data['Revenue'],
                mode='lines+markers',
                name='Revenue Trend',
                line=dict(color='#3498DB', width=3),
                marker=dict(size=8)
            ))
            
            fig_trend.update_layout(
                title="Revenue Trend Analysis",
                xaxis_title=time_col,
                yaxis_title="Revenue",
                height=400
            )
            
            trend_chart_file = save_plotly_as_image(fig_trend, "revenue_trend.png")
            if trend_chart_file:
                chart_files.append(("Revenue Trend", trend_chart_file))
    
    except Exception as e:
        st.error(f"Error creating charts: {e}")
    
    return chart_files

def create_all_analysis_charts_for_pdf(df, analysis_type, language="English"):
    """Create all charts for the selected analysis type and return file paths"""
    chart_files = []
    
    try:
        if analysis_type == "💰 Profit & Loss (P/L)":
            # 1. Waterfall Chart
            try:
                revenue = df['Revenue'].sum() if 'Revenue' in df.columns else 0
                gross_profit = df['Gross_Profit'].sum() if 'Gross_Profit' in df.columns else 0
                ebit = df['EBIT'].sum() if 'EBIT' in df.columns else 0
                net_profit = df['Net_Profit'].sum() if 'Net_Profit' in df.columns else 0
                
                if revenue > 0:
                    cogs = revenue - gross_profit if gross_profit > 0 else 0
                    operating_expenses = gross_profit - ebit if ebit > 0 and gross_profit > 0 else 0
                    other_expenses = ebit - net_profit if net_profit > 0 and ebit > 0 else 0
                    
                    categories = ['Revenue', 'COGS', 'Gross Profit', 'OpEx', 'EBIT', 'Other', 'Net Profit']
                    values = [revenue, -cogs, gross_profit, -operating_expenses, ebit, -other_expenses, net_profit]
                    
                    fig_waterfall = go.Figure(go.Waterfall(
                        name="Financial Flow",
                        orientation="v",
                        measure=["absolute", "relative", "total", "relative", "total", "relative", "total"],
                        x=categories,
                        textposition="outside",
                        text=[f"${v:,.0f}" for v in values],
                        y=values,
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    
                    fig_waterfall.update_layout(
                        title="Revenue to Net Profit Waterfall",
                        showlegend=False,
                        height=500,
                        width=800
                    )
                    
                    waterfall_file = save_plotly_as_image(fig_waterfall, "waterfall_chart.png", 800, 500)
                    if waterfall_file:
                        chart_files.append(("Financial Waterfall Analysis", waterfall_file))
            except Exception as e:
                print(f"Error creating waterfall chart: {e}")
            
            # 2. Profitability Trend
            try:
                time_cols = []
                for col in ['Year', 'Month', 'Quarter', 'Date']:
                    if col in df.columns:
                        time_cols.append(col)
                        break
                
                if time_cols and any(col in df.columns for col in ['Gross_Profit_Margin', 'EBIT_Margin', 'Profit_Margin']):
                    time_col = time_cols[0]
                    margin_cols = [col for col in ['Gross_Profit_Margin', 'EBIT_Margin', 'Profit_Margin'] if col in df.columns]
                    
                    if margin_cols:
                        trend_data = df.groupby(time_col)[margin_cols].mean().reset_index()
                        
                        fig_trend = go.Figure()
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                        
                        for i, col in enumerate(margin_cols):
                            fig_trend.add_trace(go.Scatter(
                                x=trend_data[time_col],
                                y=trend_data[col],
                                mode='lines+markers',
                                name=col.replace('_', ' '),
                                line=dict(color=colors[i % len(colors)], width=3),
                                marker=dict(size=8)
                            ))
                        
                        fig_trend.update_layout(
                            title="Profitability Margins Trend Over Time",
                            xaxis_title=time_col,
                            yaxis_title="Margin (%)",
                            height=500,
                            width=800,
                            showlegend=True
                        )
                        
                        trend_file = save_plotly_as_image(fig_trend, "profitability_trend.png", 800, 500)
                        if trend_file:
                            chart_files.append(("Profitability Trend Analysis", trend_file))
            except Exception as e:
                print(f"Error creating profitability trend: {e}")
            
            # 3. Break-even Analysis
            try:
                units_col = None
                for col in df.columns:
                    if 'units' in col.lower() or 'quantity' in col.lower():
                        units_col = col
                        break
                
                if units_col and 'Revenue' in df.columns:
                    max_units = int(df[units_col].max() * 1.2)
                    units_range = np.arange(0, max_units, max_units//50)
                    
                    avg_revenue_per_unit = df['Revenue_Per_Unit'].mean() if 'Revenue_Per_Unit' in df.columns else df['Revenue'].sum() / df[units_col].sum()
                    avg_cost_per_unit = df['Cost_Per_Unit'].mean() if 'Cost_Per_Unit' in df.columns else 0
                    
                    revenue_line = units_range * avg_revenue_per_unit
                    cost_line = units_range * avg_cost_per_unit
                    
                    fig_breakeven = go.Figure()
                    
                    fig_breakeven.add_trace(go.Scatter(
                        x=units_range, y=revenue_line,
                        mode='lines', name='Revenue',
                        line=dict(color='green', width=3)
                    ))
                    
                    fig_breakeven.add_trace(go.Scatter(
                        x=units_range, y=cost_line,
                        mode='lines', name='Costs',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig_breakeven.update_layout(
                        title="Break-Even Analysis: Units vs Revenue/Costs",
                        xaxis_title="Units Sold",
                        yaxis_title="Amount ($)",
                        height=500,
                        width=800
                    )
                    
                    breakeven_file = save_plotly_as_image(fig_breakeven, "breakeven_analysis.png", 800, 500)
                    if breakeven_file:
                        chart_files.append(("Break-Even Analysis", breakeven_file))
            except Exception as e:
                print(f"Error creating break-even analysis: {e}")
            
            # 4. Segment Profitability
            try:
                segment_cols = []
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['product', 'segment', 'category', 'region', 'department']):
                        if df[col].dtype == 'object' and df[col].nunique() < 20:
                            segment_cols.append(col)
                
                if segment_cols and 'Net_Profit' in df.columns:
                    segment_col = segment_cols[0]
                    segment_profit = df.groupby(segment_col)['Net_Profit'].sum().reset_index()
                    segment_profit = segment_profit.sort_values('Net_Profit', ascending=True)
                    
                    fig_segment = px.bar(
                        segment_profit, 
                        x='Net_Profit', 
                        y=segment_col,
                        orientation='h',
                        title=f"Profitability by {segment_col}",
                        color='Net_Profit',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    fig_segment.update_layout(height=500, width=800)
                    
                    segment_file = save_plotly_as_image(fig_segment, "segment_profitability.png", 800, 500)
                    if segment_file:
                        chart_files.append(("Segment Profitability Analysis", segment_file))
            except Exception as e:
                print(f"Error creating segment profitability: {e}")
        
        elif analysis_type == "📈 Return on Investment (ROI)":
            # 1. ROI Funnel
            try:
                if 'ROI' in df.columns:
                    roi_ranges = [
                        ('🔴 Loss (< 0%)', (df['ROI'] < 0).sum()),
                        ('🟡 Low (0-10%)', ((df['ROI'] >= 0) & (df['ROI'] <= 10)).sum()),
                        ('🟢 Good (10-25%)', ((df['ROI'] > 10) & (df['ROI'] <= 25)).sum()),
                        ('💚 Excellent (> 25%)', (df['ROI'] > 25).sum())
                    ]
                    
                    categories, counts = zip(*roi_ranges)
                    
                    fig_roi_funnel = go.Figure(data=[
                        go.Bar(x=categories, y=counts, 
                              marker_color=['red', 'orange', 'lightgreen', 'green'])
                    ])
                    
                    fig_roi_funnel.update_layout(
                        title="ROI Distribution Across Investments",
                        xaxis_title="ROI Range",
                        yaxis_title="Number of Investments",
                        height=500,
                        width=800
                    )
                    
                    roi_funnel_file = save_plotly_as_image(fig_roi_funnel, "roi_funnel.png", 800, 500)
                    if roi_funnel_file:
                        chart_files.append(("ROI Performance Distribution", roi_funnel_file))
            except Exception as e:
                print(f"Error creating ROI funnel: {e}")
            
            # 2. Cumulative ROI
            try:
                if 'ROI' in df.columns:
                    df_temp = df.copy().reset_index()
                    df_temp['Index'] = range(len(df_temp))
                    df_temp = df_temp.sort_values('Index').reset_index(drop=True)
                    df_temp['Cumulative_ROI'] = df_temp['ROI'].cumsum()
                    
                    fig_cumulative = go.Figure()
                    
                    fig_cumulative.add_trace(go.Scatter(
                        x=df_temp['Index'],
                        y=df_temp['Cumulative_ROI'],
                        mode='lines',
                        name='Cumulative ROI',
                        line=dict(color='green', width=3)
                    ))
                    
                    fig_cumulative.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    fig_cumulative.update_layout(
                        title="Cumulative ROI Performance Over Time",
                        xaxis_title="Investment Sequence",
                        yaxis_title="Cumulative ROI (%)",
                        height=500,
                        width=800
                    )
                    
                    cumulative_file = save_plotly_as_image(fig_cumulative, "cumulative_roi.png", 800, 500)
                    if cumulative_file:
                        chart_files.append(("Cumulative ROI Analysis", cumulative_file))
            except Exception as e:
                print(f"Error creating cumulative ROI: {e}")
            
            # 3. ROI by Project/Year
            try:
                project_cols = [col for col in df.columns if any(keyword in col.lower() 
                               for keyword in ['project', 'product', 'category', 'segment']) 
                               and df[col].dtype == 'object' and df[col].nunique() < 20]
                
                if project_cols and 'ROI' in df.columns:
                    project_col = project_cols[0]
                    roi_by_project = df.groupby(project_col)['ROI'].mean().reset_index()
                    roi_by_project = roi_by_project.sort_values('ROI', ascending=True)
                    
                    fig_project_roi = px.bar(
                        roi_by_project, 
                        x='ROI', 
                        y=project_col,
                        orientation='h',
                        title=f"Average ROI by {project_col}",
                        color='ROI',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    fig_project_roi.update_layout(height=500, width=800)
                    
                    project_roi_file = save_plotly_as_image(fig_project_roi, "roi_by_project.png", 800, 500)
                    if project_roi_file:
                        chart_files.append(("ROI by Project Analysis", project_roi_file))
            except Exception as e:
                print(f"Error creating ROI by project: {e}")
            
            # 4. ROI vs Risk Analysis
            try:
                if 'ROI' in df.columns:
                    group_cols = [col for col in df.columns if any(keyword in col.lower() 
                                 for keyword in ['project', 'product', 'category', 'segment']) 
                                 and df[col].dtype == 'object' and df[col].nunique() < 20]
                    
                    if group_cols:
                        group_col = group_cols[0]
                        risk_data = df.groupby(group_col)['ROI'].agg(['mean', 'std', 'count']).reset_index()
                        risk_data.columns = [group_col, 'Average_ROI', 'ROI_Risk', 'Count']
                        risk_data['ROI_Risk'] = risk_data['ROI_Risk'].fillna(0)
                        
                        fig_risk = px.scatter(
                            risk_data,
                            x='ROI_Risk',
                            y='Average_ROI',
                            size='Count',
                            color=group_col,
                            title="Risk-Return Profile by Category",
                            labels={
                                'ROI_Risk': 'Risk (ROI Std Dev %)',
                                'Average_ROI': 'Average ROI (%)',
                                'Count': 'Number of Investments'
                            }
                        )
                        
                        fig_risk.update_layout(height=500, width=800)
                        
                        risk_file = save_plotly_as_image(fig_risk, "roi_vs_risk.png", 800, 500)
                        if risk_file:
                            chart_files.append(("ROI vs Risk Analysis", risk_file))
            except Exception as e:
                print(f"Error creating ROI vs risk: {e}")
    
    except Exception as e:
        print(f"General error in chart creation: {e}")
    
    return chart_files

def generate_comprehensive_insights_for_pdf(df, analysis_type, kpis):
    """Generate detailed insights for each chart - DEBUGGED VERSION"""
    
    print(f"DEBUG: Generating insights for analysis type: {analysis_type}")
    print(f"DEBUG: Available KPIs: {list(kpis.keys())}")
    print(f"DEBUG: DataFrame columns: {list(df.columns)}")
    
    insights = {}
    
    try:
        if analysis_type == "💰 Profit & Loss (P/L)":
            # Financial Waterfall Analysis
            if 'Revenue' in df.columns and 'Net_Profit' in df.columns:
                revenue = df['Revenue'].sum()
                net_profit = df['Net_Profit'].sum()
                profit_margin = (net_profit / revenue * 100) if revenue > 0 else 0
                insights['Financial Waterfall Analysis'] = f"The company generates ${revenue:,.0f} in total revenue and achieves ${net_profit:,.0f} in net profit, resulting in a {profit_margin:.1f}% profit margin. This indicates {'strong' if profit_margin > 15 else 'moderate' if profit_margin > 5 else 'weak'} operational efficiency in converting revenue to profit."
                print(f"DEBUG: Waterfall insight created")
            
            # Profitability Trend Analysis
            time_cols = [col for col in ['Year', 'Quarter', 'Month'] if col in df.columns]
            if time_cols and 'Net_Profit' in df.columns:
                time_col = time_cols[0]
                trend_data = df.groupby(time_col)['Net_Profit'].mean()
                if len(trend_data) >= 2:
                    latest_profit = trend_data.iloc[-1]
                    previous_profit = trend_data.iloc[-2]
                    growth_rate = ((latest_profit - previous_profit) / abs(previous_profit) * 100) if previous_profit != 0 else 0
                    trend_direction = "improving" if growth_rate > 5 else "declining" if growth_rate < -5 else "stable"
                    insights['Profitability Trend Analysis'] = f"Profitability trends show a {abs(growth_rate):.1f}% {'increase' if growth_rate > 0 else 'decrease'} in the most recent period, indicating {trend_direction} business performance over time. This trend suggests {'positive momentum' if growth_rate > 0 else 'areas requiring strategic attention' if growth_rate < 0 else 'consistent performance'}."
                    print(f"DEBUG: Trend insight created")
            
            # Break-even Analysis
            if 'Revenue_Per_Unit' in df.columns and 'Cost_Per_Unit' in df.columns:
                avg_revenue_per_unit = df['Revenue_Per_Unit'].mean()
                avg_cost_per_unit = df['Cost_Per_Unit'].mean()
                profit_per_unit = avg_revenue_per_unit - avg_cost_per_unit
                margin_per_unit = (profit_per_unit / avg_revenue_per_unit * 100) if avg_revenue_per_unit > 0 else 0
                insights['Break-Even Analysis'] = f"Unit economics analysis reveals ${avg_revenue_per_unit:.2f} average revenue per unit against ${avg_cost_per_unit:.2f} in costs, generating ${profit_per_unit:.2f} profit per unit ({margin_per_unit:.1f}% margin). This indicates {'healthy' if margin_per_unit > 20 else 'acceptable' if margin_per_unit > 10 else 'concerning'} unit profitability."
                print(f"DEBUG: Break-even insight created")
            
            # Segment Profitability Analysis
            segment_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['product', 'segment', 'category', 'region']) and df[col].dtype == 'object' and df[col].nunique() < 20]
            if segment_cols and 'Net_Profit' in df.columns:
                segment_col = segment_cols[0]
                seg_data = df.groupby(segment_col)['Net_Profit'].sum()
                top_segment = seg_data.idxmax()
                bottom_segment = seg_data.idxmin()
                top_profit = seg_data.max()
                bottom_profit = seg_data.min()
                insights['Segment Profitability Analysis'] = f"Segment analysis shows '{top_segment}' as the top performer with ${top_profit:,.0f} in profits, while '{bottom_segment}' generates ${bottom_profit:,.0f}. This {abs(top_profit - bottom_profit):,.0f} profit gap indicates significant variation in segment performance and opportunities for strategic focus on high-performing areas."
                print(f"DEBUG: Segment insight created")
        
        elif analysis_type == "📈 Return on Investment (ROI)":
            if 'ROI' in df.columns:
                roi_data = df['ROI']
                avg_roi = roi_data.mean()
                median_roi = roi_data.median()
                positive_roi_pct = (roi_data > 0).mean() * 100
                high_roi_count = (roi_data > 25).sum()
                total_investments = len(roi_data)
                
                # ROI Performance Distribution
                insights['ROI Performance Distribution'] = f"Investment portfolio analysis shows an average ROI of {avg_roi:.1f}% (median: {median_roi:.1f}%) across {total_investments} investments. {positive_roi_pct:.0f}% of investments generate positive returns, with {high_roi_count} investments ({high_roi_count/total_investments*100:.1f}%) exceeding 25% ROI. This indicates {'strong' if avg_roi > 20 else 'moderate' if avg_roi > 10 else 'weak'} portfolio performance."
                print(f"DEBUG: ROI distribution insight created")
                
                # ROI vs Risk Analysis  
                roi_std = roi_data.std()
                risk_level = "high" if roi_std > 50 else "moderate" if roi_std > 25 else "low"
                insights['ROI vs Risk Analysis'] = f"Portfolio volatility of {roi_std:.1f}% indicates {risk_level} investment risk. The risk-return profile suggests a {'aggressive growth' if roi_std > 50 and avg_roi > 15 else 'balanced' if roi_std < 30 and avg_roi > 10 else 'conservative'} investment strategy with {'diversified' if roi_std < 40 else 'concentrated'} risk exposure across investments."
                print(f"DEBUG: ROI risk insight created")
                
                # Cumulative ROI Analysis
                cumulative_roi = roi_data.sum()
                performance_level = "exceptional" if cumulative_roi > 300 else "strong" if cumulative_roi > 150 else "moderate" if cumulative_roi > 50 else "underperforming"
                insights['Cumulative ROI Analysis'] = f"Total cumulative ROI of {cumulative_roi:.1f}% demonstrates {performance_level} investment strategy execution. The overall return pattern shows {'accelerating growth' if roi_data.tail(10).mean() > roi_data.head(10).mean() else 'consistent performance' if abs(roi_data.tail(10).mean() - roi_data.head(10).mean()) < 5 else 'declining momentum'} in recent investment decisions."
                print(f"DEBUG: Cumulative insight created")
                
                # ROI by Project Analysis
                project_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['project', 'product', 'category', 'segment']) and df[col].dtype == 'object' and df[col].nunique() < 20]
                if project_cols:
                    project_col = project_cols[0]
                    project_roi = df.groupby(project_col)['ROI'].mean()
                    top_project = project_roi.idxmax()
                    worst_project = project_roi.idxmin()
                    performance_gap = project_roi.max() - project_roi.min()
                    insights['ROI by Project Analysis'] = f"Project performance analysis identifies '{top_project}' as the top performer with {project_roi.max():.1f}% average ROI, while '{worst_project}' shows {project_roi.min():.1f}% ROI. The {performance_gap:.1f}% performance gap indicates {'significant variation' if performance_gap > 30 else 'moderate variation' if performance_gap > 15 else 'consistent performance'} across project categories."
                    print(f"DEBUG: Project insight created")
    
    except Exception as e:
        print(f"ERROR generating insights: {e}")
    
    print(f"DEBUG: Total insights generated: {len(insights)}")
    print(f"DEBUG: Insight keys: {list(insights.keys())}")
    return insights

def get_font_name_for_language(language, font_registered):
    """Get appropriate font name based on language and registration status"""
    if language == "Hindi" and font_registered.get("hindi", False):
        return "NotoSansDevanagari-Hindi"
    elif language == "Marathi" and font_registered.get("marathi", False):
        return "NotoSansDevanagari-Marathi"
    else:
        # Fallback to Helvetica for unsupported languages or failed registration
        return "Helvetica"

def create_multilingual_paragraph_styles(language):
    """Create paragraph styles with proper font support for different languages"""
    
    # Static Font Paths
    font_dir = "fonts"
    regular_font_path = os.path.join(font_dir, "NotoSansDevanagari-Regular.ttf")
    bold_font_path = os.path.join(font_dir, "NotoSansDevanagari-Bold.ttf")
    
    # Check if fonts are available
    devanagari_font_available = os.path.exists(regular_font_path) and os.path.exists(bold_font_path)
    
    # Register fonts if not already registered (avoids repeated registration on rerun)
    if devanagari_font_available and "NotoSansDevanagari-Regular" not in pdfmetrics.getRegisteredFontNames():
        try:
            pdfmetrics.registerFont(TTFont('NotoSansDevanagari-Regular', regular_font_path))
            pdfmetrics.registerFont(TTFont('NotoSansDevanagari-Bold', bold_font_path))
        except Exception as e:
            devanagari_font_available = False
            
    # Select the appropriate font
    if language in ["Hindi", "Marathi"] and devanagari_font_available:
        base_font = 'NotoSansDevanagari-Regular'
        bold_font = 'NotoSansDevanagari-Bold'
    else:
        # Fallback to default for English or if font loading fails
        base_font = 'Helvetica'
        bold_font = 'Helvetica-Bold'
        
    styles = getSampleStyleSheet()
    
    # Define styles
    title_style = ParagraphStyle(
        'MultilingualTitle',
        parent=styles['Title'],
        fontName=bold_font,
        fontSize=20,
        spaceAfter=30,
        alignment=1,
        textColor=colors.darkblue,
    )
    
    heading_style = ParagraphStyle(
        'MultilingualHeading',
        parent=styles['Heading1'],
        fontName=bold_font,
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue,
    )
    
    subheading_style = ParagraphStyle(
        'MultilingualSubheading',
        parent=styles['Heading2'],
        fontName=bold_font,
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.darkgreen,
    )
    
    body_style = ParagraphStyle(
        'MultilingualBody',
        parent=styles['Normal'],
        fontName=base_font,
        fontSize=10,
        spaceAfter=8,
        leading=16,
    )
    
    return title_style, heading_style, subheading_style, body_style, base_font, bold_font

def create_safe_multilingual_paragraph(text, style, language="English"):
    """Create paragraph with enhanced debugging"""
    
    if not text or not isinstance(text, str):
        print(f"DEBUG: Invalid text input: {text}")
        return None
    
    try:
        cleaned_text = text.strip()
        
        if not cleaned_text:
            print(f"DEBUG: Empty text after cleaning")
            return None
        
        # Enhanced character cleaning for problematic Unicode
        if language in ["Hindi", "Marathi"]:
            # Check if text actually contains Devanagari characters
            has_devanagari = any('\u0900' <= char <= '\u097F' for char in cleaned_text)
            print(f"DEBUG: Text has Devanagari characters: {has_devanagari}")
            
            if has_devanagari:
                # Keep Devanagari text as-is
                normalized_text = cleaned_text
            else:
                # Clean Latin text
                normalized_text = clean_problematic_characters(cleaned_text)
        else:
            normalized_text = clean_problematic_characters(cleaned_text)
        
        print(f"DEBUG: Creating paragraph with text: {normalized_text[:50]}...")
        return Paragraph(normalized_text, style)
        
    except Exception as e:
        print(f"ERROR creating paragraph: {str(e)}")
        # Emergency fallback
        try:
            ascii_text = str(text).encode('ascii', 'ignore').decode('ascii')
            if ascii_text.strip():
                print(f"DEBUG: Using ASCII fallback: {ascii_text[:30]}...")
                return Paragraph(ascii_text, style)
            else:
                return Paragraph("Content not displayable", style)
        except:
            print(f"DEBUG: Complete fallback failed")
            return None

def clean_problematic_characters(text):
    """Clean problematic Unicode characters that might cause PDF issues"""
    
    # Character replacements for common problematic Unicode characters
    replacements = {
        '\u2019': "'",  # Right single quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2022': '•',  # Bullet point
        '\u2026': '...',  # Horizontal ellipsis
        '\u00a0': ' ',   # Non-breaking space
        '\u200b': '',    # Zero-width space
        '\u200c': '',    # Zero-width non-joiner
        '\u200d': '',    # Zero-width joiner
    }
    
    cleaned_text = text
    for unicode_char, replacement in replacements.items():
        cleaned_text = cleaned_text.replace(unicode_char, replacement)
    
    return cleaned_text

def show_font_status_info(font_status, language):
    """Show user-friendly font status information"""
    
    if language in ["Hindi", "Marathi"]:
        if font_status["font_available"]:
            st.success(f"✅ {language} font loaded successfully - PDF will display {language} text correctly")
        else:
            st.warning(
                f"⚠️ {language} font could not be loaded. "
                f"PDF content will be generated in {language} by AI but displayed using English fonts. "
                f"The content will still be readable but may not display {language} characters perfectly."
            )
    else:
        st.info(f"ℹ️ Using standard fonts for {language} language")
        
def generate_enhanced_pdf_report(kpis, feature_df, business_context, target_variable, 
                                     language, analysis_type, report_style, df):
    """Generate a professional, well-structured PDF report with FULL multilingual support"""
    
    # Generate AI content in requested language
    with st.spinner(f"Generating content in {language}..."):
        ai_content = generate_ai_content_for_pdf(
            kpis, feature_df, business_context, target_variable, 
            language, analysis_type, report_style
        )
    
    # Create charts
    chart_files = create_all_analysis_charts_for_pdf(df, analysis_type, language)
    chart_insights = generate_comprehensive_insights_for_pdf(df, analysis_type, kpis)
    
    # Create PDF with proper spacing
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        leftMargin=0.8*inch,
        rightMargin=0.8*inch,
        topMargin=1*inch,
        bottomMargin=1*inch
    )
    
    story = []
    
    # Use the new multilingual style system
    title_style, heading_style, subheading_style, body_style, base_font, bold_font = create_multilingual_paragraph_styles(language)
    
    table_body_style = ParagraphStyle('TableBody', parent=body_style, fontName=base_font, alignment=1) # Centered text
    table_header_style = ParagraphStyle('TableHeader', parent=body_style, fontName=bold_font, alignment=1, textColor=colors.white) # Centered, white text for headers
    table_key_style = ParagraphStyle('TableKey', parent=body_style, fontName=bold_font, alignment=0) # Left-aligned for keys 
    table_value_style = ParagraphStyle('TableValue', parent=body_style, fontName=base_font, alignment=1) # Centered for values
       
    # Language notification - CREATE SAFELY with translation
    if language in ["Hindi", "Marathi"]:
        lang_notice_text = get_translation("multilingual_note", language).format(language=language)
        lang_notice = create_safe_multilingual_paragraph(lang_notice_text, body_style, language)
        if lang_notice:
            story.append(lang_notice)
            story.append(Spacer(1, 0.2*inch))
    
    # TITLE PAGE - CREATE SAFELY with translations
    title_text = get_translation("business_intelligence_report", language)
    title_para = create_safe_multilingual_paragraph(title_text, title_style, language)
    if title_para:
        story.append(title_para)
    
    # Get translated report style
    report_style_key = report_style.lower().replace(" ", "_")
    subtitle_text = get_translation(report_style_key, language)
    subtitle_para = create_safe_multilingual_paragraph(subtitle_text, heading_style, language)
    if subtitle_para:
        story.append(subtitle_para)
        
    story.append(Spacer(1, 0.3*inch))
    
    # Report metadata in a clean table - WITH TRANSLATIONS
    metadata_str_data = create_translated_metadata_table(business_context, analysis_type, target_variable, language)
    metadata_para_data = [
    [create_safe_multilingual_paragraph(row[0], table_key_style, language), 
     create_safe_multilingual_paragraph(row[1], table_value_style, language)] 
    for row in metadata_str_data
    ]
    metadata_table = Table(metadata_para_data, colWidths=[2*inch, 3*inch])
    metadata_table.setStyle(TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    
    story.append(metadata_table)
    story.append(PageBreak())
    
    # EXECUTIVE SUMMARY - CREATE SAFELY with translation
    exec_summary_text = get_translation("executive_summary", language)
    exec_summary_para = create_safe_multilingual_paragraph(exec_summary_text, heading_style, language)
    if exec_summary_para:
        story.append(exec_summary_para)
    
    # KPI Summary - WITH TRANSLATIONS
    kpi_header_text = get_translation("key_performance_indicators", language)
    kpi_header_para = create_safe_multilingual_paragraph(kpi_header_text, subheading_style, language)
    if kpi_header_para:
        story.append(kpi_header_para)
    
    # Create KPI table with translations
    kpi_str_data = create_translated_kpi_table(kpis, language)
    kpi_para_data = [[create_safe_multilingual_paragraph(cell, table_header_style, language) for cell in kpi_str_data[0]]] + \
                    [[create_safe_multilingual_paragraph(cell, table_body_style, language) for cell in row] for row in kpi_str_data[1:]]
    kpi_table = Table(kpi_para_data, colWidths=[2*inch, 1.2*inch, 1.3*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.lightblue, colors.white]),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Key Success Factors - WITH TRANSLATIONS
    factors_header_text = get_translation("critical_success_factors", language)
    factors_header_para = create_safe_multilingual_paragraph(factors_header_text, subheading_style, language)
    if factors_header_para:
        story.append(factors_header_para)
    
    # Create factors table with translations
    factor_str_data = create_translated_factor_table(feature_df, language)
    
    # Check if a message is returned instead of table data
    if len(factor_str_data) == 1 and isinstance(factor_str_data[0][0], str) and "no_insight_available" in factor_str_data[0][0]:
        story.append(create_safe_multilingual_paragraph(get_translation("no_insight_available", language), body_style, language))
    else:
        factor_para_data = [[create_safe_multilingual_paragraph(cell, table_header_style, language) for cell in factor_str_data[0]]] + \
                           [[create_safe_multilingual_paragraph(cell, table_body_style, language) for cell in row] for row in factor_str_data[1:]]
        factor_table = Table(factor_para_data, colWidths=[0.5*inch, 2.2*inch, 1*inch, 0.8*inch])
        factor_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.lightgreen, colors.white]),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        story.append(factor_table)
    
    story.append(PageBreak())
    
    # VISUAL ANALYSIS SECTION - WITH TRANSLATIONS
    visual_header_text = get_translation("visual_analysis_dashboard", language)
    visual_header_para = create_safe_multilingual_paragraph(visual_header_text, heading_style, language)
    if visual_header_para:
        story.append(visual_header_para)
    
    # Analysis focus with translation
    translated_analysis_type = translate_analysis_type(analysis_type, language)
    analysis_focus_text = f"{get_translation('analysis_focus', language)}: {translated_analysis_type}"
    analysis_focus_para = create_safe_multilingual_paragraph(analysis_focus_text, subheading_style, language)
    if analysis_focus_para:
        story.append(analysis_focus_para)
    
    # Add all charts with proper spacing and TRANSLATED TITLES
    for i, (chart_title, chart_file) in enumerate(chart_files):
        if os.path.exists(chart_file):
            try:
                # Chart title - TRANSLATE IT
                translated_chart_title = translate_chart_title(chart_title, language)
                chart_title_para = create_safe_multilingual_paragraph(f"{i+1}. {translated_chart_title}", subheading_style, language)
                if chart_title_para:
                    story.append(chart_title_para)
                
                # Chart image
                img = Image(chart_file, width=6.5*inch, height=4*inch)
                story.append(img)
                
                # Chart insight - TRANSLATE KEY INSIGHT LABEL
                insight_text = get_translation("no_insight_available", language)
                
                # Use a more robust check to match insight keys
                matched_insight = None
                for insight_key, insight in chart_insights.items():
                    if insight_key.lower().replace(" ", "_") in chart_title.lower().replace(" ", "_"):
                        matched_insight = insight
                        break
                
                if matched_insight:
                    insight_text = matched_insight
                
                key_insight_label = get_translation("key_insight", language)
                full_insight_text = f"<b>{key_insight_label}:</b> {insight_text}"
                insight_para = create_safe_multilingual_paragraph(full_insight_text, 'body_style')
                if insight_para:
                    story.append(insight_para)
                    
                story.append(Spacer(1, 0.3*inch))
                
                # Page break after every 2 charts
                if (i + 1) % 2 == 0 and i < len(chart_files) - 1:
                    story.append(PageBreak())
                    
            except Exception as e:
                print(f"Error adding chart {chart_title}: {e}")
                continue
    
    # AI RECOMMENDATIONS SECTION - WITH TRANSLATIONS
    story.append(PageBreak())
    
    ai_header_text = get_translation("ai_powered_recommendations", language)
    ai_header_para = create_safe_multilingual_paragraph(ai_header_text, heading_style, language)
    if ai_header_para:
        story.append(ai_header_para)
    
    if language != "English":
        lang_note_text = get_translation("generated_in_ai", language).format(language=language)
        lang_note_para = create_safe_multilingual_paragraph(f"({lang_note_text})", body_style, language)
        if lang_note_para:
            story.append(lang_note_para)
            story.append(Spacer(1, 0.1*inch))
    
    # Process AI content with better formatting - CREATE ALL PARAGRAPHS SAFELY
    if ai_content:
        # Split content into meaningful sections
        sections = ai_content.split('\n\n')
        recommendation_count = 1
        
        for section in sections:
            section = section.strip()
            if len(section) > 30:  # Skip very short sections
                # Clean formatting
                clean_section = section.replace('**', '').replace('*', '')
                clean_section = clean_section.replace('###', '').replace('##', '').replace('#', '')
                clean_section = clean_section.strip()
                
                if clean_section:
                    # Check if it's a header or recommendation - TRANSLATE RECOMMENDATION LABEL
                    if any(word in clean_section.lower() for word in ['recommendation', 'insight', 'strategy', 'action']):
                        recommendation_text = get_translation("recommendation", language)
                        rec_para = create_safe_multilingual_paragraph(f"{recommendation_text} {recommendation_count}:", subheading_style, language)
                        if rec_para:
                            story.append(rec_para)
                        recommendation_count += 1
                    
                    content_para = create_safe_multilingual_paragraph(clean_section, body_style, language)
                    if content_para:
                        story.append(content_para)
                        story.append(Spacer(1, 0.1*inch))
    
    # FINANCIAL IMPACT PROJECTION - WITH TRANSLATIONS
    story.append(PageBreak())
    story.append(Spacer(1, 0.2*inch))
    
    financial_header_text = get_translation("expected_financial_impact", language)
    financial_header_para = create_safe_multilingual_paragraph(financial_header_text, subheading_style, language)
    if financial_header_para:
        story.append(financial_header_para)
    
    # Calculate realistic projections
    if 'Total_Revenue' in kpis and 'Total_Profit' in kpis:
        
        try:
            current_revenue = float(kpis['Total_Revenue'])
            current_profit = float(kpis['Total_Profit'])
        except (ValueError, TypeError):
            st.error("❌ Invalid KPI values detected. Cannot perform calculations.")
            return None
        
        current_margin = (current_profit / current_revenue * 100) if current_revenue > 0 else 0

        projected_margin = current_margin * 1.15
        projected_revenue = kpis['Total_Revenue'] * 1.10
        projected_profit = kpis['Total_Profit'] * 1.25
        
        # Create impact table with translations
        impact_str_data = [
        [
            get_translation("metric", language), 
            get_translation("current", language), 
            get_translation("projected", language), 
            get_translation("improvement", language)
        ],
        [
            get_translation("profitmargin", language), 
            f"{current_margin:.1f}%", 
            f"{projected_margin:.1f}%", 
            "15%"
        ],
        [
            get_translation("revenueimpact", language), 
            format_currency_value(current_revenue),  # Use corrected value
            format_currency_value(projected_revenue),    # Use corrected value
            "10%"
        ],
        [
            get_translation("additionalprofit", language), 
            format_currency_value(current_profit),  # Use corrected value
            format_currency_value(projected_profit),     # Use corrected value
            "25%"
        ]
        ]
        impact_para_data = [ [Paragraph(cell, table_header_style) for cell in impact_str_data[0]] ] + \
                       [ [Paragraph(cell, table_body_style) for cell in row] for row in impact_str_data[1:] ]
        impact_table = Table(impact_para_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1*inch])
        impact_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.navy),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.lightgrey, colors.white]),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        story.append(impact_table)
    else:
        # Handle case where KPIs are not available
        no_impact_text = get_translation("no_specific_insight_available", language)
        story.append(create_safe_multilingual_paragraph(no_impact_text, body_style, language))
        
    def add_whatif_to_pdf_report(story, language, saved_scenarios=None):
        """Add What-If analysis section to PDF report"""
        
        if saved_scenarios is None:
            saved_scenarios = st.session_state.get("whatif_scenarios", {})
        if not saved_scenarios:
            return
  
    # Create styles
    title_style, heading_style, subheading_style, body_style, base_font, bold_font = create_multilingual_paragraph_styles(language)
    
    story.append(PageBreak())
    
    # What-If section header
    whatif_header = get_translation("whatif_analysis", language, fallback="What-If Scenario Analysis")
    whatif_para = create_safe_multilingual_paragraph(whatif_header, heading_style, language)
    if whatif_para:
        story.append(whatif_para)
    
    # Summary table
    whatif_data = [[
        "Scenario Name", "Impact", "Impact %", "Predicted Value"
    ]]
    
    saved_scenarios = st.session_state.get("whatif_scenarios", {})
    for name, scenario in saved_scenarios.items():
        pred = scenario['prediction']
        whatif_data.append([
            name,
            f"{pred['impact']:+,.2f}",
            f"{pred['impact_percentage']:+.1f}%",
            f"{pred['predicted_value']:,.2f}"
        ])
  
    whatif_table = Table(whatif_data, colWidths=[2*inch, 1*inch, 1*inch, 1.5*inch])
    whatif_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.lightblue, colors.white]),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    
    story.append(whatif_table)
    
    # Build PDF
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as build_error:
        st.error(f"Error building PDF: {str(build_error)}")
        return None
    finally:
        # Cleanup
        for _, chart_file in chart_files:
            try:
                if os.path.exists(chart_file):
                    os.unlink(chart_file)
            except:
                pass
    
def handle_export_requests(export_formats, language, report_style, df=None, kpis=None, feature_df=None, 
                          business_context=None, target_variable=None, analysis_type=None):
    """Enhanced export handler with comprehensive PDF generation"""
    st.subheader("📤 Export Your Analysis")
    
    if "📋 PDF Report" in export_formats:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"📊 Report Type: {report_style} | Language: {language}")
            
            # Show what will be included in the enhanced report
            inclusions = {
                "Executive Summary": [
                    "Key Performance Indicators", 
                    "Top Success Factors", 
                    f"All {analysis_type} Visualizations",
                    "Chart-specific Insights",
                    "Strategic Recommendations", 
                    "Expected Financial Impact"
                ],
                "Detailed Analysis": [
                    "Comprehensive KPI Analysis",
                    "Detailed Factor Analysis", 
                    f"Complete {analysis_type} Chart Suite",
                    "In-depth Chart Insights",
                    "Risk Assessment",
                    "Implementation Roadmap",
                    "ROI Projections"
                ],
                "Technical Report": [
                    "Model Performance Metrics",
                    "Feature Importance Analysis",
                    f"All {analysis_type} Technical Charts", 
                    "Statistical Insights",
                    "Data Quality Assessment",
                    "Technical Recommendations"
                ]
            }
            
            st.write("**Enhanced Report will include:**")
            for item in inclusions[report_style]:
                st.write(f"✅ {item}")
                
            # Show specific charts that will be included
            chart_list = []
            if analysis_type == "💰 Profit & Loss (P/L)":
                chart_list = ["Financial Waterfall Chart", "Profitability Trend Analysis", 
                             "Break-Even Analysis", "Segment Profitability Chart"]
            elif analysis_type == "📈 Return on Investment (ROI)":
                chart_list = ["ROI Distribution Funnel", "Cumulative ROI Performance", 
                             "ROI by Project/Category", "Risk-Return Scatter Plot"]
            
            if chart_list:
                st.write(f"**{analysis_type} Charts included:**")
                for chart in chart_list:
                    st.write(f"📈 {chart}")
        
        with col2:
            if st.button("📋 Generate Enhanced PDF", type="primary", use_container_width=True):
                if all([df is not None, kpis, feature_df is not None, business_context, target_variable, analysis_type]):
                    try:
                        # Use the enhanced PDF generation function
                        pdf_bytes = generate_enhanced_pdf_report(
                            kpis=kpis,
                            feature_df=feature_df, 
                            business_context=business_context,
                            target_variable=target_variable,
                            language=language,
                            analysis_type=analysis_type,
                            report_style=report_style,
                            df=df
                        )
                        
                        if pdf_bytes:
                            # Success message
                            st.success(f"✅ Enhanced {report_style} generated with all {analysis_type} charts!")
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Complete PDF Report",
                                data=pdf_bytes,
                                file_name=f"enhanced_business_analysis_{report_style.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        else:
                            st.error("❌ Failed to generate PDF. Please try again.")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating enhanced PDF: {str(e)}")
                        st.error("Please check your Gemini API configuration and data completeness.")
                else:
                    st.error("❌ Missing required data. Please run the analysis first.")
            
            if st.button("👀 Preview Report Content", use_container_width=True):
                if all([df is not None, kpis, analysis_type]):
                    st.info("**Report Preview:**")
                    st.write(f"- Business Context: {business_context}")
                    st.write(f"- Analysis Type: {analysis_type}")
                    st.write(f"- Target Variable: {target_variable}")
                    st.write(f"- Number of KPIs: {len(kpis)}")
                    st.write(f"- Number of Key Factors: {len(feature_df) if feature_df is not None else 0}")
                    
                    # Count available charts
                    chart_count = 0
                    if analysis_type == "💰 Profit & Loss (P/L)":
                        if 'Revenue' in df.columns: chart_count += 1
                        if any(col in df.columns for col in ['Year', 'Month', 'Quarter']): chart_count += 1
                        if any('units' in col.lower() for col in df.columns): chart_count += 1
                        if any('segment' in col.lower() or 'product' in col.lower() for col in df.columns): chart_count += 1
                    elif analysis_type == "📈 Return on Investment (ROI)":
                        if 'ROI' in df.columns: chart_count += 4  # All ROI charts available
                    
                    st.write(f"- Available Charts: {chart_count}")
                    st.success("Report ready for generation!")
    
# ===========================
# DATA PREPARATION FUNCTIONS
# ===========================
def prepare_data(df, target_name, use_polynomial=False, use_scaling=True):
    """Prepare features for modeling dynamically (generalized)."""
    feature_cols = [col for col in df.columns if col != target_name]

    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical_cols:
        df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    else:
        df_encoded = pd.DataFrame(index=df.index)

    X = pd.concat([df[numeric_cols], df_encoded], axis=1)

    pipeline_steps = []
    if use_scaling:
        pipeline_steps.append(("scaler", StandardScaler()))
    if use_polynomial:
        pipeline_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    return X, pipeline_steps

def get_model_with_params(model_name):
    if model_name == "Random Forest":
        return RandomForestRegressor(random_state=42, n_jobs=-1), {}
    elif model_name == "XGBoost":
        return XGBRegressor(random_state=42, n_jobs=-1, verbosity=0), {}
    else:
        return LinearRegression(), {}

def train_evaluate_model_user_data(X, y, model_name, test_size_pct, scoring_metric, pipeline_steps, log_transform=False):
    """Train and evaluate model on user's dataset"""
    y_transformed = y.copy()
    if log_transform and (y > 0).all():
        y_transformed = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transformed, test_size=test_size_pct, random_state=42
    )

    # Use pretrained model if available
    if st.session_state.model_training_complete and model_name in st.session_state.best_models:
        st.info(f"🔄 Using pre-trained {model_name} model and fine-tuning on your data...")
        
        # Get the pretrained model
        pretrained_model = st.session_state.best_models[model_name]
        
        # Create a new model with same architecture but retrain on new data
        base_model, _ = get_model_with_params(model_name)
        pipe = Pipeline(pipeline_steps + [("model", base_model)])
        
        # If the pretrained model has hyperparameters, use them
        if model_name in st.session_state.model_results:
            best_params = st.session_state.model_results[model_name]['best_params']
            if best_params:
                # Apply best parameters to new model
                for param_name, param_value in best_params.items():
                    if param_name.startswith('model__'):
                        param_key = param_name.replace('model__', '')
                        if hasattr(base_model, param_key):
                            setattr(base_model, param_key, param_value)
    else:
        base_model, _ = get_model_with_params(model_name)
        pipe = Pipeline(pipeline_steps + [("model", base_model)])

    with st.spinner(f"Training {model_name} on your dataset..."):
        pipe.fit(X_train, y_train)

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    if log_transform and (y > 0).all():
        y_train_pred = np.expm1(y_train_pred)
        y_test_pred = np.expm1(y_test_pred)
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
    else:
        y_train_actual = y_train
        y_test_actual = y_test

    metrics = {
        "train_r2": r2_score(y_train_actual, y_train_pred),
        "test_r2": r2_score(y_test_actual, y_test_pred),
        "train_mae": mean_absolute_error(y_train_actual, y_train_pred),
        "test_mae": mean_absolute_error(y_test_actual, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
    }
    metrics["overfitting_score"] = metrics["train_r2"] - metrics["test_r2"]

    return pipe, metrics, X_train, X_test, y_test_actual, y_test_pred

# ===========================
# USER DATA UPLOAD SECTION
# ===========================
st.header("📂 Upload Your Dataset")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data
    raw_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    # Analyze features
    num_cols, cat_cols = analyze_dataset_features(raw_data)

    # Clean dataset
    data_cleaned, cat_cols_updated, num_cols_updated = auto_clean_dataset(raw_data)
    
    # Engineer features
    data_engineered, new_features = engineer_features_with_metrics(data_cleaned, num_cols_updated, cat_cols_updated)
    
    # Store in session state
    st.session_state["data_cleaned"] = data_cleaned
    st.session_state["cat_cols"] = cat_cols_updated
    st.session_state["num_cols"] = num_cols_updated
    st.session_state["data_engineered"] = data_engineered

# ===========================
# USER-FRIENDLY SIDEBAR
# ===========================
with st.sidebar:
    st.header("🎯 Analysis Settings")
    
    # 1. BUSINESS CONTEXT SELECTION
    st.subheader("🏢 Business Context")
    business_type = st.selectbox(
        "What type of business analysis?",
        [
            "General Business Analysis",
            "Sales & Revenue Analysis", 
            "Marketing ROI Analysis",
            "Financial Performance",
            "Investment Analysis",
            "Project Performance",
            "Cost Analysis",
            "Custom Analysis"
        ]
    )
    
    if st.session_state.get("data_engineered") is not None:
        data_eng = st.session_state["data_engineered"]
        numeric_targets = data_eng.select_dtypes(include=["number"]).columns.tolist()
        
        # Auto-suggest targets based on business type
        suggested_targets = get_suggested_targets(business_type, numeric_targets)
        
        if len(suggested_targets) > 1:
            target_name = st.selectbox(
                "What do you want to predict?", 
                suggested_targets,
                help="Choose the main metric you want to analyze and predict"
            )
        else:
            target_name = suggested_targets[0] if suggested_targets else numeric_targets[0]
            st.info(f"🎯 Analyzing: {target_name}")
    
    st.markdown("---")
    
    #2. FOCUS AREAS
    st.subheader("📊 Analysis Type")
    analysis_type = st.radio(
        "Select primary analysis focus:",
        ["💰 Profit & Loss (P/L)", "📈 Return on Investment (ROI)"],
        help="Choose whether to focus on P/L metrics or ROI performance"    
    )
    
    st.markdown("---")
    
    #3. ALERT THRESHOLDS
    st.subheader("⚠️ Performance Alerts")
    
    enable_alerts = st.checkbox("🚨 Enable performance alerts")
    
    if enable_alerts:
        col1, col2 = st.columns(2)
        with col1:
            profit_threshold = st.number_input("Profit Alert (below %)", value=-5.0, step=1.0)
        with col2:
            roi_threshold = st.number_input("ROI Alert (below %)", value=0.0, step=1.0)
    
    st.markdown("---")
    
    # 4. EXPORT & SHARING
    st.subheader("📄 Export Options")

    # Language for reports
    language_choice = st.selectbox(
        "Report Language:",
        ["English", "Marathi", "Hindi"],
        help="Language for PDF reports and AI recommendations"
    )

    # Store in session state for use across functions
    st.session_state["selected_language"] = language_choice

    # Only show report style since PDF is the only option
    report_style = st.radio(
        "Report style:",
        ["Executive Summary", "Detailed Analysis", "Technical Report"]
    )

    st.markdown("---")
    
# ===========================
# MAIN WORKFLOW & PREDICTIONS
# ===========================
# STEP 1: COMPUTATION BLOCK (Modified)
if st.button("🚀 Generate Business Intelligence & Predictions", type="primary", use_container_width=True):
        # Hide old results
        st.session_state.analysis_run_successfully = False
        
        with st.spinner("Analyzing data, training models, and generating insights..."):
            data_eng = st.session_state["data_engineered"]
            
            # Calculate KPIs
            kpis = calculate_kpis(data_eng)
            st.session_state["kpi_data"] = kpis
            
            # Generate alerts if enabled
            if 'enable_alerts' in locals() and enable_alerts:
                alerts = create_performance_alerts(data_eng, profit_threshold, roi_threshold)
                st.session_state["performance_alerts"] = alerts
            
            # Model training
            X, pipeline_steps = prepare_data(data_eng, target_name)
            y = data_eng[target_name]
            model_name_to_train = "XGBoost"
            
            if not st.session_state.get('model_training_complete'):
                best_models, model_results = train_models_with_hyperparameter_tuning(X, y)
                st.session_state.best_models = best_models
                st.session_state.model_results = model_results
                st.session_state.model_training_complete = True
            
            if model_name_to_train in st.session_state.best_models:
                final_model = st.session_state.best_models[model_name_to_train]
                model_metrics = st.session_state.model_results[model_name_to_train]
                
                st.session_state.final_model = final_model
                st.session_state.model_metrics = model_metrics
                st.session_state.X_data = X
                
                chart_insights = generate_comprehensive_insights_for_pdf(data_eng, analysis_type, kpis)
                st.session_state.chart_insights = chart_insights
                
                st.session_state.analysis_run_successfully = True
            else:
                st.error(f"Model '{model_name_to_train}' not found after training.")

# STEP 2: DISPLAY BLOCK (Reordered)
if st.session_state.get("analysis_run_successfully"):            
    if st.session_state.get("performance_alerts"):
        display_alerts(st.session_state["performance_alerts"])

    create_kpi_cards(st.session_state["kpi_data"])
    st.markdown("---")

    create_focused_analytics_by_type(st.session_state["data_engineered"], analysis_type)
    st.markdown("---")
    
    feature_df = create_business_feature_insights(
        model=st.session_state.final_model, 
        X=st.session_state.X_data, 
        target_name=target_name,
        business_type=business_type, 
        language_choice=language_choice,
        analysis_type=analysis_type
    )
    
    if feature_df is not None:
        st.session_state.feature_df = feature_df
    
    if st.session_state.get("analysis_run_successfully"):
        st.markdown("---")
    
    # Add What-If Analysis section
    whatif_expander = st.expander("🔮 What-If Scenario Analysis", expanded=False)
    with whatif_expander:
        create_whatif_interface()
    
    # 6. Display PDF Export Section LAST
    handle_export_requests(
        export_formats=["📋 PDF Report"], 
        language=st.session_state.get("selected_language", "English"), 
        report_style=report_style,
        df=st.session_state["data_engineered"],
        kpis=st.session_state["kpi_data"],
        feature_df=st.session_state.get("feature_df"),
        business_context=business_type,
        target_variable=target_name,
        analysis_type=analysis_type
    )
    
def create_business_feature_insights(model, X, target_name, business_type, language_choice, analysis_type=None):
    """Business-friendly feature importance (WITHOUT automatic AI recommendations)."""

    try:
        # Determine feature importances from the model pipeline
        if hasattr(model.named_steps["model"], "feature_importances_"):
            importances = model.named_steps["model"].feature_importances_
        elif hasattr(model.named_steps["model"], "coef_"):
            importances = np.abs(model.named_steps["model"].coef_)
        else:
            st.info("Feature importance analysis not available for this model type.")
            return None

        # Create and display the feature importance dataframe and chart
        fi_df = pd.DataFrame({
            "Factor": [name.replace("_", " ").title() for name in X.columns[:len(importances)]],
            "Impact_Score": importances
        }).sort_values("Impact_Score", ascending=False).head(10)

        # Store feature_df in session state for PDF generation
        st.session_state["feature_df"] = fi_df

        # NOW call the AI recommendations function with chart insights
        if st.session_state.get("kpi_data"):
            current_analysis_type = st.session_state.get("current_analysis_type")
            chart_insights = st.session_state.get("chart_insights", {})
            
            generate_ai_recommendations(
                kpis=st.session_state["kpi_data"],
                feature_df=fi_df,
                business_context=business_type,
                target_variable=target_name,
                language=language_choice,
                analysis_type=current_analysis_type,
                chart_insights=chart_insights
            )
        
        return fi_df

    except Exception as e:
        st.error(f"Could not analyze key factors: {str(e)}")
        return None
