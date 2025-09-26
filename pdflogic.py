
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
    """Save Plotly figure as image for PDF inclusion"""
    try:
        # Save as static image
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.write(img_bytes)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        st.error(f"Error saving chart: {e}")
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
        
        # 2. Profitability Analysis (if available)
        if 'Net_Profit' in df.columns:
            # Profit distribution
            profit_data = df['Net_Profit']
            positive_profit = (profit_data > 0).sum()
            negative_profit = (profit_data <= 0).sum()
            
            fig_profit = go.Figure(data=[
                go.Pie(labels=['Profitable', 'Loss/Break-even'], 
                      values=[positive_profit, negative_profit],
                      colors=['#2ECC71', '#E74C3C'])
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

def generate_comprehensive_pdf_report(kpis, feature_df, business_context, target_variable, 
                                    language, analysis_type, report_style, df):
    """Generate comprehensive PDF report with AI content and visualizations"""
    
    # Generate AI content
    with st.spinner(f"Generating {report_style} content in {language}..."):
        ai_content = generate_ai_content_for_pdf(
            kpis, feature_df, business_context, target_variable, 
            language, analysis_type, report_style
        )
    
    # Create charts for PDF
    with st.spinner("Creating visualizations for PDF..."):
        chart_files = create_summary_charts_for_pdf(df, kpis)
    
    # Generate PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Title Page
    story.append(Paragraph(f"Business Intelligence Report", title_style))
    story.append(Paragraph(f"({report_style})", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    
    # Report metadata
    metadata = [
        f"Business Context: {business_context}",
        f"Analysis Type: {analysis_type}",
        f"Target Variable: {target_variable.replace('_', ' ')}",
        f"Report Language: {language}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    
    for item in metadata:
        story.append(Paragraph(item, styles['Normal']))
        story.append(Spacer(1, 6))
    
    story.append(PageBreak())
    
    # KPI Summary Table
    story.append(Paragraph("Key Performance Indicators", heading_style))
    
    kpi_data = [[key.replace('_', ' ').title(), f"{value:,.2f}"] 
               for key, value in kpis.items()]
    
    kpi_table = Table(kpi_data, colWidths=[3*inch, 2*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 14),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))
    
    story.append(kpi_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Top Factors Table
    story.append(Paragraph("Key Success Factors", heading_style))
    
    factor_data = [["Factor", "Impact Score"]]
    for index, row in feature_df.head(5).iterrows():
        factor_data.append([row['Factor'], f"{row['Impact_Score']:.3f}"])
    
    factor_table = Table(factor_data, colWidths=[3*inch, 2*inch])
    factor_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND',(0,1),(-1,-1),colors.lightblue),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))
    
    story.append(factor_table)
    story.append(PageBreak())
    
    # Add Charts
    for chart_title, chart_file in chart_files:
        if os.path.exists(chart_file):
            story.append(Paragraph(chart_title, heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Add image to PDF
            img = Image(chart_file, width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
            story.append(PageBreak())
    
    # AI Generated Content
    story.append(Paragraph("Detailed Analysis", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Split AI content into paragraphs
    content_paragraphs = ai_content.split('\n')
    for paragraph in content_paragraphs:
        if paragraph.strip():
            # Handle bold text formatting
            if paragraph.startswith('**') and paragraph.endswith('**'):
                story.append(Paragraph(paragraph.replace('**', ''), heading_style))
            else:
                story.append(Paragraph(paragraph, styles['Normal']))
            story.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    # Cleanup temporary files
    for _, chart_file in chart_files:
        try:
            if os.path.exists(chart_file):
                os.unlink(chart_file)
        except:
            pass
    
    return buffer.getvalue()

def handle_export_requests(export_formats, language, report_style, df=None, kpis=None, feature_df=None, 
                          business_context=None, target_variable=None, analysis_type=None):
    """Enhanced export handler with actual PDF generation"""
    st.subheader("📤 Export Your Analysis")
    
    if "📋 PDF Report" in export_formats:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"📊 Report Type: {report_style} | Language: {language}")
            
            # Show what will be included
            inclusions = {
                "Executive Summary": ["Key findings", "Strategic recommendations", "Financial impact", "Charts"],
                "Detailed Analysis": ["Performance analysis", "Factor analysis", "Risk assessment", "Implementation roadmap", "Charts"],
                "Technical Report": ["Model performance", "Feature importance", "Statistical insights", "Technical recommendations", "Charts"]
            }
            
            st.write("**Report will include:**")
            for item in inclusions[report_style]:
                st.write(f"✅ {item}")
        
        with col2:
            if st.button("📋 Generate PDF Report", type="primary", use_container_width=True):
                if all([df is not None, kpis, feature_df is not None, business_context, target_variable, analysis_type]):
                    try:
                        # Generate PDF
                        pdf_bytes = generate_comprehensive_pdf_report(
                            kpis=kpis,
                            feature_df=feature_df,
                            business_context=business_context,
                            target_variable=target_variable,
                            language=language,
                            analysis_type=analysis_type,
                            report_style=report_style,
                            df=df
                        )
                        
                        # Success message
                        st.success(f"✅ {report_style} generated successfully in {language}!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"business_analysis_{report_style.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
                        st.error("Please check your Gemini API configuration and try again.")
                else:
                    st.error("❌ Missing required data. Please run the analysis first.")

# Update the main workflow call
def update_main_workflow_pdf_export():
    """Updated function call for the main workflow"""
    # In your main workflow, replace the existing handle_export_requests call with:
    
    # Export options (place this after all analysis is complete)
    if st.session_state.get("kpi_data") and st.session_state.get("data_engineered") is not None:
        handle_export_requests(
            export_formats=["📋 PDF Report"], 
            language=st.session_state.get("selected_language", "English"), 
            report_style=report_style,  # From sidebar
            df=st.session_state["data_engineered"],
            kpis=st.session_state["kpi_data"],
            feature_df=feature_df,  # From your feature importance analysis
            business_context=business_type,  # From sidebar
            target_variable=target_name,  # From analysis
            analysis_type=analysis_type  # From sidebar
        )
        
# INTEGRATION UPDATES FOR YOUR STREAMLIT APP

# 1. UPDATE THE create_business_feature_insights FUNCTION
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

        fig = px.bar(
            fi_df, x="Impact_Score", y="Factor", orientation="h",
            title=f"🎯 Top Factors Influencing {target_name.replace('_', ' ')}"
        )
        st.plotly_chart(fig, use_container_width=True)

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

# 2. UPDATE THE MAIN WORKFLOW SECTION (Replace the existing ML Model section)
def updated_main_workflow_section():
    """Updated main workflow section with PDF integration"""
    
    # [Previous sections remain the same until ML Model section]
    
    # ML Model section (simplified for users)
    feature_df = None  # Initialize feature_df
    
    if target_name:
        st.header("🤖 Predictive Intelligence")
        
        with st.spinner("Building predictive model..."):
            X, pipeline_steps = prepare_data(data_eng, target_name, use_polynomial=False, use_scaling=True)
            y = data_eng[target_name]
            
            # Auto-select best model from pre-trained options
            if st.session_state.model_training_complete:
                best_model_name = max(st.session_state.model_results.keys(), 
                                    key=lambda x: st.session_state.model_results[x]['test_r2'])
                model_name = best_model_name
            else:
                model_name = "Random Forest"  # Default fallback
            
            model, metrics, X_train, X_test, y_test_actual, y_test_pred = train_evaluate_model_user_data(
                X, y, model_name, 0.2, "r2", pipeline_steps, log_transform=False
            )
            
            # Store model metrics in session state for PDF
            st.session_state["model_metrics"] = metrics
            st.session_state["model_name"] = model_name
            
            # Simplified metrics display for business users
            create_business_model_summary(metrics, model_name, target_name)
            
            # Feature importance in business terms - capture the returned dataframe
            feature_df = create_business_feature_insights(model, X, target_name, business_type, language_choice)

    # Store all necessary data for PDF generation
    st.session_state["target_name"] = target_name if 'target_name' in locals() else None
    st.session_state["business_type"] = business_type
    st.session_state["analysis_type"] = analysis_type
    
    # Export options - Updated call
    if st.session_state.get("kpi_data") and st.session_state.get("data_engineered") is not None:
        handle_export_requests(
            export_formats=["📋 PDF Report"], 
            language=language_choice, 
            report_style=report_style,
            df=st.session_state["data_engineered"],
            kpis=st.session_state["kpi_data"],
            feature_df=feature_df or st.session_state.get("feature_df"),  # Use stored or current
            business_context=business_type,
            target_variable=target_name if 'target_name' in locals() else "Unknown",
            analysis_type=analysis_type
        )
    
    st.success("✅ Business intelligence analysis completed!")

# 3. REPLACE THE EXISTING handle_export_requests FUNCTION WITH THE NEW ONE FROM ABOVE

# 4. ADD THE NEW PDF GENERATION FUNCTIONS TO YOUR IMPORTS
# Add these imports at the top of your file:
"""
from reportlab.platypus import Image
import tempfile
import os
from PIL import Image as PILImage
"""

# 5. UPDATE THE CHART INSIGHT GENERATION
def generate_chart_insights_for_pdf(df):
    """Enhanced chart insights for PDF inclusion"""
    insights = {}

    # Revenue and Profit Analysis
    if 'Revenue' in df.columns and 'Net_Profit' in df.columns:
        total_rev = df['Revenue'].sum()
        total_profit = df['Net_Profit'].sum()
        profit_margin = (total_profit / total_rev * 100) if total_rev > 0 else 0
        insights["Financial Performance"] = f"Total Revenue: ${total_rev:,.0f}, Net Profit: ${total_profit:,.0f}, Margin: {profit_margin:.1f}%"

    # ROI Analysis
    if 'ROI' in df.columns:
        avg_roi = df['ROI'].mean()
        positive_roi_count = (df['ROI'] > 0).sum()
        total_investments = len(df['ROI'].dropna())
        success_rate = (positive_roi_count / total_investments * 100) if total_investments > 0 else 0
        insights["Investment Performance"] = f"Average ROI: {avg_roi:.1f}%, Success Rate: {success_rate:.1f}%"

    # Profitability Distribution
    if 'Net_Profit' in df.columns:
        profitable_count = (df['Net_Profit'] > 0).sum()
        total_count = len(df['Net_Profit'].dropna())
        profitability_rate = (profitable_count / total_count * 100) if total_count > 0 else 0
        insights["Profitability Distribution"] = f"Profitable entries: {profitable_count}/{total_count} ({profitability_rate:.1f}%)"

    # Segment Performance
    segment_cols = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['segment', 'category', 'region', 'product']) 
                   and df[col].dtype == 'object' and df[col].nunique() < 10]
    
    if segment_cols and 'Net_Profit' in df.columns:
        segment_col = segment_cols[0]
        seg_performance = df.groupby(segment_col)['Net_Profit'].sum()
        best_segment = seg_performance.idxmax()
        worst_segment = seg_performance.idxmin()
        insights["Segment Analysis"] = f"Best performing {segment_col.lower()}: {best_segment}, Lowest: {worst_segment}"

    # Store in session state for AI processing
    st.session_state["chart_insights"] = insights
    return insights

# 6. ADD THIS CALL IN YOUR MAIN WORKFLOW AFTER DATA ENGINEERING
def add_chart_insights_generation():
    """Add this after your data engineering section"""
    if st.session_state.get("data_engineered") is not None:
        generate_chart_insights_for_pdf(st.session_state["data_engineered"])

# 7. USAGE EXAMPLE - REPLACE YOUR EXISTING MAIN BUTTON SECTION WITH:
def complete_integrated_workflow():
    """Complete integrated workflow with PDF generation"""
    
    if st.session_state.get("data_engineered") is not None:
        if st.button("🚀 Generate Business Intelligence & Predictions", type="primary", use_container_width=True):
            data_eng = st.session_state["data_engineered"]
            
            # Generate chart insights for PDF
            generate_chart_insights_for_pdf(data_eng)
            
            # Display performance alerts if enabled
            if 'enable_alerts' in locals() and enable_alerts:
                alerts = create_performance_alerts(data_eng, profit_threshold, roi_threshold)
                display_alerts(alerts)
            
            # Calculate and display KPIs
            kpis = calculate_kpis(data_eng)
            st.session_state["kpi_data"] = kpis
            create_kpi_cards(kpis)
            
            st.markdown("---")
            
            # Create visualizations based on selected analysis type
            create_focused_analytics_by_type(data_eng, analysis_type)

            # Add miscellaneous charts section
            st.markdown("---")
            st.header("📊 Additional Business Insights")

            # Create a 2x1 grid for miscellaneous charts
            col1, col2 = st.columns(2)

            with col1:
                # Auto-detect business issues
                detect_business_issues()
        
                # Show seasonal patterns if available
                create_seasonal_analysis(data_eng)

            with col2:
                # Show top performers
                create_top_performers_analysis(data_eng)
        
                # Show growth trends
                create_growth_analysis(data_eng)
                    
            st.markdown("---")
            
            # ML Model section
            feature_df = None
            
            if target_name:
                st.header("🤖 Predictive Intelligence")
                
                with st.spinner("Building predictive model..."):
                    X, pipeline_steps = prepare_data(data_eng, target_name, use_polynomial=False, use_scaling=True)
                    y = data_eng[target_name]
                    
                    # Auto-select best model from pre-trained options
                    if st.session_state.model_training_complete:
                        best_model_name = max(st.session_state.model_results.keys(), 
                                            key=lambda x: st.session_state.model_results[x]['test_r2'])
                        model_name = best_model_name
                    else:
                        model_name = "Random Forest"  # Default fallback
                    
                    model, metrics, X_train, X_test, y_test_actual, y_test_pred = train_evaluate_model_user_data(
                        X, y, model_name, 0.2, "r2", pipeline_steps, log_transform=False
                    )
                    
                    # Store model metrics in session state for PDF
                    st.session_state["model_metrics"] = metrics
                    st.session_state["model_name"] = model_name
                    
                    # Simplified metrics display for business users
                    create_business_model_summary(metrics, model_name, target_name)
                    
                    # Feature importance in business terms - capture the returned dataframe
                    feature_df = create_business_feature_insights(model, X, target_name, business_type, language_choice)

            # Store all necessary data for PDF generation
            st.session_state["target_name"] = target_name if 'target_name' in locals() else None
            st.session_state["business_type"] = business_type
            st.session_state["analysis_type"] = analysis_type
            
            # Export options - Updated call with all required parameters
            if st.session_state.get("kpi_data") and feature_df is not None:
                handle_export_requests(
                    export_formats=["📋 PDF Report"], 
                    language=language_choice, 
                    report_style=report_style,
                    df=st.session_state["data_engineered"],
                    kpis=st.session_state["kpi_data"],
                    feature_df=feature_df,
                    business_context=business_type,
                    target_variable=target_name,
                    analysis_type=analysis_type
                )
            
            st.success("✅ Business intelligence analysis completed!")

# 8. ADDITIONAL HELPER FUNCTIONS FOR ENHANCED PDF CONTENT

def create_report_sections_content(report_style, ai_content):
    """Parse and structure AI content into proper sections"""
    sections = {}
    
    if report_style == "Executive Summary":
        # Expected sections for executive summary
        section_markers = [
            "Business Performance Overview",
            "Key Findings", 
            "Critical Success Factors",
            "Strategic Recommendations",
            "Financial Impact"
        ]
    
    elif report_style == "Detailed Analysis":
        # Expected sections for detailed analysis
        section_markers = [
            "Executive Summary",
            "Current Performance Analysis",
            "Factor Analysis",
            "Trend Analysis", 
            "Risk Assessment",
            "Detailed Recommendations",
            "Implementation Roadmap",
            "Expected ROI"
        ]
    
    elif report_style == "Technical Report":
        # Expected sections for technical report
        section_markers = [
            "Data Overview",
            "Feature Engineering",
            "Model Performance",
            "Feature Importance Analysis",
            "Model Interpretation",
            "Statistical Insights",
            "Technical Recommendations",
            "Limitations & Assumptions",
            "Future Enhancements"
        ]
    
    # Parse AI content into sections
    current_section = "Introduction"
    sections[current_section] = []
    
    lines = ai_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is a section header
        is_section_header = False
        for marker in section_markers:
            if marker.lower() in line.lower() and ('**' in line or '##' in line or '#' in line):
                current_section = marker
                sections[current_section] = []
                is_section_header = True
                break
        
        if not is_section_header:
            sections[current_section].append(line)
    
    return sections

def add_charts_to_story(story, chart_files, styles):
    """Add charts to PDF story with proper formatting"""
    for chart_title, chart_file in chart_files:
        if os.path.exists(chart_file):
            # Add chart title
            story.append(Paragraph(chart_title, styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Add chart image
            try:
                img = Image(chart_file, width=6*inch, height=3*inch)
                story.append(img)
            except Exception as e:
                # Fallback if image can't be loaded
                story.append(Paragraph(f"[Chart: {chart_title} - Error loading image]", styles['Italic']))
            
            story.append(Spacer(1, 0.2*inch))
    
    return story

def create_enhanced_kpi_table(kpis, styles):
    """Create an enhanced KPI table with better formatting"""
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    
    # Prepare data with categories
    financial_kpis = []
    performance_kpis = []
    
    for key, value in kpis.items():
        clean_name = key.replace('_', ' ').title()
        formatted_value = f"${value:,.0f}" if 'total' in key.lower() or 'profit' in key.lower() or 'revenue' in key.lower() else f"{value:.2f}%"
        
        if any(word in key.lower() for word in ['revenue', 'profit', 'loss', 'ebit']):
            financial_kpis.append([clean_name, formatted_value])
        else:
            performance_kpis.append([clean_name, formatted_value])
    
    # Create tables
    tables = []
    
    if financial_kpis:
        financial_data = [["Financial Metric", "Value"]] + financial_kpis
        financial_table = Table(financial_data, colWidths=[3*inch, 2*inch])
        financial_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),
            ('TEXTCOLOR',(0,0),(-1,0), colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND',(0,1),(-1,-1), colors.lightgreen),
            ('GRID',(0,0),(-1,-1),1, colors.black)
        ]))
        tables.append(("Financial KPIs", financial_table))
    
    if performance_kpis:
        performance_data = [["Performance Metric", "Value"]] + performance_kpis
        performance_table = Table(performance_data, colWidths=[3*inch, 2*inch])
        performance_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR',(0,0),(-1,0), colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND',(0,1),(-1,-1), colors.lightblue),
            ('GRID',(0,0),(-1,-1),1, colors.black)
        ]))
        tables.append(("Performance KPIs", performance_table))
    
    return tables

# 9. ERROR HANDLING AND FALLBACKS

def safe_chart_generation(df, kpis):
    """Safe chart generation with error handling"""
    chart_files = []
    
    try:
        # Try to create KPI chart
        if kpis:
            chart_files.extend(create_summary_charts_for_pdf(df, kpis))
    except Exception as e:
        st.warning(f"Some charts could not be generated: {str(e)}")
    
    # Always return at least an empty list
    return chart_files if chart_files else []

def validate_pdf_data(df, kpis, feature_df, business_context, target_variable, analysis_type):
    """Validate data before PDF generation"""
    errors = []
    
    if df is None or df.empty:
        errors.append("Dataset is empty or missing")
    
    if not kpis:
        errors.append("KPIs are missing")
    
    if feature_df is None or feature_df.empty:
        errors.append("Feature importance data is missing")
    
    if not business_context:
        errors.append("Business context is not specified")
    
    if not target_variable:
        errors.append("Target variable is not specified")
    
    if not analysis_type:
        errors.append("Analysis type is not specified")
    
    return errors

# 10. FINAL INTEGRATION FUNCTION TO REPLACE IN YOUR MAIN CODE

def final_handle_export_requests_integration(export_formats, language, report_style, df=None, kpis=None, 
                                           feature_df=None, business_context=None, target_variable=None, analysis_type=None):
    """Final version with complete error handling"""
    st.subheader("📤 Export Your Analysis")
    
    if "📋 PDF Report" in export_formats:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"📊 Report Type: {report_style} | Language: {language}")
            
            # Validate data before showing options
            validation_errors = validate_pdf_data(df, kpis, feature_df, business_context, target_variable, analysis_type)
            
            if validation_errors:
                st.warning("⚠️ Missing data for PDF generation:")
                for error in validation_errors:
                    st.write(f"• {error}")
                st.write("Please run the complete analysis first.")
            else:
                # Show what will be included
                inclusions = {
                    "Executive Summary": ["Key findings", "Strategic recommendations", "Financial impact", "Charts"],
                    "Detailed Analysis": ["Performance analysis", "Factor analysis", "Risk assessment", "Implementation roadmap", "Charts"],
                    "Technical Report": ["Model performance", "Feature importance", "Statistical insights", "Technical recommendations", "Charts"]
                }
                
                st.write("**Report will include:**")
                for item in inclusions[report_style]:
                    st.write(f"✅ {item}")
        
        with col2:
            # Only show button if data is valid
            if not validation_errors:
                if st.button("📋 Generate PDF Report", type="primary", use_container_width=True):
                    try:
                        # Generate PDF with error handling
                        with st.spinner(f"Generating {report_style} in {language}..."):
                            pdf_bytes = generate_comprehensive_pdf_report(
                                kpis=kpis,
                                feature_df=feature_df,
                                business_context=business_context,
                                target_variable=target_variable,
                                language=language,
                                analysis_type=analysis_type,
                                report_style=report_style,
                                df=df
                            )
                        
                        # Success message
                        st.success(f"✅ {report_style} generated successfully!")
                        
                        # Download button
                        filename = f"business_analysis_{report_style.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
                        st.error("Please check your Gemini API configuration and try again.")
                        # Log the full error for debugging
                        st.write("Error details:", str(e))
            else:
                st.button("📋 Generate PDF Report", disabled=True, use_container_width=True)    