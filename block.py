
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