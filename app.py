import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

import os
# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure TensorFlow to use CPU
try:
    tf.config.set_visible_devices([], 'GPU')
    physical_devices = tf.config.list_physical_devices('CPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Load the trained model with h5 format
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model('darija_sentiment_model.keras', compile=False)
            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        
        # Load the tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_sentiment(text, model, tokenizer):
    try:
        with tf.device('/CPU:0'):
            sequences = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
            prediction = model.predict(padded, verbose=0)[0][0]
            sentiment = "إيجابي (Positive)" if prediction > 0.5 else "سلبي (Negative)"
            confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        return sentiment, confidence
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def analyze_dataset(df, model, tokenizer):
    if 'text' not in df.columns:
        st.error("The uploaded file must contain a 'text' column")
        return None
    
    results = []
    total = len(df)
    progress_bar = st.progress(0)
    
    with tf.device('/CPU:0'):
        for idx, text in enumerate(df['text']):
            try:
                sequences = tokenizer.texts_to_sequences([str(text)])
                padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
                prediction = model.predict(padded, verbose=0)[0][0]
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                confidence = float(prediction if prediction > 0.5 else 1 - prediction)
                results.append({'text': text, 'sentiment': sentiment, 'confidence': confidence})
                
                # Update progress bar
                progress_bar.progress((idx + 1) / total)
            except Exception as e:
                st.warning(f"Skipped one entry due to error: {str(e)}")
                continue
    
    progress_bar.empty()
    return pd.DataFrame(results)

def main():
    try:
        physical_devices = tf.config.list_physical_devices('CPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
        
    st.set_page_config(
        page_title="Darija Sentiment Analysis",
        page_icon="🇲🇦",
        layout="wide"
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Please ensure model files (darija_sentiment_model.keras and tokenizer.pickle) are in the same directory as this script.")
        return
    
    st.title("تحليل المشاعر بالدارجة 🇲🇦")
    st.title("Darija Sentiment Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Text Analysis", "Dataset Analysis"])
    
    # Tab 1: Single Text Analysis
    with tab1:
        st.subheader("تحليل النص / Text Analysis")
        
        # Text input
        text_input = st.text_area(
            "أدخل النص بالدارجة / Enter Darija text:",
            height=100
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("تحليل / Analyze", key="analyze_single"):
                if text_input.strip() == "":
                    st.warning("Please enter some text to analyze.")
                else:
                    with st.spinner("جاري التحليل... / Analyzing..."):
                        sentiment, confidence = predict_sentiment(text_input, model, tokenizer)
                        
                        # Display results
                        st.subheader("النتيجة / Result:")
                        st.markdown(f"*المشاعر / Sentiment:* {sentiment}")
                        st.markdown(f"*نسبة الثقة / Confidence:* {confidence:.2%}")
                        
                        # Progress bar for confidence
                        st.progress(confidence)
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence * 100,
                            title = {'text': "Confidence"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "gray"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig)
    
    # Tab 2: Dataset Analysis
    with tab2:
        st.subheader("تحليل البيانات / Dataset Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV, Excel, or TXT file with a 'text' column)",
            type=['csv', 'xlsx', 'txt']
        )
        
        if uploaded_file is not None:
            try:
                # Handle different file types
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.txt'):
                    # Assume each line in the TXT file represents a text entry
                    df = pd.DataFrame({'text': uploaded_file.read().decode('utf-8').splitlines()})
                
                if 'text' not in df.columns:
                    st.error("The uploaded file must contain a 'text' column")
                else:
                    if st.button("Analyze Dataset", key="analyze_dataset"):
                        with st.spinner("Analyzing dataset..."):
                            results_df = analyze_dataset(df, model, tokenizer)
                            
                            if results_df is not None:
                                # Display summary statistics
                                st.subheader("Summary Statistics")
                                
                                # Calculate sentiment distribution
                                sentiment_counts = results_df['sentiment'].value_counts()
                                
                                # Create two columns for charts
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Pie chart
                                    fig_pie = px.pie(
                                        values=sentiment_counts.values,
                                        names=sentiment_counts.index,
                                        title="Sentiment Distribution"
                                    )
                                    st.plotly_chart(fig_pie)
                                
                                with col2:
                                    # Bar chart
                                    fig_bar = px.bar(
                                        x=sentiment_counts.index,
                                        y=sentiment_counts.values,
                                        title="Sentiment Counts",
                                        labels={'x': 'Sentiment', 'y': 'Count'}
                                    )
                                    st.plotly_chart(fig_bar)
                                
                                # Display confidence distribution
                                st.subheader("Confidence Distribution")
                                fig_hist = px.histogram(
                                    results_df,
                                    x='confidence',
                                    nbins=20,
                                    title="Confidence Distribution"
                                )
                                st.plotly_chart(fig_hist)
                                
                                # Display the first few results
                                st.subheader("Sample Results")
                                st.dataframe(results_df.head())
                                
                                # Download button for results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Results",
                                    data=csv,
                                    file_name="sentiment_analysis_results.csv",
                                    mime="text/csv"
                                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    
    # Sidebar information
    st.sidebar.title("معلومات / Information")
    st.sidebar.info(
        """
        This app analyzes sentiment in Darija (Moroccan Arabic) text.
        
        هذا التطبيق يحلل المشاعر في النصوص المكتوبة بالدارجة المغربية
        """
    )
    
    # Example usage
    st.sidebar.title("أمثلة / Examples")
    st.sidebar.markdown(
        """
        - مزيان بزاف هاد الفيلم
        - ما عجبنيش هاد الماكلة
        - شكرا على هاد الخدمة الرائعة
        """
    )

if __name__ == "__main__":
    main()