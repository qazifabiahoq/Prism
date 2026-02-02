import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re
from groq import Groq

# Page configuration
st.set_page_config(
    page_title="MoneyMind - Smart Money, Smarter You",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Finance Industry Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-blue: #1E3A8A;
        --accent-green: #10B981;
        --light-bg: #F9FAFB;
        --card-bg: #FFFFFF;
        --text-dark: #1F2937;
        --text-light: #6B7280;
        --border-color: #E5E7EB;
        --success-green: #059669;
        --warning-red: #DC2626;
    }
    
    .stApp {
        background: var(--light-bg);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    body, p, div, span, label, input, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-dark);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: var(--text-dark);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, #1E40AF 100%);
        padding: 2.5rem 2rem;
        border-radius: 0;
        margin: -6rem -5rem 2rem -5rem;
        text-align: center;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .app-tagline {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Cards */
    .card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 1rem;
    }
    
    /* Metrics */
    .metric-container {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-light);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-blue);
        margin: 0.5rem 0;
    }
    
    .metric-value.positive {
        color: var(--success-green);
    }
    
    .metric-value.negative {
        color: var(--warning-red);
    }
    
    .metric-subtext {
        font-size: 0.8rem;
        color: var(--text-light);
        margin-top: 0.25rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, #1E40AF 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        width: 100%;
        height: 50px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    .stDownloadButton > button {
        background: var(--accent-green);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: var(--success-green);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: var(--card-bg);
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-blue);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--card-bg);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--primary-blue);
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Text inputs */
    .stTextInput input,
    .stTextArea textarea {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-blue);
        color: #FFFFFF;
        border-color: var(--primary-blue);
    }
    
    /* Section dividers */
    .section-divider {
        border-top: 1px solid var(--border-color);
        margin: 2rem 0;
    }
    
    /* Professional insight box */
    .insight-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 4px solid var(--primary-blue);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .insight-box h4 {
        color: var(--primary-blue);
        margin-bottom: 0.75rem;
    }
    
    .insight-box p {
        color: var(--text-dark);
        line-height: 1.6;
        margin: 0;
    }
    
    /* Data table styling */
    .dataframe {
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Hide emoji */
    .no-emoji {
        font-style: normal;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    """Initialize Groq API client"""
    api_key = st.secrets.get("groq", {}).get("api_key", None)
    if not api_key:
        st.error("⚠️ Groq API key not found. Please add it to Streamlit secrets.")
        st.stop()
    return Groq(api_key=api_key)

def analyze_spending_with_ai(df, question=""):
    """Use Groq AI to analyze spending patterns"""
    try:
        client = get_groq_client()
        
        # Prepare data summary
        total_transactions = len(df)
        
        # Try to detect amount column
        amount_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['amount', 'value', 'total', 'price']):
                amount_col = col
                break
        
        if amount_col:
            total_spent = df[amount_col].sum()
            avg_transaction = df[amount_col].mean()
            
            data_summary = f"""
Transaction Data Summary:
- Total Transactions: {total_transactions}
- Total Amount: ${total_spent:,.2f}
- Average Transaction: ${avg_transaction:,.2f}
- Date Range: {df.columns[0] if len(df.columns) > 0 else 'N/A'}

Sample Transactions:
{df.head(10).to_string()}
"""
        else:
            data_summary = f"""
Transaction Data Summary:
- Total Transactions: {total_transactions}
- Columns: {', '.join(df.columns)}

Sample Data:
{df.head(10).to_string()}
"""
        
        # Create prompt
        if question:
            prompt = f"""You are a professional financial advisor. Answer this question about the user's spending:

{question}

Based on this data:
{data_summary}

Provide a clear, actionable answer in 2-3 paragraphs. Focus on practical insights."""
        else:
            prompt = f"""You are a professional financial advisor. Analyze this spending data and provide insights:

{data_summary}

Provide:
1. Key spending patterns (2-3 observations)
2. One actionable recommendation
3. One potential savings opportunity

Keep it concise and professional (3-4 paragraphs max)."""
        
        # Call Groq API
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional financial advisor helping young adults understand their finances. Be clear, encouraging, and practical."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"I'm having trouble analyzing your data right now. Please make sure your Groq API key is set up correctly. Error: {str(e)}"

def categorize_transaction(description):
    """Simple rule-based transaction categorization"""
    description_lower = str(description).lower()
    
    # Food & Dining
    if any(word in description_lower for word in ['restaurant', 'cafe', 'coffee', 'starbucks', 'mcdonald', 'food', 'grocery', 'uber eats', 'doordash']):
        return 'Food & Dining'
    
    # Transportation
    if any(word in description_lower for word in ['uber', 'lyft', 'gas', 'fuel', 'parking', 'transit', 'subway']):
        return 'Transportation'
    
    # Shopping
    if any(word in description_lower for word in ['amazon', 'walmart', 'target', 'shopping', 'store', 'mall']):
        return 'Shopping'
    
    # Entertainment
    if any(word in description_lower for word in ['netflix', 'spotify', 'movie', 'theater', 'game', 'entertainment']):
        return 'Entertainment'
    
    # Bills & Utilities
    if any(word in description_lower for word in ['electric', 'water', 'internet', 'phone', 'utility', 'bill']):
        return 'Bills & Utilities'
    
    # Default
    return 'Other'

def create_spending_chart(df):
    """Create a professional spending visualization"""
    try:
        # Try to find amount and category columns
        amount_col = None
        date_col = None
        desc_col = None
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['amount', 'value', 'total', 'price']) and amount_col is None:
                amount_col = col
            elif any(keyword in col.lower() for keyword in ['date', 'time']) and date_col is None:
                date_col = col
            elif any(keyword in col.lower() for keyword in ['description', 'merchant', 'name', 'category']) and desc_col is None:
                desc_col = col
        
        if amount_col and desc_col:
            # Categorize transactions
            df['Category'] = df[desc_col].apply(categorize_transaction)
            
            # Create category spending chart
            category_spending = df.groupby('Category')[amount_col].sum().sort_values(ascending=True)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=category_spending.index,
                    x=category_spending.values,
                    orientation='h',
                    marker=dict(
                        color=['#1E3A8A', '#2563EB', '#3B82F6', '#60A5FA', '#93C5FD', '#DBEAFE'],
                    ),
                    text=[f'${v:,.0f}' for v in category_spending.values],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title='Spending by Category',
                xaxis_title='Amount ($)',
                yaxis_title='',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif')
            )
            
            return fig
        
        return None
        
    except Exception as e:
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1 class="app-title">MoneyMind</h1>
    <p class="app-tagline">Smart Money, Smarter You</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### About MoneyMind")
    st.markdown("""
    Your AI-powered financial assistant for better money management.
    
    **Features:**
    - Upload bank statements (CSV)
    - AI spending analysis
    - Budget recommendations
    - Financial Q&A
    """)
    
    st.markdown("---")
    
    st.markdown("### How to Use")
    st.markdown("""
    1. **Upload** your bank statement CSV
    2. **Review** AI-powered insights
    3. **Ask** financial questions
    4. **Get** personalized advice
    """)
    
    st.markdown("---")
    
    st.markdown("### Setup Required")
    st.markdown("""
    **Groq API Key** (Free)
    
    1. Sign up at [console.groq.com](https://console.groq.com)
    2. Get your free API key
    3. Add to Streamlit secrets
    
    **CSV Format:**
    - Date column
    - Amount column
    - Description/Merchant column
    """)

# Main content - Tabs
tab1, tab2, tab3 = st.tabs(["📊 Upload Data", "💡 Insights", "💬 Ask MoneyMind"])

with tab1:
    st.markdown("### Upload Your Bank Statement")
    st.markdown("Upload a CSV file with your transaction history to get started.")
    
    uploaded_file = st.file_uploader(
        "Choose your CSV file",
        type=['csv'],
        help="Upload a CSV file with columns like: Date, Amount, Description"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            
            st.success("✓ File uploaded successfully!")
            
            # Show preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Total Rows</div>
                    <div class="metric-value">{len(df):,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Columns</div>
                    <div class="metric-value">{len(df.columns)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Try to find amount column
                amount_col = None
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['amount', 'value', 'total']):
                        amount_col = col
                        break
                
                if amount_col:
                    total = df[amount_col].sum()
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Total Amount</div>
                        <div class="metric-value">${total:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-container">
                        <div class="metric-label">Status</div>
                        <div class="metric-value" style="font-size: 1.2rem;">Ready</div>
                    </div>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Make sure your file is a valid CSV with proper formatting.")
    
    else:
        st.info("👆 Upload a CSV file to get started")
        
        st.markdown("""
        <div class="insight-box">
            <h4>CSV Format Example</h4>
            <p>Your CSV should have columns like:</p>
            <ul>
                <li><strong>Date:</strong> Transaction date (e.g., 2024-01-15)</li>
                <li><strong>Amount:</strong> Transaction amount (e.g., 45.99)</li>
                <li><strong>Description:</strong> Merchant or description (e.g., "Starbucks")</li>
            </ul>
            <p style="margin-top: 1rem;"><em>Other formats may work too - MoneyMind will try to detect your columns automatically.</em></p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### AI-Powered Insights")
    
    if 'data' in st.session_state and st.session_state['data'] is not None:
        df = st.session_state['data']
        
        # Generate insights button
        if st.button("🔍 Generate AI Insights", use_container_width=True):
            with st.spinner("Analyzing your spending patterns..."):
                insights = analyze_spending_with_ai(df)
                st.session_state['insights'] = insights
        
        # Show insights if available
        if 'insights' in st.session_state:
            st.markdown(f"""
            <div class="insight-box">
                <h4>Your Financial Analysis</h4>
                <p>{st.session_state['insights']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show chart if possible
        st.markdown("#### Spending Breakdown")
        chart = create_spending_chart(df)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Upload a CSV with Amount and Description columns to see spending breakdown")
        
        # Top transactions
        try:
            amount_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['amount', 'value', 'total']):
                    amount_col = col
                    break
            
            if amount_col:
                st.markdown("#### Largest Transactions")
                top_transactions = df.nlargest(5, amount_col)
                st.dataframe(top_transactions, use_container_width=True)
        except:
            pass
        
    else:
        st.info("👈 Upload a CSV file in the 'Upload Data' tab first")

with tab3:
    st.markdown("### Ask MoneyMind Anything")
    
    if 'data' in st.session_state and st.session_state['data'] is not None:
        df = st.session_state['data']
        
        # Quick questions
        st.markdown("#### Quick Questions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("How can I save more money?", use_container_width=True):
                st.session_state['question'] = "Based on my spending, how can I save more money?"
        
        with col2:
            if st.button("Where am I overspending?", use_container_width=True):
                st.session_state['question'] = "Where am I overspending the most?"
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("Budget recommendations?", use_container_width=True):
                st.session_state['question'] = "What budget should I follow based on my spending?"
        
        with col4:
            if st.button("Biggest expenses?", use_container_width=True):
                st.session_state['question'] = "What are my biggest expense categories?"
        
        st.markdown("---")
        
        # Custom question
        question = st.text_input(
            "Or ask your own question:",
            placeholder="e.g., Should I pay off my credit card or invest?",
            value=st.session_state.get('question', '')
        )
        
        if st.button("Get Answer", use_container_width=True) and question:
            with st.spinner("Thinking..."):
                answer = analyze_spending_with_ai(df, question)
                
                st.markdown(f"""
                <div class="insight-box">
                    <h4>Answer</h4>
                    <p>{answer}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Clear the question
                if 'question' in st.session_state:
                    del st.session_state['question']
        
    else:
        st.info("👈 Upload a CSV file first to ask questions about your spending")
        
        st.markdown("""
        <div class="insight-box">
            <h4>What You Can Ask</h4>
            <p>Once you upload your data, you can ask questions like:</p>
            <ul>
                <li>"How much am I spending on food each month?"</li>
                <li>"Should I pay off my credit card or start investing?"</li>
                <li>"What percentage of my income should go to savings?"</li>
                <li>"Where can I cut back on spending?"</li>
                <li>"Am I on track with my budget?"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-light); font-size: 0.875rem;">
    <p><strong>MoneyMind</strong> - Smart Money, Smarter You</p>
    <p>Built for CJCP Hacks 25-26 | Powered by Groq AI</p>
</div>
""", unsafe_allow_html=True)
