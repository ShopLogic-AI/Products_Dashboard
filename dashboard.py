import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
from groq_utils import get_rag_response


# Page configuration
st.set_page_config(
    page_title="eCommerce Intelligence Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)


# def get_groq_response(messages, model="llama-3.3-70b-versatile"):
#     response = client.chat.completions.create(
#         messages=messages,
#         model=model,
#         temperature=0.5,
#         max_tokens=1024,
#         stream=False
#     )
#     return response.choices[0].message.content.strip()

@st.cache_data
def load_product_data():
    try:
        df = pd.read_csv("product_data.csv")
        return df
    except Exception as e:
        st.warning(f"Erreur de chargement des données: {e}")
        return pd.DataFrame()

df = load_product_data()

def get_data_summary(df):
    if df.empty:
        return "Pas de données disponibles."

    summary = f"""
Voici un résumé des données produits:

- Nombre total de produits : {len(df)}
- Vendeurs uniques : {df['vendor'].nunique()}
- Types de produits : {df['product_type'].nunique()}
- Prix moyen : ${df['price'].mean():.2f}
- Stock moyen : {df['available'].mean():.1f} unités
- Produits en stock critique (≤2) : {len(df[df['available'] <= 2])}
- Produits avec remise : {len(df[df['discount_percentage'] > 0])}
- Remise moyenne : {df[df['discount_percentage'] > 0]['discount_percentage'].mean():.1f}%
"""

    return summary

# def get_groq_response(messages, model="llama-3.3-70b-versatile"):
#     data_context = get_data_summary(df)  # Use the global df
#     system_prompt = f"""
# Tu es un assistant e-commerce intelligent.
#
# Voici le contexte des données produits disponibles :
# {data_context}
#
# Instructions :
# - Réponds en français
# - Utilise les données fournies pour donner des réponses précises
# - Donne des recommandations actionnables
# - Utilise des emojis pour plus de clarté
# """
#
#     # Insert system prompt at the beginning
#     full_messages = [{"role": "system", "content": system_prompt}] + messages
#
#     try:
#         response = client.chat.completions.create(
#             messages=full_messages,
#             model=model,
#             temperature=0.5,
#             max_tokens=1500,
#             stream=False
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"❌ Erreur lors de l'appel à Groq : {str(e)}"

def get_groq_response(messages):
    user_prompt = messages[-1]["content"]
    return get_rag_response(user_prompt)

warnings.filterwarnings('ignore')


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .debug-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Database connection function with better error handling
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from MySQL database with comprehensive error handling"""
    try:
        # Try to connect to database
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="ecommerce_data",
            connect_timeout=10
        )

        # First, let's check what data exists
        check_query = "SELECT COUNT(*) as total_rows FROM shopify_products_variants"
        total_rows = pd.read_sql(check_query, conn)

        if total_rows['total_rows'].iloc[0] == 0:
            st.error("❌ Database table 'shopify_products_variants' is empty!")
            conn.close()
            return pd.DataFrame(), "Empty table"

        # Load data with more flexible query (handle NULLs better)
        query = """
                SELECT id, \
                       product_id, \
                       title, \
                       handle, \
                       description, \
                       vendor, \
                       product_type, \
                       tags, \
                       published_at, \
                       created_at, \
                       updated_at, \
                       variant_id, \
                       variant_title, \
                       option1, \
                       option2, \
                       option3, \
                       sku, \
                       requires_shipping, \
                       taxable, \
                       COALESCE(available, 0)        as available, \
                       COALESCE(price, 0)            as price, \
                       COALESCE(compare_at_price, 0) as compare_at_price, \
                       COALESCE(grams, 0)            as grams, \
                       variant_created_at, \
                       variant_updated_at, \
                       variant_featured_image_url, \
                       product_main_image_url, \
                       option_color, \
                       option_size
                FROM shopify_products_variants \
                """

        df = pd.read_sql(query, conn)
        conn.close()

        # Data preprocessing with better error handling
        try:
            # Handle datetime columns
            for col in ['created_at', 'updated_at', 'published_at', 'variant_created_at', 'variant_updated_at']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # Fill missing values
            df['vendor'] = df['vendor'].fillna('Unknown Vendor')
            df['product_type'] = df['product_type'].fillna('Unknown Type')
            df['title'] = df['title'].fillna('Untitled Product')
            df['description'] = df['description'].fillna('')
            df['tags'] = df['tags'].fillna('')

            # Create derived columns
            df['discount_percentage'] = np.where(
                (df['compare_at_price'] > 0) & (df['price'] > 0),
                ((df['compare_at_price'] - df['price']) / df['compare_at_price'] * 100),
                0
            )

            df['stock_status'] = df['available'].apply(
                lambda
                    x: 'Out of Stock' if x == 0 else 'Low Stock' if x <= 2 else 'Medium Stock' if x <= 10 else 'High Stock'
            )

            # Create price categories with better handling
            df['price_category'] = pd.cut(
                df['price'],
                bins=[0, 25, 50, 100, 500, float('inf')],
                labels=['Budget', 'Economy', 'Mid-range', 'Premium', 'Luxury'],
                include_lowest=True
            )
            df['price_category'] = df['price_category'].fillna('Budget')

            return df, "success"

        except Exception as e:
            st.error(f"❌ Error in data preprocessing: {str(e)}")
            return df, f"preprocessing_error: {str(e)}"

    except mysql.connector.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        st.error(f"❌ {error_msg}")
        return pd.DataFrame(), error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        st.error(f"❌ {error_msg}")
        return pd.DataFrame(), error_msg


# Load data
df, load_status = load_data()

# Debug information
if st.sidebar.checkbox("🔧 Show Debug Info"):
    st.sidebar.markdown("### 🔍 Debug Information")
    st.sidebar.write(f"**Load Status:** {load_status}")
    if not df.empty:
        st.sidebar.write(f"**Total Rows:** {len(df):,}")
        st.sidebar.write(f"**Columns:** {len(df.columns)}")
        st.sidebar.write(f"**Price Range:** ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        st.sidebar.write(f"**Available Range:** {df['available'].min()} - {df['available'].max()}")
        st.sidebar.write(f"**Null Values:**")
        null_counts = df.isnull().sum()
        for col, count in null_counts[null_counts > 0].items():
            st.sidebar.write(f"  - {col}: {count}")

if df.empty:
    st.error("❌ No data available. Please check your database connection and ensure the table has data.")

    # Show connection test
    st.markdown("### 🔧 Database Connection Test")
    if st.button("Test Database Connection"):
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="ecommerce_data"
            )
            st.success("✅ Database connection successful!")

            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES LIKE 'shopify_products_variants'")
            result = cursor.fetchone()

            if result:
                st.success("✅ Table 'shopify_products_variants' exists!")

                # Check row count
                cursor.execute("SELECT COUNT(*) FROM shopify_products_variants")
                count = cursor.fetchone()[0]
                st.info(f"📊 Table has {count} rows")

                if count == 0:
                    st.warning("⚠️ Table is empty. Please insert some data first.")
                else:
                    # Show sample data
                    cursor.execute("SELECT * FROM shopify_products_variants LIMIT 5")
                    columns = [desc[0] for desc in cursor.description]
                    sample_data = cursor.fetchall()
                    sample_df = pd.DataFrame(sample_data, columns=columns)
                    st.write("**Sample Data:**")
                    st.dataframe(sample_df)
            else:
                st.error("❌ Table 'shopify_products_variants' does not exist!")

            conn.close()

        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")

    st.stop()

st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Overview", "📊 Product Insights", "📦 Stock Analysis",
         "💰 Price Dynamics", "🏪 Vendor Comparison", "🔍 Advanced Filters", "📥 Export & Download", "🤖 AI Assistant"]
)

# Sidebar Filters (Global) with better defaults
st.sidebar.markdown("## 🎛️ Global Filters")

# Product type filter
product_types = ['All'] + sorted([pt for pt in df['product_type'].unique() if pd.notna(pt)])
selected_product_type = st.sidebar.selectbox("Product Type", product_types)

# Vendor filter
vendors = ['All'] + sorted([v for v in df['vendor'].unique() if pd.notna(v)])
selected_vendor = st.sidebar.selectbox("Vendor", vendors)

# Price range filter with better handling
if df['price'].max() > 0:
    price_min, price_max = st.sidebar.slider(
        "Price Range ($)",
        min_value=0.0,
        max_value=float(df['price'].max()),
        value=(0.0, float(df['price'].max())),
        step=1.0
    )
else:
    price_min, price_max = 0.0, 1000.0
    st.sidebar.warning("⚠️ No valid price data found")

# Date range filter with better handling
if df['created_at'].notna().any():
    date_min = df['created_at'].min().date()
    date_max = df['created_at'].max().date()
    selected_date_range = st.sidebar.date_input(
        "Creation Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )
else:
    selected_date_range = []
    st.sidebar.warning("⚠️ No valid date data found")

# Apply filters with better error handling
filtered_df = df.copy()

try:
    if selected_product_type != 'All':
        filtered_df = filtered_df[filtered_df['product_type'] == selected_product_type]

    if selected_vendor != 'All':
        filtered_df = filtered_df[filtered_df['vendor'] == selected_vendor]

    filtered_df = filtered_df[
        (filtered_df['price'] >= price_min) &
        (filtered_df['price'] <= price_max)
        ]

    if len(selected_date_range) == 2 and df['created_at'].notna().any():
        start_date, end_date = selected_date_range
        filtered_df = filtered_df[
            (filtered_df['created_at'].dt.date >= start_date) &
            (filtered_df['created_at'].dt.date <= end_date)
            ]

except Exception as e:
    st.sidebar.error(f"Filter error: {str(e)}")
    filtered_df = df.copy()

# Data freshness indicator
st.sidebar.markdown("## 📅 Data Freshness")
if df['updated_at'].notna().any():
    last_update = df['updated_at'].max()
    st.sidebar.info(f"Last updated: {last_update.strftime('%Y-%m-%d %H:%M')}")
else:
    st.sidebar.warning("No update timestamp available")

# Show filter results
st.sidebar.markdown("## 📊 Filter Results")
st.sidebar.info(f"Showing {len(filtered_df):,} of {len(df):,} products")

# Main content based on selected page
if page == "🏠 Overview":
    st.markdown('<h1 class="main-header">🛒 eCommerce Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Welcome to your comprehensive eCommerce analytics platform")

    # Key Performance Indicators
    st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_products = len(filtered_df)
        st.metric("🛍️ Total Products", f"{total_products:,}")

    with col2:
        low_stock = len(filtered_df[filtered_df['available'] <= 2])
        low_stock_pct = (low_stock / total_products * 100) if total_products > 0 else 0
        st.metric("⚠️ Low Stock Items", f"{low_stock:,}", delta=f"{low_stock_pct:.1f}%")

    with col3:
        avg_price = filtered_df['price'].mean() if len(filtered_df) > 0 else 0
        st.metric("💰 Average Price", f"${avg_price:.2f}")

    with col4:
        unique_vendors = filtered_df['vendor'].nunique()
        st.metric("🏪 Unique Vendors", f"{unique_vendors:,}")

    # Quick insights
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Stock Distribution</div>', unsafe_allow_html=True)
        if len(filtered_df) > 0:
            stock_dist = filtered_df['stock_status'].value_counts()
            fig_stock = px.pie(values=stock_dist.values, names=stock_dist.index,
                               color_discrete_sequence=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1'])
            fig_stock.update_layout(height=400)
            st.plotly_chart(fig_stock, use_container_width=True)
        else:
            st.info("No data available for stock distribution")

    with col2:
        st.markdown('<div class="section-header">💎 Price Categories</div>', unsafe_allow_html=True)
        if len(filtered_df) > 0:
            price_dist = filtered_df['price_category'].value_counts()
            fig_price = px.bar(x=price_dist.index, y=price_dist.values,
                               color=price_dist.values, color_continuous_scale='viridis')
            fig_price.update_layout(height=400, xaxis_title="Price Category", yaxis_title="Number of Products")
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("No data available for price categories")

# elif page == "🤖 AI Assistant":
#     st.markdown('<h1 class="main-header">🤖 AI Assistant</h1>', unsafe_allow_html=True)
#
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [{"role": "assistant",
#                                          "content": "Bonjour ! Je suis votre assistant intelligent en eCommerce. Comment puis-je vous aider ?"}]
#
#     for message in st.session_state["messages"]:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#     if prompt := st.chat_input("Posez une question sur les données ou demandez des recommandations."):
#         st.session_state["messages"].append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
#
#         with st.spinner("💬 Réflexion en cours..."):
#             response = get_groq_response(st.session_state["messages"])
#             st.session_state["messages"].append({"role": "assistant", "content": response})
#
#         with st.chat_message("assistant"):
#             st.markdown(response)

elif page == "🤖 AI Assistant":
    st.markdown('<h1 class="main-header">🤖 AI Assistant (MCP Architecture)</h1>', unsafe_allow_html=True)

    # MCP Architecture Info
    with st.expander("ℹ️ Architecture MCP", expanded=False):
        st.markdown("""
        **Architecture Model Context Protocol (MCP):**
        - 🏠 **MCP Host**: Interface Streamlit avec authentification et logs
        - 🤖 **MCP Client**: Wrapper LLM qui structure les prompts pour Groq
        - 🔒 **MCP Server**: Couche sécurisée qui filtre les données sensibles
        - 📊 **Isolation des données**: Seuls les résumés KPI sont partagés avec l'IA
        - 📝 **Audit**: Toutes les interactions sont loggées dans `mcp_log.txt`
        """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "Bonjour ! Je suis votre assistant intelligent eCommerce avec architecture MCP sécurisée. Comment puis-je vous aider à analyser vos données produits ?"
        }]

    # Display chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Posez une question sur les données ou demandez des recommandations..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response through MCP architecture
        with st.spinner("🔒 Traitement sécurisé via MCP..."):
            try:
                # Import here to avoid circular imports
                from groq_utils import get_groq_response

                response = get_groq_response(st.session_state["messages"])
                st.session_state["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Erreur MCP: {str(e)}"
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                response = error_msg

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

    # Quick action buttons
    st.markdown("### 🚀 Actions Rapides")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 KPI Généraux"):
            quick_prompt = "Donne-moi un résumé des KPI principaux de mon eCommerce"
            st.session_state["messages"].append({"role": "user", "content": quick_prompt})
            with st.spinner("Analyse en cours..."):
                from groq_utils import get_groq_response

                response = get_groq_response(st.session_state["messages"])
                st.session_state["messages"].append({"role": "assistant", "content": response})
            st.rerun()

    with col2:
        if st.button("⚠️ Produits Critiques"):
            quick_prompt = "Quels sont les produits critiques avec un stock inférieur à 2 et une remise supérieure à 20% ?"
            st.session_state["messages"].append({"role": "user", "content": quick_prompt})
            with st.spinner("Analyse en cours..."):
                from groq_utils import get_groq_response

                response = get_groq_response(st.session_state["messages"])
                st.session_state["messages"].append({"role": "assistant", "content": response})
            st.rerun()

    with col3:
        if st.button("🏪 Analyse Vendeurs"):
            quick_prompt = "Donne-moi une analyse des performances par vendeur"
            st.session_state["messages"].append({"role": "user", "content": quick_prompt})
            with st.spinner("Analyse en cours..."):
                from groq_utils import get_groq_response

                response = get_groq_response(st.session_state["messages"])
                st.session_state["messages"].append({"role": "assistant", "content": response})
            st.rerun()

    # Clear chat button
    if st.button("🗑️ Vider l'historique"):
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "Historique vidé. Comment puis-je vous aider ?"
        }]
        st.rerun()

    # Show logs info
    st.markdown("---")
    st.info("📝 Toutes les interactions sont enregistrées dans `mcp_log.txt` pour audit et conformité.")

elif page == "📦 Stock Analysis":
    st.markdown('<h1 class="main-header">📦 Stock Analysis</h1>', unsafe_allow_html=True)

    # Stock overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        out_of_stock = len(filtered_df[filtered_df['available'] == 0])
        st.metric("🚫 Out of Stock", f"{out_of_stock:,}")

    with col2:
        low_stock = len(filtered_df[(filtered_df['available'] > 0) & (filtered_df['available'] <= 2)])
        st.metric("⚠️ Low Stock (1-2)", f"{low_stock:,}")

    with col3:
        medium_stock = len(filtered_df[(filtered_df['available'] > 2) & (filtered_df['available'] <= 10)])
        st.metric("📦 Medium Stock (3-10)", f"{medium_stock:,}")

    with col4:
        high_stock = len(filtered_df[filtered_df['available'] > 10])
        st.metric("✅ High Stock (10+)", f"{high_stock:,}")

    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">📊 Stock Distribution by Product Type</div>',
                        unsafe_allow_html=True)
            stock_by_type = filtered_df.groupby('product_type')['available'].agg(['mean', 'sum', 'count']).reset_index()
            stock_by_type.columns = ['product_type', 'avg_stock', 'total_stock', 'product_count']

            if len(stock_by_type) > 0:
                fig_stock_type = px.bar(stock_by_type, x='product_type', y='total_stock',
                                        hover_data=['avg_stock', 'product_count'],
                                        title="Total Stock by Product Type")
                fig_stock_type.update_xaxes(tickangle=45)
                fig_stock_type.update_layout(height=500)
                st.plotly_chart(fig_stock_type, use_container_width=True)
            else:
                st.info("No product type data available")

        with col2:
            st.markdown('<div class="section-header">📈 Stock Level Distribution</div>', unsafe_allow_html=True)
            if filtered_df['available'].max() > 0:
                fig_hist = px.histogram(filtered_df, x='available', nbins=min(30, filtered_df['available'].nunique()),
                                        title="Distribution of Stock Levels")
                fig_hist.update_layout(height=500, xaxis_title="Available Quantity", yaxis_title="Number of Products")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No stock data available for histogram")

        # Critical stock alerts
        st.markdown('<div class="section-header">🚨 Critical Stock Alerts</div>', unsafe_allow_html=True)

        critical_stock = filtered_df[filtered_df['available'] <= 2].sort_values('available')
        if not critical_stock.empty:
            display_cols = ['title', 'product_type', 'vendor', 'available', 'price']
            # Only show columns that exist
            available_cols = [col for col in display_cols if col in critical_stock.columns]
            st.dataframe(
                critical_stock[available_cols].head(20),
                use_container_width=True
            )
        else:
            st.success("✅ No critical stock issues found!")
    else:
        st.warning("⚠️ No data available after applying filters")

elif page == "💰 Price Dynamics":
    st.markdown('<h1 class="main-header">💰 Price Dynamics</h1>', unsafe_allow_html=True)

    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">💵 Price vs Compare Price Analysis</div>', unsafe_allow_html=True)
            # Filter out products without compare_at_price
            price_comparison_df = filtered_df[
                (filtered_df['compare_at_price'].notna()) &
                (filtered_df['compare_at_price'] > 0) &
                (filtered_df['price'] > 0)
                ]

            if not price_comparison_df.empty:
                fig_price_scatter = px.scatter(price_comparison_df, x='price', y='compare_at_price',
                                               color='discount_percentage', size='available',
                                               hover_data=['title', 'vendor'],
                                               title="Current Price vs Original Price")

                # Add diagonal line for reference
                max_price = max(price_comparison_df['price'].max(), price_comparison_df['compare_at_price'].max())
                fig_price_scatter.add_trace(go.Scatter(x=[0, max_price], y=[0, max_price],
                                                       mode='lines', name='Equal Price Line',
                                                       line=dict(dash='dash', color='red')))
                fig_price_scatter.update_layout(height=500)
                st.plotly_chart(fig_price_scatter, use_container_width=True)
            else:
                st.info("No products with compare prices available")

        with col2:
            st.markdown('<div class="section-header">📊 Price Distribution by Product Type</div>',
                        unsafe_allow_html=True)
            if filtered_df['price'].max() > 0:
                fig_box = px.box(filtered_df, x='product_type', y='price',
                                 title="Price Distribution by Product Type")
                fig_box.update_xaxes(tickangle=45)
                fig_box.update_layout(height=500)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No valid price data available")

        # Discount analysis
        st.markdown('<div class="section-header">🏷️ Discount Analysis</div>', unsafe_allow_html=True)

        discount_df = filtered_df[filtered_df['discount_percentage'] > 0]

        if not discount_df.empty:
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_discount = discount_df['discount_percentage'].mean()
                st.metric("📉 Average Discount", f"{avg_discount:.1f}%")

            with col2:
                max_discount = discount_df['discount_percentage'].max()
                st.metric("🎯 Maximum Discount", f"{max_discount:.1f}%")

            with col3:
                discounted_products = len(discount_df)
                st.metric("🏷️ Products on Sale", f"{discounted_products:,}")

            # Discount distribution
            fig_discount = px.histogram(discount_df, x='discount_percentage', nbins=20,
                                        title="Distribution of Discount Percentages")
            fig_discount.update_layout(height=400)
            st.plotly_chart(fig_discount, use_container_width=True)
        else:
            st.info("No products with discounts found in the current selection.")
    else:
        st.warning("⚠️ No data available after applying filters")

elif page == "🏪 Vendor Comparison":
    st.markdown('<h1 class="main-header">🏪 Vendor Comparison</h1>', unsafe_allow_html=True)

    if len(filtered_df) > 0:
        # Vendor performance metrics
        vendor_stats = filtered_df.groupby('vendor').agg({
            'product_id': 'count',
            'price': ['mean', 'min', 'max'],
            'available': ['sum', 'mean'],
            'discount_percentage': 'mean'
        }).round(2)

        vendor_stats.columns = ['Product_Count', 'Avg_Price', 'Min_Price', 'Max_Price',
                                'Total_Stock', 'Avg_Stock', 'Avg_Discount']
        vendor_stats = vendor_stats.reset_index()

        # Top vendors by product count
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">🏆 Top 15 Vendors by Product Count</div>', unsafe_allow_html=True)
            top_vendors = vendor_stats.nlargest(15, 'Product_Count')
            if not top_vendors.empty:
                fig_vendors = px.bar(top_vendors, x='Product_Count', y='vendor', orientation='h',
                                     color='Avg_Price', color_continuous_scale='viridis')
                fig_vendors.update_layout(height=600)
                st.plotly_chart(fig_vendors, use_container_width=True)
            else:
                st.info("No vendor data available")

        with col2:
            st.markdown('<div class="section-header">💰 Vendor Price Comparison</div>', unsafe_allow_html=True)
            if not vendor_stats.empty:
                fig_vendor_price = px.scatter(vendor_stats, x='Avg_Price', y='Product_Count',
                                              size='Total_Stock', color='Avg_Discount',
                                              hover_data=['vendor'],
                                              title="Vendor Performance Matrix")
                fig_vendor_price.update_layout(height=600)
                st.plotly_chart(fig_vendor_price, use_container_width=True)
            else:
                st.info("No vendor data available")

        # Vendor details table
        st.markdown('<div class="section-header">📋 Vendor Performance Table</div>', unsafe_allow_html=True)

        # Add search functionality
        search_vendor = st.text_input("🔍 Search Vendor:", placeholder="Enter vendor name...")

        if search_vendor:
            vendor_display = vendor_stats[vendor_stats['vendor'].str.contains(search_vendor, case=False, na=False)]
        else:
            vendor_display = vendor_stats.head(20)

        if not vendor_display.empty:
            st.dataframe(vendor_display, use_container_width=True)
        else:
            st.info("No vendors found matching your search")
    else:
        st.warning("⚠️ No data available after applying filters")

elif page == "🔍 Advanced Filters":
    st.markdown('<h1 class="main-header">🔍 Advanced Filters & Search</h1>', unsafe_allow_html=True)

    # Advanced filtering options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🎛️ Advanced Filters</div>', unsafe_allow_html=True)

        # Stock status filter
        stock_options = st.multiselect("Stock Status",
                                       options=['Out of Stock', 'Low Stock', 'Medium Stock', 'High Stock'],
                                       default=['Out of Stock', 'Low Stock', 'Medium Stock', 'High Stock'])

        # Price category filter
        available_price_cats = [cat for cat in ['Budget', 'Economy', 'Mid-range', 'Premium', 'Luxury']
                                if cat in filtered_df['price_category'].values]
        price_categories = st.multiselect("Price Categories",
                                          options=available_price_cats,
                                          default=available_price_cats)

        # Discount filter
        has_discount = st.checkbox("Only show products with discounts")

        # Availability threshold
        min_availability = st.number_input("Minimum Availability", min_value=0, value=0)

    with col2:
        st.markdown('<div class="section-header">🔍 Text Search</div>', unsafe_allow_html=True)

        # Text search
        search_title = st.text_input("Search in Product Title:", placeholder="Enter keywords...")
        search_description = st.text_input("Search in Description:", placeholder="Enter keywords...")
        search_tags = st.text_input("Search in Tags:", placeholder="Enter keywords...")

    # Apply advanced filters
    advanced_filtered_df = filtered_df.copy()

    try:
        if stock_options:
            advanced_filtered_df = advanced_filtered_df[advanced_filtered_df['stock_status'].isin(stock_options)]

        if price_categories:
            advanced_filtered_df = advanced_filtered_df[advanced_filtered_df['price_category'].isin(price_categories)]

        if has_discount:
            advanced_filtered_df = advanced_filtered_df[advanced_filtered_df['discount_percentage'] > 0]

        advanced_filtered_df = advanced_filtered_df[advanced_filtered_df['available'] >= min_availability]

        if search_title:
            advanced_filtered_df = advanced_filtered_df[
                advanced_filtered_df['title'].str.contains(search_title, case=False, na=False)
            ]

        if search_description:
            advanced_filtered_df = advanced_filtered_df[
                advanced_filtered_df['description'].str.contains(search_description, case=False, na=False)
            ]

        if search_tags:
            advanced_filtered_df = advanced_filtered_df[
                advanced_filtered_df['tags'].str.contains(search_tags, case=False, na=False)
            ]
    except Exception as e:
        st.error(f"Error applying filters: {str(e)}")
        advanced_filtered_df = filtered_df.copy()

    # Display results
    st.markdown('<div class="section-header">📊 Filtered Results</div>', unsafe_allow_html=True)
    st.info(f"Found {len(advanced_filtered_df)} products matching your criteria")

    if not advanced_filtered_df.empty:
        # Display options
        available_columns = ['title', 'product_type', 'vendor', 'price', 'compare_at_price',
                             'available', 'stock_status', 'discount_percentage', 'created_at']
        # Only show columns that exist in the dataframe
        available_columns = [col for col in available_columns if col in advanced_filtered_df.columns]

        display_columns = st.multiselect(
            "Select columns to display:",
            options=available_columns,
            default=available_columns[:6]  # Show first 6 by default
        )

        if display_columns:
            # Pagination
            items_per_page = st.selectbox("Items per page:", [10, 25, 50, 100], index=1)
            total_pages = len(advanced_filtered_df) // items_per_page + (
                1 if len(advanced_filtered_df) % items_per_page > 0 else 0)

            if total_pages > 1:
                page_num = st.selectbox("Page:", range(1, total_pages + 1))
                start_idx = (page_num - 1) * items_per_page
                end_idx = start_idx + items_per_page
                display_df = advanced_filtered_df[display_columns].iloc[start_idx:end_idx]
            else:
                display_df = advanced_filtered_df[display_columns]

            st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("No products found matching your criteria. Try adjusting your filters.")

elif page == "📥 Export & Download":
    st.markdown('<h1 class="main-header">📥 Export & Download</h1>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Export Options</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Data Export")

        # Select data to export
        export_option = st.selectbox(
            "Choose data to export:",
            ["Current filtered data", "All data", "Summary statistics", "Vendor analysis"]
        )

        # Select format
        export_format = st.selectbox("Export format:", ["CSV", "Excel"])

        # Prepare data based on selection
        try:
            if export_option == "Current filtered data":
                export_data = filtered_df
            elif export_option == "All data":
                export_data = df
            elif export_option == "Summary statistics":
                export_data = df.describe()
            else:  # Vendor analysis
                export_data = df.groupby('vendor').agg({
                    'product_id': 'count',
                    'price': ['mean', 'min', 'max'],
                    'available': ['sum', 'mean']
                }).round(2)

            # Generate download button
            if not export_data.empty:
                csv_data = export_data.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {export_format}",
                    data=csv_data,
                    file_name=f"ecommerce_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data available for export")

        except Exception as e:
            st.error(f"Error preparing export data: {str(e)}")

    with col2:
        st.markdown("### 📈 Report Generation")

        # Generate summary report
        if st.button("📊 Generate Summary Report"):
            try:
                st.markdown("#### 📋 Executive Summary")

                total_products = len(df)
                total_vendors = df['vendor'].nunique()
                avg_price = df['price'].mean()
                total_stock = df['available'].sum()

                summary_text = f"""
**eCommerce Intelligence Report**
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Key Metrics:**
- Total Products: {total_products:,}
- Unique Vendors: {total_vendors:,}
- Average Price: ${avg_price:.2f}
- Total Stock Units: {total_stock:,}

**Stock Analysis:**
- Out of Stock: {len(df[df['available'] == 0]):,} products
- Low Stock (≤2): {len(df[df['available'] <= 2]):,} products
- Well Stocked (>10): {len(df[df['available'] > 10]):,} products

**Top Product Categories:**
{df['product_type'].value_counts().head().to_string()}

**Top Vendors by Product Count:**
{df['vendor'].value_counts().head().to_string()}
                """

                st.text_area("Report Content:", summary_text, height=400)

                # Download report as text file
                st.download_button(
                    label="📥 Download Report",
                    data=summary_text,
                    file_name=f"ecommerce_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        🛒 eCommerce Intelligence Dashboard | Built with Streamlit & Plotly | 
        Data Pipeline: Shopify → MySQL → Kubeflow → Analytics
    </div>
    """,
    unsafe_allow_html=True
)

# Performance metrics in sidebar
st.sidebar.markdown("## ⚡ Performance")
st.sidebar.info(f"Loaded {len(df):,} products")
st.sidebar.info(f"Filtered to {len(filtered_df):,} products")

print("Dashboard loaded successfully!")
