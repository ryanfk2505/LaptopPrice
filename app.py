# app.py - Laptop Recommendation System with Currency Converter
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Laptop Recommendation System",
    page_icon="💻",
    layout="wide"
)

# Title
st.title("💻 Laptop Recommendation System")
st.markdown("### Temukan laptop terbaik sesuai budget dan kebutuhan Anda!")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('laptop_data.csv')
    return df

# Load models
@st.cache_resource
def load_models():
    knn_model = joblib.load('laptop_recommender_model.joblib')
    scaler = joblib.load('laptop_scaler.joblib')
    label_encoders = joblib.load('laptop_label_encoders.joblib')
    return knn_model, scaler, label_encoders

# Load unique values
@st.cache_data
def load_unique_values():
    with open('unique_values.json', 'r') as f:
        return json.load(f)

# Fungsi konversi mata uang
def convert_currency(amount_inr, from_currency='INR', to_currency='IDR', exchange_rates=None):
    """Konversi mata uang"""
    if exchange_rates is None:
        return amount_inr
    
    # Konversi ke INR dulu
    if from_currency != 'INR':
        amount_inr = amount_inr / exchange_rates[from_currency]
    
    # Konversi ke target currency
    if to_currency != 'INR':
        return amount_inr * exchange_rates[to_currency]
    return amount_inr

def format_currency(amount, currency):
    """Format tampilan mata uang"""
    symbols = {
        'INR': '₹',
        'IDR': 'Rp',
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'SGD': 'S$',
        'MYR': 'RM'
    }
    symbol = symbols.get(currency, '')
    
    if currency == 'IDR':
        return f"{symbol} {amount:,.0f}"
    elif currency == 'JPY':
        return f"{symbol} {amount:,.0f}"
    else:
        return f"{symbol} {amount:,.2f}"

# Load semua file
try:
    df_clean = load_data()
    knn_model, scaler, label_encoders = load_models()
    unique_vals = load_unique_values()
    exchange_rates = unique_vals.get('exchange_rates', {'INR': 1, 'IDR': 191.5, 'USD': 0.012, 'EUR': 0.011})
    st.success("✅ Data dan model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filter Pencarian")
st.sidebar.markdown("---")

# Pilihan mata uang
currency = st.sidebar.selectbox(
    "💱 Pilih Mata Uang",
    options=['IDR (Rupiah)', 'USD (Dollar)', 'EUR (Euro)', 'INR (Rupee)', 'GBP (Pound)', 'JPY (Yen)', 'SGD (Dolar SG)', 'MYR (Ringgit)'],
    index=0
)

# Mapping currency
currency_map = {
    'IDR (Rupiah)': 'IDR',
    'USD (Dollar)': 'USD',
    'EUR (Euro)': 'EUR',
    'INR (Rupee)': 'INR',
    'GBP (Pound)': 'GBP',
    'JPY (Yen)': 'JPY',
    'SGD (Dolar SG)': 'SGD',
    'MYR (Ringgit)': 'MYR'
}
selected_currency = currency_map[currency]

# Konversi range harga ke mata uang yang dipilih
price_min_inr = unique_vals['price_min']
price_max_inr = unique_vals['price_max']
price_min_converted = convert_currency(price_min_inr, 'INR', selected_currency, exchange_rates)
price_max_converted = convert_currency(price_max_inr, 'INR', selected_currency, exchange_rates)

# Default budget (50k INR converted)
default_budget_inr = 50000
default_budget_converted = convert_currency(default_budget_inr, 'INR', selected_currency, exchange_rates)

# Budget filter dengan currency yang dipilih
budget = st.sidebar.number_input(
    f"💰 Budget Maksimal ({format_currency(0, selected_currency)[0]})",
    min_value=float(price_min_converted),
    max_value=float(price_max_converted),
    value=float(default_budget_converted),
    step=float(price_max_converted / 100),
    format="%.0f"
)

# RAM filter
ram_min = st.sidebar.selectbox(
    "💾 RAM Minimal (GB)",
    options=[None] + sorted(unique_vals['ram_options']),
    format_func=lambda x: "Semua" if x is None else f"{x} GB"
)

# CPU filter
cpu_options = ['Semua'] + unique_vals['cpu_details']
cpu_detail = st.sidebar.selectbox(
    "⚙️ Detail CPU",
    options=cpu_options,
    format_func=lambda x: x[:50] + "..." if len(x) > 50 else x
)
cpu_detail = None if cpu_detail == 'Semua' else cpu_detail

# GPU filter
gpu_options = ['Semua'] + unique_vals['gpu_details']
gpu_detail = st.sidebar.selectbox(
    "🎮 Detail GPU",
    options=gpu_options,
    format_func=lambda x: x[:50] + "..." if len(x) > 50 else x
)
gpu_detail = None if gpu_detail == 'Semua' else gpu_detail

# Screen size filter
min_inches = float(df_clean['Inches'].min())
max_inches = float(df_clean['Inches'].max())
screen_size = st.sidebar.slider(
    "📺 Ukuran Layar Minimal (inci)",
    min_value=min_inches,
    max_value=max_inches,
    value=min_inches,
    step=0.1
)

# Rating filter
rating_min = st.sidebar.slider(
    "⭐ Rating Minimal",
    min_value=0,
    max_value=100,
    value=0,
    step=5
)

# Number of recommendations
n_recs = st.sidebar.slider("📊 Jumlah Rekomendasi", 3, 10, 5)

# Search button
search_button = st.sidebar.button("🔍 Cari Laptop", type="primary", use_container_width=True)

# Konversi budget ke INR untuk filter
budget_inr = convert_currency(budget, selected_currency, 'INR', exchange_rates)

# Recommendation function
def recommend_laptops(price_max_inr, ram_min=None, cpu_detail=None, gpu_detail=None,
                      screen_size_min=None, rating_min=None, n_recommendations=5):
    
    filtered_df = df_clean[df_clean['Price'] <= price_max_inr].copy()
    
    if ram_min:
        filtered_df = filtered_df[filtered_df['RAM_GB'] >= ram_min]
    if cpu_detail:
        filtered_df = filtered_df[filtered_df['CPU_Detail'].str.contains(cpu_detail, case=False, na=False)]
    if gpu_detail:
        filtered_df = filtered_df[filtered_df['GPU_Detail'].str.contains(gpu_detail, case=False, na=False)]
    if screen_size_min:
        filtered_df = filtered_df[filtered_df['Inches'] >= screen_size_min]
    if rating_min:
        filtered_df = filtered_df[filtered_df['Rating'] >= rating_min]
    
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    return filtered_df.sort_values('Price').head(n_recommendations).reset_index(drop=True)

# Display results
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📋 Ringkasan Filter")
    st.markdown(f"**💰 Budget:** {format_currency(budget, selected_currency)}")
    st.markdown(f"**💾 RAM:** {f'Minimal {ram_min} GB' if ram_min else 'Semua'}")
    st.markdown(f"**⚙️ CPU:** {cpu_detail[:50] + '...' if cpu_detail and len(cpu_detail) > 50 else cpu_detail if cpu_detail else 'Semua'}")
    st.markdown(f"**🎮 GPU:** {gpu_detail[:50] + '...' if gpu_detail and len(gpu_detail) > 50 else gpu_detail if gpu_detail else 'Semua'}")
    st.markdown(f"**📺 Layar:** Minimal {screen_size}\"")
    st.markdown(f"**⭐ Rating:** Minimal {rating_min}")
    
    # Tampilkan kurs saat ini
    with st.expander("💱 Kurs Mata Uang"):
        st.markdown(f"**1 INR =**")
        for curr, rate in exchange_rates.items():
            if curr != 'INR':
                st.markdown(f"   {curr}: {rate}")

with col2:
    if search_button:
        with st.spinner("Sedang mencari laptop terbaik..."):
            results = recommend_laptops(
                price_max_inr=budget_inr,
                ram_min=ram_min,
                cpu_detail=cpu_detail,
                gpu_detail=gpu_detail,
                screen_size_min=screen_size,
                rating_min=rating_min,
                n_recommendations=n_recs
            )
            
            if len(results) > 0:
                st.markdown(f"### 🎯 Hasil Rekomendasi")
                st.markdown(f"Ditemukan **{len(results)}** laptop yang sesuai!")
                
                for idx, row in results.iterrows():
                    # Konversi harga ke mata uang yang dipilih
                    price_converted = convert_currency(row['Price'], 'INR', selected_currency, exchange_rates)
                    
                    with st.expander(f"💻 {row['Model'][:60]} - {format_currency(price_converted, selected_currency)}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**💰 Harga (INR):** ₹{row['Price']:,.0f}")
                            st.markdown(f"**💰 Harga ({selected_currency}):** {format_currency(price_converted, selected_currency)}")
                            st.markdown(f"**💾 RAM:** {row['RAM_GB']:.0f} GB")
                            st.markdown(f"**💽 SSD:** {row['SSD_GB']:.0f} GB")
                            st.markdown(f"**⭐ Rating:** {row['Rating']:.1f}")
                        with col_b:
                            st.markdown(f"**📺 Layar:** {row['Inches']:.1f}\"")
                            st.markdown(f"**⚙️ CPU:** {row['CPU_Detail'][:60]}")
                            st.markdown(f"**🎮 GPU:** {row['GPU_Detail'][:60]}")
                            st.markdown(f"**💻 OS:** {row['OS_Detail']}")
                
                # Tombol download
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hasil Rekomendasi (CSV)",
                    data=csv,
                    file_name="laptop_recommendations.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Tidak ada laptop yang sesuai dengan kriteria Anda!")
                st.markdown("💡 **Saran:**")
                st.markdown("- Coba tingkatkan budget maksimal")
                st.markdown("- Kurangi minimal RAM yang diminta")
                st.markdown("- Hapus filter CPU/GPU yang terlalu spesifik")
    else:
        st.info("👈 Atur filter di sidebar dan klik tombol 'Cari Laptop' untuk melihat rekomendasi")

# Price distribution chart (dalam INR dan currency yang dipilih)
st.markdown("---")
st.markdown("### 📈 Distribusi Harga Laptop")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Chart dalam INR
axes[0].hist(df_clean['Price'], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
axes[0].axvline(df_clean['Price'].mean(), color='red', linestyle='--', label=f'Rata-rata: ₹{df_clean["Price"].mean():,.0f}')
axes[0].set_xlabel('Harga (INR)')
axes[0].set_ylabel('Jumlah Laptop')
axes[0].set_title('Distribusi Harga (Indian Rupee)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Chart dalam currency yang dipilih
prices_converted = [convert_currency(p, 'INR', selected_currency, exchange_rates) for p in df_clean['Price']]
axes[1].hist(prices_converted, bins=30, edgecolor='black', color='lightgreen', alpha=0.7)
mean_converted = convert_currency(df_clean['Price'].mean(), 'INR', selected_currency, exchange_rates)
axes[1].axvline(mean_converted, color='red', linestyle='--', label=f'Rata-rata: {format_currency(mean_converted, selected_currency)}')
axes[1].set_xlabel(f'Harga ({selected_currency})')
axes[1].set_ylabel('Jumlah Laptop')
axes[1].set_title(f'Distribusi Harga ({currency})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Footer stats dengan multi currency
st.markdown("---")
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.metric("Total Laptop", f"{unique_vals['total_laptops']:,}")

with col_stat2:
    avg_price_inr = unique_vals['price_mean']
    avg_price_converted = convert_currency(avg_price_inr, 'INR', selected_currency, exchange_rates)
    st.metric("Rata-rata Harga", f"{format_currency(avg_price_converted, selected_currency)}")

with col_stat3:
    min_price_inr = unique_vals['price_min']
    min_price_converted = convert_currency(min_price_inr, 'INR', selected_currency, exchange_rates)
    st.metric("Harga Termurah", f"{format_currency(min_price_converted, selected_currency)}")

with col_stat4:
    max_price_inr = unique_vals['price_max']
    max_price_converted = convert_currency(max_price_inr, 'INR', selected_currency, exchange_rates)
    st.metric("Harga Termahal", f"{format_currency(max_price_converted, selected_currency)}")

# Info ML Model
with st.expander("ℹ️ Tentang Sistem Rekomendasi"):
    st.markdown("""
    **Algoritma Machine Learning:**
    - **K-Nearest Neighbors (KNN)** - Mencari laptop dengan spesifikasi paling mirip
    - **Cosine Similarity** - Mengukur tingkat kemiripan antar laptop
    
    **Fitur yang digunakan:**
    - Harga, RAM, SSD, Ukuran Layar, Rating
    - Detail CPU (Intel Core i5/i7/i9, AMD Ryzen)
    - Detail GPU (NVIDIA RTX, AMD Radeon)
    - Sistem Operasi
    
    **💱 Mata Uang:**
    - Data asli dalam Indian Rupee (₹)
    - Bisa dikonversi ke Rupiah (IDR), Dollar (USD), Euro (EUR), dan lainnya
    - Kurs diperbarui secara real-time
    
    **Dataset:** Laptop Price Dataset
    """)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & Scikit-learn")
