# app.py - Laptop Recommendation System
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

# Load semua file
try:
    df_clean = load_data()
    knn_model, scaler, label_encoders = load_models()
    unique_vals = load_unique_values()
    st.success("✅ Data dan model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filter Pencarian")
st.sidebar.markdown("---")

# Budget filter
budget = st.sidebar.slider(
    "💰 Budget Maksimal (₹)",
    min_value=unique_vals['price_min'],
    max_value=unique_vals['price_max'],
    value=min(50000, unique_vals['price_max']),
    step=5000,
    format="₹%d"
)

# RAM filter
ram_min = st.sidebar.selectbox(
    "💾 RAM Minimal (GB)",
    options=[None] + sorted(unique_vals['ram_options']),
    format_func=lambda x: "Semua" if x is None else f"{x} GB"
)

# CPU filter (searchable dengan dropdown)
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

# Recommendation function
def recommend_laptops(price_max, ram_min=None, cpu_detail=None, gpu_detail=None,
                      screen_size_min=None, rating_min=None, n_recommendations=5):
    
    filtered_df = df_clean[df_clean['Price'] <= price_max].copy()
    
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
    st.markdown(f"**💰 Budget:** ₹{budget:,}")
    st.markdown(f"**💾 RAM:** {f'Minimal {ram_min} GB' if ram_min else 'Semua'}")
    st.markdown(f"**⚙️ CPU:** {cpu_detail[:50] + '...' if cpu_detail and len(cpu_detail) > 50 else cpu_detail if cpu_detail else 'Semua'}")
    st.markdown(f"**🎮 GPU:** {gpu_detail[:50] + '...' if gpu_detail and len(gpu_detail) > 50 else gpu_detail if gpu_detail else 'Semua'}")
    st.markdown(f"**📺 Layar:** Minimal {screen_size}\"")
    st.markdown(f"**⭐ Rating:** Minimal {rating_min}")

with col2:
    if search_button:
        with st.spinner("Sedang mencari laptop terbaik..."):
            results = recommend_laptops(
                price_max=budget,
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
                    with st.expander(f"💻 {row['Model'][:60]} - ₹{row['Price']:,.0f}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**💰 Harga:** ₹{row['Price']:,.0f}")
                            st.markdown(f"**💾 RAM:** {row['RAM_GB']:.0f} GB")
                            st.markdown(f"**💽 SSD:** {row['SSD_GB']:.0f} GB")
                            st.markdown(f"**⭐ Rating:** {row['Rating']:.1f}")
                        with col_b:
                            st.markdown(f"**📺 Layar:** {row['Inches']:.1f}\"")
                            st.markdown(f"**⚙️ CPU:** {row['CPU_Detail'][:60]}")
                            st.markdown(f"**🎮 GPU:** {row['GPU_Detail'][:60]}")
                            st.markdown(f"**💻 OS:** {row['OS_Detail']}")
                
                # Tombol export hasil
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

# Price distribution chart
st.markdown("---")
st.markdown("### 📈 Distribusi Harga Laptop")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_clean['Price'], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
ax.axvline(df_clean['Price'].mean(), color='red', linestyle='--', label=f'Rata-rata: ₹{df_clean["Price"].mean():,.0f}')
ax.axvline(df_clean['Price'].median(), color='green', linestyle='--', label=f'Median: ₹{df_clean["Price"].median():,.0f}')
ax.set_xlabel('Harga (₹)')
ax.set_ylabel('Jumlah Laptop')
ax.set_title('Distribusi Harga Laptop')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Footer stats
st.markdown("---")
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.metric("Total Laptop", f"{unique_vals['total_laptops']:,}")

with col_stat2:
    st.metric("Rata-rata Harga", f"₹{unique_vals['price_mean']:,}")

with col_stat3:
    st.metric("Harga Termurah", f"₹{unique_vals['price_min']:,}")

with col_stat4:
    st.metric("Harga Termahal", f"₹{unique_vals['price_max']:,}")

# Info ML Model
with st.expander("ℹ️ Tentang Sistem Rekomendasi"):
    st.markdown("""
    **Algoritma Machine Learning yang digunakan:**
    - **K-Nearest Neighbors (KNN)** - Mencari laptop dengan spesifikasi paling mirip
    - **Cosine Similarity** - Mengukur tingkat kemiripan antar laptop
    
    **Fitur yang digunakan untuk rekomendasi:**
    - Harga, RAM, SSD, Ukuran Layar, Rating
    - Detail CPU (Intel Core i5/i7/i9, AMD Ryzen)
    - Detail GPU (NVIDIA RTX, AMD Radeon)
    - Sistem Operasi
    
    **Dataset:** Laptop Price Dataset (₹ dalam Indian Rupees)
    """)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & Scikit-learn")
