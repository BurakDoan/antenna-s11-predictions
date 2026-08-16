import streamlit as st
import requests
import pandas as pd

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Mikroşerit Anten S11 Tahmin Portalı",
    page_icon="📡",
    layout="wide"
)

# Özel CSS Tasarımı
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1rem; color: #475569; margin-bottom: 1.5rem; }
    .stMetric { background-color: #F8FAFC; padding: 15px; border-radius: 10px; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# Başlık Alanı
st.markdown('<div class="main-header">📡 Mikroşerit Anten S₁₁ Parametresi Tahmin Portalı</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Makine Öğrenmesi Tabanlı Elektromanyetik Performans ve Empedans Uyumu Analizi</div>', unsafe_allow_html=True)

# Yan Panel - Geometrik Parametreler
st.sidebar.header("📐 Anten Tasarım Parametreleri")
st.sidebar.markdown("---")

patch_length = st.sidebar.number_input(
    "Yama Uzunluğu (l) [mm]",
    value=28.61, step=0.1, format="%.2f",
    help="Patch Length: Antenin çalışma frekansını (rezonansını) belirleyen temel boyut."
)
patch_width = st.sidebar.number_input(
    "Yama Genişliği (w) [mm]",
    value=33.63, step=0.1, format="%.2f",
    help="Patch Width: Empedans uyumunu ve ışıma direncini doğrudan etkiler."
)
ground_length = st.sidebar.number_input(
    "Toprak Düzlemi Uzunluğu (lg) [mm]",
    value=57.23, step=0.1, format="%.2f",
    help="Ground Length: Alt iletken tabakanın toplam uzunluğu."
)
ground_width = st.sidebar.number_input(
    "Toprak Düzlemi Genişliği (wg) [mm]",
    value=75.85, step=0.1, format="%.2f",
    help="Ground Width: Alt iletken tabakanın toplam genişliği."
)
gap = st.sidebar.number_input(
    "Besleme Yarığı Mesafesi (g) [mm]",
    value=0.25, step=0.01, format="%.2f",
    help="Gap: Mikroşerit besleme hattı ile yama arasındaki boşluk."
)

# Sekmeli Ana Ekran Mimarisi
tab1, tab2, tab3 = st.tabs(["🚀 Tahmin & Analiz", "📚 Proje Metodolojisi & S₁₁ Fiziksel Temeli", "🏗️ MLOps Sistem Mimarisi"])

with tab1:
    st.subheader("📊 Model Çıkarım Paneli")
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("##### 🔍 Girdi Parametre Özeti")
        param_df = pd.DataFrame({
            "Parametre": ["Yama Uzunluğu (l)", "Yama Genişliği (w)", "Toprak Uzunluğu (lg)", "Toprak Genişliği (wg)", "Yarık Boyutu (g)"],
            "Değer": [f"{patch_length} mm", f"{patch_width} mm", f"{ground_length} mm", f"{ground_width} mm", f"{gap} mm"]
        })
        st.dataframe(param_df, hide_index=True, use_container_width=True)
        
        predict_btn = st.button("⚡ S₁₁ Değerini Hesapla", type="primary", use_container_width=True)

    with col_result:
        st.markdown("##### 📈 Hesaplanan Performans")
        if predict_btn:
            payload = {
                "patch_length": patch_length,
                "patch_width": patch_width,
                "ground_length": ground_length,
                "ground_width": ground_width,
                "gap": gap
            }
            
            try:
                res = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
                if res.status_code == 200:
                    s11_val = res.json()["S11_dB_prediction"]
                    
                    st.metric(
                        label="Geri Dönüş Kaybı (S₁₁)",
                        value=f"{s11_val:.4f} dB",
                        delta="Yüksek Verim / Rezonans" if s11_val <= -10 else "Zayıf Empedans Uyumu",
                        delta_color="normal" if s11_val <= -10 else "inverse"
                    )

                    if s11_val <= -10:
                        st.success(f"✅ **Mükemmel Rezonans:** S₁₁ = {s11_val:.2f} dB (≤ -10 dB). Verilen gücün %90'ından fazlası antenden başarıyla ışıtmaktadır.")
                    else:
                        st.warning(f"⚠️ **Yüksek Yansıma Kaybı:** S₁₁ = {s11_val:.2f} dB (> -10 dB). Anten bu geometrik boyutlarda yeterli rezonansa girememektedir.")
                else:
                    st.error("API Bağlantı Hatası! FastAPI servisini kontrol edin.")
            except Exception as e:
                st.error(f"Tahmin servisine bağlanılamadı: {e}")
        else:
            st.info("Hesaplama yapmak için sol taraftaki boyutları ayarlayıp **'S₁₁ Değerini Hesapla'** butonuna tıklayın.")

with tab2:
    st.subheader("📖 Proje Hakkında & Sıkça Sorulan Sorular")
    
    st.markdown("""
    ### ❓ "Neden Frekans Değeri Girilmeden Tahmin Yapılabiliyor?"
    Mikroşerit anten teorisinde rezonans frekansı ($f_r$), doğrudan **yama uzunluğu ($l$)** ve kullanılan **taban malzemesinin (dielektrik katsayısı $\\varepsilon_r$ ve kalınlık $h$)** bir fonksiyonudur:

    $$f_r \\approx \\frac{c}{2l \\sqrt{\\varepsilon_{eff}}}$$

    * **Sabit Substrat Yapısı:** Bu proje kapsamında eğitilen Makine Öğrenmesi modeli, belirli bir katman kalınlığı ve sabit dielektrik malzemesi ($\text{FR-4}$) üzerindeki simülasyon verileriyle eğitilmiştir.
    * **Geometriden Doğrudan Tahmin:** Yama boyutları ($l, w$) değiştikçe antenin rezonansa girdiği frekans otomatik olarak kayar. Modelimiz, geometrik boyutların kombinasyonunu girdi alarak o geometrinin sunduğu **en ideal $S_{11}$ empedans uyum değerini** doğrudan tahmin etmektedir.

    ---

    ### 📡 S₁₁ (Geri Dönüş Kaybı / Return Loss) Nedir?
    * **$S_{11} \\le -10\\text{ dB}$:** Anten ile besleme hattı arasındaki empedans uyumu mükemmeldir. Porttan verilen gücün **%90'ından fazlası** antenden yayılır.
    * **$S_{11} > -10\\text{ dB}$:** Gücün büyük kısmı geriye yansır, anten verimsiz çalışır.
    """)

with tab3:
    st.subheader("🏗️ Endüstriyel MLOps Mimari Detayları")
    st.markdown("""
    Bu sistem, makine öğrenmesi modellerini üretim ortamına (production) taşımak için modern **MLOps** mimarisine uygun olarak tasarlanmıştır:

    1. **Veri Hatları & İşleme (`src/data_pipeline.py`):** Ham elektromanyetik simülasyon verileri temizlenir, $VIF$ ve korelasyon analizleriyle öznitelikler ölçeklendirilir.
    2. **Deney Takibi & MLflow (`src/train.py`):** Random Forest ve XGBoost algoritmalarının hiperparametreleri MLflow üzerinde takip edilerek en düşük RMSE değerine sahip model kayıt altına alınır.
    3. **REST API Servisi (`main.py`):** FastAPI framework'ü ile oluşturulan yüksek performanslı çıkarım (inference) servisi.
    4. **Docker Containerization:** FastAPI ve Streamlit servisleri Docker ve Docker-Compose ile izole konteyner yapılarına dönüştürülmüştür.
    """)