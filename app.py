import streamlit as st
import torch
import os
from pathlib import Path
import re
from nltk.tokenize import word_tokenize
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data for Indonesian
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

# Import transformers dengan error handling
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification,
        BertTokenizer,
        BertForTokenClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing transformers: {e}")
    TRANSFORMERS_AVAILABLE = False

@st.cache_resource
def load_indonesian_financial_model(model_path):
    """Load model IndoBERT untuk analisis keuangan"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, "Transformers library tidak tersedia"
    
    try:
        # Cek apakah path model ada
        if not os.path.exists(model_path):
            return None, None, f"Path model tidak ditemukan: {model_path}"
        
        st.info(f"Loading IndoBERT model dari: {model_path}")
        
        # Coba load dengan AutoTokenizer/AutoModel dulu
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            method = "Auto"
        except Exception:
            # Fallback ke BERT specific classes
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForTokenClassification.from_pretrained(model_path)
            method = "BERT"
        
        # Validasi model
        if tokenizer is None or model is None:
            raise ValueError("Model atau tokenizer gagal dimuat")
        
        # Test inference
        test_input = tokenizer("Tes model IndoBERT", return_tensors="pt")
        with torch.no_grad():
            _ = model(**test_input)
        
        return model, tokenizer, f"Success - Method: {method}"
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        return None, None, error_msg

def preprocess_indonesian_text(text):
    """Preprocessing khusus untuk teks Indonesia"""
    # Normalisasi tanda baca Indonesia
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    text = re.sub(r'\.{2,}', '...', text)  # Multiple dots
    
    # Handle currency formatting Indonesia
    text = re.sub(r'Rp\.?\s*(\d+)', r'Rp \1', text)
    text = re.sub(r'(\d+)\s*miliar', r'\1 miliar', text)
    text = re.sub(r'(\d+)\s*triliun', r'\1 triliun', text)
    
    return text.strip()

def merge_bio_entities(tokens, labels, confidences):
    """
    Menggabungkan entitas BIO yang berkesinambungan menjadi satu entitas utuh
    
    Args:
        tokens: List of tokens
        labels: List of BIO labels (B-FIN, I-FIN, etc.)
        confidences: List of confidence scores
    
    Returns:
        List of merged entities
    """
    entities = []
    current_entity = None
    
    for i, (token, label, confidence) in enumerate(zip(tokens, labels, confidences)):
        if label.startswith('B-'):
            # Mulai entitas baru
            if current_entity:
                # Simpan entitas sebelumnya jika ada
                entities.append(current_entity)
            
            # Buat entitas baru
            entity_type = label[2:]  # Hapus 'B-'
            current_entity = {
                'text': token,
                'label': entity_type,
                'start_idx': i,
                'end_idx': i,
                'tokens': [token],
                'confidence': confidence,
                'token_confidences': [confidence]
            }
            
        elif label.startswith('I-') and current_entity:
            # Lanjutkan entitas yang sedang berlangsung
            entity_type = label[2:]  # Hapus 'I-'
            
            # Pastikan tipenya sama dengan entitas yang sedang berlangsung
            if entity_type == current_entity['label']:
                current_entity['text'] += ' ' + token
                current_entity['end_idx'] = i
                current_entity['tokens'].append(token)
                current_entity['token_confidences'].append(confidence)
                # Update confidence dengan rata-rata
                current_entity['confidence'] = sum(current_entity['token_confidences']) / len(current_entity['token_confidences'])
            else:
                # Tipe tidak cocok, simpan entitas lama dan mulai baru
                entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'label': entity_type,
                    'start_idx': i,
                    'end_idx': i,
                    'tokens': [token],
                    'confidence': confidence,
                    'token_confidences': [confidence]
                }
        
        elif label == 'O':
            # Bukan entitas, simpan entitas yang sedang berlangsung jika ada
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Jangan lupa simpan entitas terakhir
    if current_entity:
        entities.append(current_entity)
    
    return entities

def prediksi_entitas_keuangan(teks, model, tokenizer, daftar_label):
    """Prediksi entitas keuangan dari teks Indonesia dengan proper BIO tagging"""
    if tokenizer is None or model is None:
        return [], "Model tidak tersedia"
    
    try:
        # Preprocessing
        teks_bersih = preprocess_indonesian_text(teks)
        
        # Tokenisasi dengan NLTK untuk bahasa Indonesia
        tokens = word_tokenize(teks_bersih, language='english')  # NLTK belum perfect untuk Indonesian
        
        # Encoding
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Prediksi
        with torch.no_grad():
            outputs = model(**{k: v for k, v in encoding.items() if k != 'offset_mapping'})
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)
        
        # Mapping ke entitas dengan BIO tagging
        token_labels = []
        token_confidences = []
        word_ids = encoding.word_ids()
        
        # Mapping predictions ke tokens asli
        for i, (word_id, label_id) in enumerate(zip(word_ids, predicted_labels[0])):
            if word_id is not None and word_id < len(tokens):
                if word_id >= len(token_labels):
                    # Token baru
                    label = daftar_label[label_id.item()] if label_id.item() < len(daftar_label) else "O"
                    confidence = float(torch.max(predictions[0][i]))
                    token_labels.append(label)
                    token_confidences.append(confidence)
                else:
                    # Token yang sudah ada (sub-word), ambil confidence tertinggi
                    existing_confidence = token_confidences[word_id]
                    new_confidence = float(torch.max(predictions[0][i]))
                    if new_confidence > existing_confidence:
                        label = daftar_label[label_id.item()] if label_id.item() < len(daftar_label) else "O"
                        token_labels[word_id] = label
                        token_confidences[word_id] = new_confidence
        
        # Pastikan jumlah labels sama dengan jumlah tokens
        while len(token_labels) < len(tokens):
            token_labels.append("O")
            token_confidences.append(0.0)
        
        # Merge entitas BIO
        merged_entities = merge_bio_entities(tokens, token_labels, token_confidences)
        
        return merged_entities, "Success"
        
    except Exception as e:
        return [], f"Error dalam prediksi: {str(e)}"

def highlight_financial_entities_bio(text, entities):
    """Highlight entitas keuangan dengan BIO tagging yang sudah digabung"""
    if not entities:
        return text
    
    # Tokenize text untuk mendapatkan posisi yang tepat
    tokens = word_tokenize(preprocess_indonesian_text(text))
    
    # Color scheme untuk entitas keuangan
    color_map = {
        'FIN': '#4CAF50',    # Hijau untuk finansial
        'ORG': '#2196F3',    # Biru untuk organisasi/perusahaan
        'PER': '#FF9800',    # Orange untuk person/eksekutif
        'LOC': '#9C27B0',    # Ungu untuk lokasi
        'MISC': '#607D8B'    # Abu-abu untuk miscellaneous
    }
    
    highlighted_text = ""
    i = 0
    
    # Sort entities berdasarkan posisi start
    sorted_entities = sorted(entities, key=lambda x: x['start_idx'])
    entity_idx = 0
    
    while i < len(tokens):
        # Cek apakah token saat ini adalah awal dari entitas
        if entity_idx < len(sorted_entities) and i == sorted_entities[entity_idx]['start_idx']:
            entity = sorted_entities[entity_idx]
            color = color_map.get(entity['label'], '#E0E0E0')
            
            # Highlight seluruh entitas
            highlighted_text += f'''<span style="
                background-color: {color}; 
                color: white; 
                padding: 2px 6px; 
                margin: 1px; 
                border-radius: 4px; 
                font-weight: bold;
                box-shadow: 0 1px 2px rgba(0,0,0,0.2);
            ">{entity['text']} <sub style="font-size: 9px; opacity: 0.8;">{entity['label']} ({entity['confidence']:.2f})</sub></span> '''
            
            # Skip tokens yang sudah di-highlight
            i = entity['end_idx'] + 1
            entity_idx += 1
        else:
            # Token biasa
            highlighted_text += f"{tokens[i]} "
            i += 1
    
    return highlighted_text

def analyze_financial_entities_bio(entities):
    """Analisis entitas keuangan yang sudah digabung dengan BIO tagging"""
    analysis = {
        'companies': [],
        'executives': [],
        'financial_terms': [],
        'locations': [],
        'miscellaneous': [],
        'summary': {}
    }
    
    for entity in entities:
        entity_info = {
            'name': entity['text'],
            'confidence': entity['confidence'],
            'token_count': len(entity['tokens'])
        }
        
        if entity['label'] == 'ORG':
            analysis['companies'].append(entity_info)
        elif entity['label'] == 'PER':
            analysis['executives'].append(entity_info)
        elif entity['label'] == 'FIN':
            analysis['financial_terms'].append(entity_info)
        elif entity['label'] == 'LOC':
            analysis['locations'].append(entity_info)
        elif entity['label'] == 'MISC':
            analysis['miscellaneous'].append(entity_info)
    
    # Summary dengan entitas unik
    analysis['summary'] = {
        'total_companies': len(set([c['name'] for c in analysis['companies']])),
        'total_executives': len(set([e['name'] for e in analysis['executives']])),
        'total_financial_terms': len(set([f['name'] for f in analysis['financial_terms']])),
        'total_locations': len(set([l['name'] for l in analysis['locations']])),
        'total_entities': len(entities),
        'avg_confidence': sum([e['confidence'] for e in entities]) / len(entities) if entities else 0
    }
    
    return analysis

def main():
    st.set_page_config(
        page_title="Analisis Entitas Keuangan Indonesia - BIO Tagging",
        page_icon="💰",
        layout="wide"
    )
    
    # Header dengan styling
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">💰 Ekstraksi Entitas Keuangan Indonesia</h1>
        <p style="color: #e3f2fd; text-align: center; margin: 0.5rem 0 0 0;">
            Analisis artikel keuangan dengan BIO Tagging untuk penggabungan entitas yang sempurna
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download NLTK data
    download_nltk_data()
    
    # Sidebar untuk konfigurasi
    with st.sidebar:
        st.header("⚙️ Konfigurasi Model")
        
        # Path model
        default_path = "model/model_indobert"
        model_path = st.text_input(
            "Path Model IndoBERT:",
            value=default_path,
            help="Path ke model IndoBERT yang sudah dilatih"
        )
        
        # Label entities dengan penjelasan BIO
        st.subheader("🏷️ BIO Tagging Labels")
        st.markdown("""
        **BIO Tagging Format:**
        - **B-**: Beginning (Awal entitas)
        - **I-**: Inside (Lanjutan entitas)  
        - **O**: Outside (Bukan entitas)
        """)
        
        daftar_label = ['O', 'B-FIN', 'I-FIN', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
        
        # Tampilkan label dengan penjelasan
        label_explanations = {
            'O': "🚫 Bukan entitas",
            'B-FIN': "💰 Awal istilah finansial",
            'I-FIN': "💰 Lanjutan istilah finansial",
            'B-ORG': "🏢 Awal nama organisasi",
            'I-ORG': "🏢 Lanjutan nama organisasi", 
            'B-PER': "👤 Awal nama person",
            'I-PER': "👤 Lanjutan nama person"
        }
        
        for label in daftar_label:
            st.write(f"**{label}**: {label_explanations.get(label, '📝 Lainnya')}")
        
        # Load model button
        if st.button("🚀 Load Model IndoBERT", type="primary"):
            if os.path.exists(model_path):
                with st.spinner("Loading model IndoBERT..."):
                    model, tokenizer, status = load_indonesian_financial_model(model_path)
                    
                    if model is not None and tokenizer is not None:
                        st.session_state['model'] = model
                        st.session_state['tokenizer'] = tokenizer
                        st.session_state['labels'] = daftar_label
                        st.session_state['model_path'] = model_path
                        st.success(f"✅ Model berhasil dimuat!")
                        st.info(status)
                    else:
                        st.error(f"❌ {status}")
            else:
                st.error(f"❌ Path model tidak ditemukan: {model_path}")
        
        # Model status
        if 'model' in st.session_state:
            st.success(f"✅ Model aktif: {st.session_state.get('model_path', 'Unknown')}")
        else:
            st.warning("⏳ Model belum dimuat")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input Artikel Keuangan")
        
        # Sample articles untuk testing BIO tagging
        sample_articles = {
            "Pilih contoh artikel...": "",
            "Multi-token Entities": """Bank Central Asia Tbk melaporkan laba bersih sebesar Rp 31.4 triliun pada tahun 2023. Direktur Utama Bank Central Asia Jahja Setiaatmadja menyatakan kinerja positif. PT Telekomunikasi Indonesia Tbk juga mencatat pertumbuhan yang signifikan di Bursa Efek Indonesia.""",
            "Complex Financial Terms": """Kementerian Keuangan Republik Indonesia mengumumkan penerbitan Sukuk Negara Ritel seri SR-017 dengan total target Rp 15 triliun. Menteri Keuangan Sri Mulyani Indrawati menjelaskan instrumen investasi syariah ini untuk membiayai proyek infrastruktur strategis nasional.""",
            "Multiple Person Names": """Presiden Joko Widodo bertemu dengan CEO Apple Inc Tim Cook dan pendiri Microsoft Bill Gates untuk membahas investasi teknologi. Menteri Koordinator Bidang Perekonomian Airlangga Hartarto turut mendampingi dalam pertemuan tersebut."""
        }
        
        selected_sample = st.selectbox("Atau pilih artikel contoh:", list(sample_articles.keys()))
        
        if selected_sample != "Pilih contoh artikel...":
            artikel = st.text_area(
                "Artikel Keuangan:",
                value=sample_articles[selected_sample],
                height=300,
                help="Artikel untuk testing BIO tagging dan penggabungan entitas"
            )
        else:
            artikel = st.text_area(
                "Artikel Keuangan:",
                height=300,
                placeholder="Masukkan artikel keuangan berbahasa Indonesia di sini...",
                help="Masukkan artikel keuangan berbahasa Indonesia"
            )
    
    with col2:
        st.header("📊 Tentang BIO Tagging")
        st.markdown("""
        **BIO Tagging mengatasi masalah:**
        - Entitas multi-kata seperti "Bank Central Asia"
        - Nama panjang seperti "Kementerian Keuangan Republik Indonesia"
        - Istilah finansial kompleks
        
        **Contoh BIO Tagging:**
        ```
        Bank     → B-ORG
        Central  → I-ORG  
        Asia     → I-ORG
        Tbk      → I-ORG
        ```
        
        **Setelah penggabungan:**
        - "Bank Central Asia Tbk" (ORG)
        
        **Fitur Aplikasi:**
        - ✅ Penggabungan entitas otomatis
        - ✅ Confidence scoring rata-rata
        - ✅ Highlighting multi-token entities
        - ✅ Analisis statistik lengkap
        """)
    
    # Analisis artikel
    if artikel.strip() and 'model' in st.session_state:
        st.header("🎯 Hasil Analisis BIO Tagging")
        
        with st.spinner("Menganalisis dengan BIO tagging..."):
            model = st.session_state['model']
            tokenizer = st.session_state['tokenizer']
            daftar_label = st.session_state['labels']
            
            # Prediksi entitas dengan BIO tagging
            entities, status = prediksi_entitas_keuangan(artikel, model, tokenizer, daftar_label)
            
            if status == "Success" and entities:
                # Analisis entitas yang sudah digabung
                analysis = analyze_financial_entities_bio(entities)
                
                # Display hasil dalam tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📄 Teks Highlighted", "📊 Statistik", "📋 Entitas Lengkap", "🔍 Detail BIO"])
                
                with tab1:
                    st.subheader("Artikel dengan Entitas Tergabung")
                    highlighted_text = highlight_financial_entities_bio(artikel, entities)
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                
                with tab2:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "🏢 Perusahaan",
                            analysis['summary']['total_companies'],
                            help="Perusahaan/organisasi unik teridentifikasi"
                        )
                    
                    with col2:
                        st.metric(
                            "👤 Eksekutif",
                            analysis['summary']['total_executives'],
                            help="Tokoh/eksekutif unik teridentifikasi"
                        )
                    
                    with col3:
                        st.metric(
                            "💰 Istilah Finansial",
                            analysis['summary']['total_financial_terms'],
                            help="Istilah finansial unik teridentifikasi"
                        )
                    
                    with col4:
                        st.metric(
                            "🎯 Avg Confidence",
                            f"{analysis['summary']['avg_confidence']:.2f}",
                            help="Rata-rata confidence score"
                        )
                    
                    # Detailed breakdown dalam columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if analysis['companies']:
                            st.subheader("🏢 Perusahaan/Organisasi")
                            unique_companies = {}
                            for company in analysis['companies']:
                                name = company['name']
                                if name not in unique_companies or company['confidence'] > unique_companies[name]['confidence']:
                                    unique_companies[name] = company
                            
                            for name, info in sorted(unique_companies.items(), key=lambda x: x[1]['confidence'], reverse=True):
                                st.write(f"• **{name}** (conf: {info['confidence']:.2f}, tokens: {info['token_count']})")
                        
                        if analysis['financial_terms']:
                            st.subheader("💰 Istilah Finansial")
                            unique_terms = {}
                            for term in analysis['financial_terms']:
                                name = term['name']
                                if name not in unique_terms or term['confidence'] > unique_terms[name]['confidence']:
                                    unique_terms[name] = term
                            
                            for name, info in sorted(unique_terms.items(), key=lambda x: x[1]['confidence'], reverse=True):
                                st.write(f"• **{name}** (conf: {info['confidence']:.2f})")
                    
                    with col2:
                        if analysis['executives']:
                            st.subheader("👤 Tokoh/Eksekutif")
                            unique_executives = {}
                            for exec in analysis['executives']:
                                name = exec['name']
                                if name not in unique_executives or exec['confidence'] > unique_executives[name]['confidence']:
                                    unique_executives[name] = exec
                            
                            for name, info in sorted(unique_executives.items(), key=lambda x: x[1]['confidence'], reverse=True):
                                st.write(f"• **{name}** (conf: {info['confidence']:.2f}, tokens: {info['token_count']})")
                
                with tab3:
                    st.subheader("📋 Semua Entitas Tergabung")
                    
                    # Sort berdasarkan confidence
                    sorted_entities = sorted(entities, key=lambda x: x['confidence'], reverse=True)
                    
                    for i, entity in enumerate(sorted_entities, 1):
                        confidence_color = "🟢" if entity['confidence'] > 0.8 else "🟡" if entity['confidence'] > 0.6 else "🔴"
                        
                        entity_emoji = {
                            'FIN': "💰",
                            'ORG': "🏢", 
                            'PER': "👤",
                            'LOC': "🌍",
                            'MISC': "📝"
                        }.get(entity['label'], "📝")
                        
                        st.write(f"{i}. {confidence_color} {entity_emoji} **{entity['text']}**")
                        st.write(f"   • Label: {entity['label']}")
                        st.write(f"   • Confidence: {entity['confidence']:.3f}")
                        st.write(f"   • Token count: {len(entity['tokens'])}")
                        st.write(f"   • Tokens: {', '.join(entity['tokens'])}")
                        st.write("---")
                
                with tab4:
                    st.subheader("🔍 Detail BIO Tagging")
                    
                    # Tokenize artikel
                    tokens = word_tokenize(preprocess_indonesian_text(artikel))
                    
                    st.write("**Token-level BIO Tags:**")
                    
                    # Buat mapping dari entities ke tokens
                    token_labels = ['O'] * len(tokens)
                    token_confidences = [0.0] * len(tokens)
                    
                    for entity in entities:
                        for i in range(entity['start_idx'], entity['end_idx'] + 1):
                            if i < len(tokens):
                                if i == entity['start_idx']:
                                    token_labels[i] = f"B-{entity['label']}"
                                else:
                                    token_labels[i] = f"I-{entity['label']}"
                                token_confidences[i] = entity['confidence']
                    
                    # Display dalam table format
                    bio_data = []
                    for i, (token, label, conf) in enumerate(zip(tokens, token_labels, token_confidences)):
                        bio_data.append({
                            'Index': i,
                            'Token': token,
                            'BIO Label': label,
                            'Confidence': f"{conf:.3f}" if conf > 0 else "-"
                        })
                    
                    # Show first 50 tokens to avoid overwhelming
                    display_count = min(50, len(bio_data))
                    st.write(f"Menampilkan {display_count} dari {len(bio_data)} tokens:")
                    
                    import pandas as pd
                    df = pd.DataFrame(bio_data[:display_count])
                    st.dataframe(df, use_container_width=True)
                    
                    if len(bio_data) > 50:
                        st.info(f"... dan {len(bio_data) - 50} tokens lainnya")
            
            elif status != "Success":
                st.error(f"❌ Error: {status}")
            else:
                st.info("ℹ️ Tidak ada entitas keuangan yang terdeteksi dalam artikel.")
    
    elif artikel.strip() and 'model' not in st.session_state:
        st.warning("⚠️ Silakan load model IndoBERT terlebih dahulu di sidebar.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🇮🇩 Enhanced Financial NER dengan BIO Tagging | Powered by IndoBERT</p>
        <p><em>Fitur penggabungan entitas otomatis untuk hasil yang lebih akurat</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()