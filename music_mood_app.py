import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
import pickle
from sklearn.metrics import davies_bouldin_score
from datetime import datetime

# [Fungsi kmeans, evaluate_clustering, plot_clusters TETAP SAMA]
def kmeans(data, k, max_iter=100):
    # Inisialisasi centroid secara acak
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iter):
        # Menghitung jarak ke centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign cluster
        labels = np.argmin(distances, axis=0)

        # Update centroid
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Cek konvergensi
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

def evaluate_clustering(true_labels, pred_labels, classes):
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        confusion[classes.index(t), classes.index(p)] += 1

    metrics = {}
    for i, cls in enumerate(classes):
        tp = confusion[i,i]
        fp = confusion[:,i].sum() - tp
        fn = confusion[i,:].sum() - tp
        tn = confusion.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'fdr': fdr,
            'f1': f1
        }

    accuracy = np.trace(confusion) / confusion.sum()
    return confusion, metrics, accuracy


def plot_clusters(data, cluster_labels, centroids, cluster_names, actual_labels):
    # Fungsi untuk membuat plot
    plt.figure(figsize=(15, 6))
    
    # Plot hasil klasterisasi
    plt.subplot(121)
    colors = ['red', 'yellow', 'blue', 'green']
    for i in range(4):
        plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1],
                    c=colors[i], label=cluster_names[i], alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200)
    plt.title('Hasil Klasterisasi')
    plt.xlabel('Valence')
    plt.ylabel('Energy')
    plt.legend()

    # Plot label aktual
    plt.subplot(122)
    classes = ['angry', 'happy', 'sad', 'chill']
    for i, cls in enumerate(classes):
        mask = np.array(actual_labels) == cls
        plt.scatter(data[mask, 0], data[mask, 1], c=colors[i], label=cls, alpha=0.6)
    plt.title('Label Aktual')
    plt.xlabel('Valence')
    plt.ylabel('Energy')
    plt.legend()

    return plt

def main():
    st.title('Music Mood Analysis')
    
    # Inisialisasi session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Clustering"
    
    if 'databases' not in st.session_state:
        st.session_state.databases = {}
    
    # Navigation
    page = st.sidebar.radio(
        "Pilih Menu:",
        ["Clustering", "Find for song mood"],
        index=0 if st.session_state.get('current_page', 'Clustering') == "Clustering" else 1
    )

    # Update current page
    st.session_state.current_page = page
    
    if page == "Clustering":
        clustering_page()
    else:
        search_page()

def clustering_page():
    st.header("Clustering")
    
    uploaded_file = st.file_uploader("Upload file XLSX", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        columns = df.columns.tolist()
        
        # Konfirmasi kolom
        st.subheader("Konfirmasi Kolom")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            track_col = st.selectbox("Track Name", columns, index=columns.index('track_name') if 'track_name' in columns else 0)
        with col2:
            artist_col = st.selectbox("Artist Name", columns, index=columns.index('artist_name') if 'artist_name' in columns else 0)
        with col3:
            valence_col = st.selectbox("Valence", columns, index=columns.index('valence') if 'valence' in columns else 0)
        with col4:
            energy_col = st.selectbox("Energy", columns, index=columns.index('energy') if 'energy' in columns else 0)
        
        if st.button("Proses Clustering"):
            with st.spinner('Sedang memproses clustering...'):
                # Proses clustering
                data = df[[valence_col, energy_col]].values
                centroids, cluster_labels = kmeans(data, k=4)
                
                # Label cluster
                valences = centroids[:, 0]
                energies = centroids[:, 1]
                mean_valence = np.mean(valences)
                mean_energy = np.mean(energies)
                
                cluster_names = []
                for c in centroids:
                    valence, energy = c
                    is_valence_high = valence >= mean_valence
                    is_energy_high = energy >= mean_energy
                    cluster_names.append(
                        'angry' if (is_energy_high and not is_valence_high) else
                        'happy' if (is_energy_high and is_valence_high) else
                        'sad' if (not is_energy_high and not is_valence_high) else
                        'chill'
                    )
                
                # Label aktual
                actual_labels = [
                    'angry' if (e >= 0.5 and v < 0.5) else
                    'happy' if (e >= 0.5 and v >= 0.5) else
                    'sad' if (e < 0.5 and v < 0.5) else
                    'chill'
                    for v, e in data
                ]
                
                # Evaluasi clustering
                classes = ['angry', 'happy', 'sad', 'chill']
                confusion_matrix, metrics, accuracy = evaluate_clustering(
                    actual_labels, 
                    [cluster_names[label] for label in cluster_labels], 
                    classes
                )
                dbi_score = davies_bouldin_score(data, cluster_labels)
                
                # Buat DataFrame hasil
                result_df = df.copy()
                result_df['Actual_Label'] = actual_labels
                result_df['Predicted_Label'] = [cluster_names[label] for label in cluster_labels]
                
                # Pindahkan bagian penyimpanan ke session state
                st.session_state.result_data = {
                    'metadata': {
                        'accuracy': accuracy,
                        'dbi_score': dbi_score,
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'num_songs': len(result_df)
                    },
                    'data': result_df
                }

                # Hapus save_path timestamp dari sini
                
        if 'result_data' in st.session_state:
            # Input nama file dengan nilai default berdasarkan timestamp
            default_name = f"cluster_{st.session_state.result_data['metadata']['timestamp']}"
            save_name = st.text_input(
                "Nama File Hasil Clustering:",
                value=default_name,
                help="Contoh: hasil_clustering_terbaru"
            )
            
            if st.button("Simpan dan Lanjut ke Pencarian"):
                if save_name:
                    # Format nama file
                    clean_name = save_name.strip().replace(" ", "_")
                    if not clean_name.endswith('.pkl'):
                        clean_name += '.pkl'
                    
                    save_dir = "saved_databases"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, clean_name)
                    
                    if os.path.exists(save_path):
                        st.warning("File sudah ada, silakan gunakan nama lain")
                    else:
                        # Simpan file dengan nama custom
                        with open(save_path, "wb") as f:
                            pickle.dump(st.session_state.result_data, f)
                        
                        # Hapus data temporary dan navigasi
                        del st.session_state.result_data
                        st.session_state.current_page = "Find for song mood"
                        st.experimental_rerun()
                else:
                    st.error("Harap beri nama file sebelum menyimpan")

def search_page():
    st.header("Find for song mood")
    
    save_dir = "saved_databases"
    try:
        files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
        
        if not files:
            st.info("Belum ada database clustering. Silakan lakukan clustering terlebih dahulu.")
            return
            
        selected_file = st.selectbox("Pilih Database", files)
        
        with open(os.path.join(save_dir, selected_file), "rb") as f:
            result_data = pickle.load(f)
            
            # Handle struktur data lama dan baru
            if 'data' in result_data:  # Format baru
                df = result_data['data']
                accuracy = result_data['metadata']['accuracy']
            else:  # Format lama
                df = result_data
                accuracy = None
            
            # Tampilkan akurasi jika ada
            if accuracy is not None:
                st.subheader("Hasil Evaluasi Klasterisasi")
                st.metric("Akurasi Keseluruhan", f"{accuracy:.1%}")
            
            # Fitur pencarian
            search_query = st.text_input("Masukkan judul lagu:")
            
            if search_query:
                results = df[df['track_name'].str.contains(
                    search_query, 
                    case=False, 
                    na=False
                )]
                
                if not results.empty:
                    selected_song = st.selectbox("Pilih lagu:", results['track_name'])
                    song_data = results[results['track_name'] == selected_song].iloc[0]
                    
                    st.subheader("Detail Lagu")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Judul Lagu", song_data['track_name'])
                        st.metric("Artis", song_data['artist_name'])
                        st.metric("Mood", song_data['Predicted_Label'])
                    with col2:
                        st.metric("Happiness", f"{int(song_data['valence']*100)}%")
                        st.metric("Energy", f"{int(song_data['energy']*100)}%")
                else:
                    st.error("Lagu tidak ditemukan. Coba kata kunci lain.")
                    
    except Exception as e:
        st.error(f"Error memuat database: {str(e)}")
        st.info("Silakan lakukan ulang proses clustering untuk database ini.")
        
if __name__ == "__main__":
    main()