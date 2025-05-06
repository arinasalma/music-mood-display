import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
from io import BytesIO

# Fungsi-fungsi yang sama seperti sebelumnya
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

from sklearn.metrics import davies_bouldin_score

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
    
    if 'databases' not in st.session_state:
        st.session_state.databases = {}
    
    option = st.sidebar.radio("Pilih Menu:", ["Clustering", "Find for song mood"])
    
    if option == "Clustering":
        st.header("Clustering")
        uploaded_file = st.file_uploader("Upload file XLSX", type="xlsx")
        
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            columns = df.columns.tolist()
            
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
                    data = df[[valence_col, energy_col]].values
                    
                    # Proses clustering
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
                        if is_energy_high:
                            cluster_names.append('angry' if not is_valence_high else 'happy')
                        else:
                            cluster_names.append('sad' if not is_valence_high else 'chill')
                    
                    # Label aktual
                    actual_labels = []
                    for v, e in data:
                        if e >= 0.5:
                            actual_labels.append('angry' if v < 0.5 else 'happy')
                        else:
                            actual_labels.append('sad' if v < 0.5 else 'chill')
                    
                    # Evaluasi
                    classes = ['angry', 'happy', 'sad', 'chill']
                    confusion_matrix, metrics, accuracy = evaluate_clustering(actual_labels, [cluster_names[label] for label in cluster_labels], classes)
                    dbi_score = davies_bouldin_score(data, cluster_labels)
                    
                    # Simpan hasil ke session state
                    st.session_state.current_result = {
                        'df': df,
                        'centroids': centroids,
                        'cluster_labels': cluster_labels,
                        'cluster_names': cluster_names,
                        'actual_labels': actual_labels,
                        'metrics': metrics,
                        'accuracy': accuracy,
                        'dbi_score': dbi_score,
                        'confusion_matrix': confusion_matrix
                    }
                
                st.success('Clustering selesai!')
                
                # Tampilkan plot
                fig = plot_clusters(data, cluster_labels, centroids, cluster_names, actual_labels)
                st.pyplot(fig)
                
                # Tampilkan metrik
                st.subheader("Hasil Clustering")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Jumlah per Kluster (Prediksi):**")
                    st.write(pd.Series([cluster_names[label] for label in cluster_labels]).value_counts())
                with col2:
                    st.write("**Jumlah per Kluster (Aktual):**")
                    st.write(pd.Series(actual_labels).value_counts())
                
                st.write(f"**Akurasi:** {accuracy:.3f}")
                st.write(f"**DBI Score:** {dbi_score:.3f}")
                
                # Form penyimpanan
                save_name = st.text_input("Nama untuk menyimpan hasil clustering:")
                if st.button("Simpan Hasil"):
                    if save_name:
                        result_df = df.copy()
                        result_df['Actual_Label'] = actual_labels
                        result_df['Predicted_Label'] = [cluster_names[label] for label in cluster_labels]
                        st.session_state.databases[save_name] = result_df
                        st.success(f"Hasil clustering disimpan dengan nama '{save_name}'!")
                    else:
                        st.error("Harap beri nama untuk menyimpan hasil")

    elif option == "Find for song mood":
        st.header("Find for song mood")
        
        if not st.session_state.databases:
            st.warning("Belum ada database yang tersimpan. Lakukan clustering terlebih dahulu.")
            return
            
        selected_db = st.selectbox("Pilih Database", list(st.session_state.databases.keys()))
        df = st.session_state.databases[selected_db]
        
        search_query = st.text_input("Masukkan judul lagu:")
        if search_query:
            results = df[df['track_name'].str.contains(search_query, case=False)]
            if not results.empty:
                selected_song = st.selectbox("Pilih lagu:", results['track_name'])
                song_data = results[results['track_name'] == selected_song].iloc[0]
                
                st.subheader("Hasil Pencarian")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Judul Lagu:** {song_data['track_name']}")
                    st.write(f"**Artis:** {song_data['artist_name']}")
                    st.write(f"**Mood:** {song_data['Predicted_Label']}")
                with col2:
                    st.write(f"**Happiness:** {int(song_data['valence']*100)}%")
                    st.write(f"**Energy:** {int(song_data['energy']*100)}%")
                
                st.write(f"**Akurasi Klasterisasi:** {st.session_state.current_result['accuracy']:.1%}")
            else:
                st.error("Maaf lagu tidak ditemukan, harap masukkan kata kunci lain.")

if __name__ == "__main__":
    main()
