import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.transform import Rotation as R
import umap

def extract_features_with_tokens(motion_folder):
    """
    物理的特徴量とユニバーサルトークンを統合して抽出する
    """
    files = {
        'q': 'q.csv',
        'dq': 'dq.csv',
        't_quat': 'torso_quat.csv',
        'tokens': 'token_state.csv',
        'm_name': 'motion_name.csv'
    }
    
    data = {}
    for key, name in files.items():
        path = os.path.join(motion_folder, name)
        if os.path.exists(path):
            try:
                data[key] = pd.read_csv(path)
                if data[key].empty:
                    return None
            except Exception as e:
                print(f"Warning: Failed to read {path}: {e}")
                return None
        else:
            # 必須ファイルがない場合はNoneを返す
            return None

    features = {}
    
    # --- 1. 物理的特徴量 (従来の5項目) ---
    dq_cols = [c for c in data['dq'].columns if 'dq_' in c]
    features['mean_vel'] = data['dq'][dq_cols].abs().mean().mean()
    features['max_vel'] = data['dq'][dq_cols].abs().max().max()

    quats = data['t_quat'][['torso_qw', 'torso_qx', 'torso_qy', 'torso_qz']].values
    quats_xyzw = quats[:, [1, 2, 3, 0]]
    # ノルムが0のデータ（不正な回転）を除外
    norms = np.linalg.norm(quats_xyzw, axis=1)
    valid_mask = norms > 1e-6
    
    if np.any(valid_mask):
        rot = R.from_quat(quats_xyzw[valid_mask])
        euler = rot.as_euler('xyz', degrees=True)
        features['trunk_pitch_avg'] = np.mean(euler[:, 1])
    else:
        # 有効なデータがない場合はデフォルト値を設定
        features['trunk_pitch_avg'] = 0.0

    q_cols = [c for c in data['q'].columns if 'q_' in c]
    features['avg_range'] = (data['q'][q_cols].max() - data['q'][q_cols].min()).mean()
    
    # 胴体の加速度分散（動きの激しさ・震え）
    if os.path.exists(os.path.join(motion_folder, 'torso_accel.csv')):
        acc = pd.read_csv(os.path.join(motion_folder, 'torso_accel.csv'))
        features['accel_var'] = acc[['torso_ax', 'torso_ay', 'torso_az']].var().mean()
    else:
        features['accel_var'] = 0

    # --- 2. ユニバーサルトークンの抽出 (64項目) ---
    token_cols = [c for c in data['tokens'].columns if 'token_' in c]
    token_means = data['tokens'][token_cols].mean()
    for col in token_cols:
        features[col] = token_means[col]

    # モーション名を保存
    m_name = data['m_name']['motion_name'].iloc[0]
    
    return features, m_name

# --- メイン処理 ---
root_dir = "/home/okubo/okubo/GR00T-WholeBodyControl/gear_sonic_deploy/logs/latest/"
all_data = []
motion_names = []

print("Searching for motion data...")
if os.path.exists(root_dir):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            result = extract_features_with_tokens(folder_path)
            if result:
                feat, name = result
                all_data.append(feat)
                motion_names.append(name)

if not all_data:
    print("データが見つかりませんでした。パスとファイル名を確認してください。")
else:
    df = pd.DataFrame(all_data)
    
    # 特徴量のみ（数値データのみ）を抽出
    X = df.drop(columns=[], errors='ignore') 
    
    # --- 標準化 ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 最適なクラスタ数kの探索 ---
    best_k = 0
    best_score = -1
    K_range = range(2, 15)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal k found: {best_k}")

    # --- 最終クラスタリング ---
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    df['motion_name'] = motion_names

    # --- UMAPによる可視化 ---
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X_scaled)

    # --- グラフ描画 ---
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='tab20', s=100, alpha=0.7)
    
    # 注釈を削除（プロットを見やすくするため）

    plt.title(f'Clustering with Universal Tokens (k={best_k})')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("emotion_token_analysis.png", dpi=300)
    plt.show()

    # --- 相関確認の出力 ---
    print("\n=== Feature Correlation with Clusters ===")
    corr_cluster = df.select_dtypes(include=[np.number]).corr()['cluster'].abs().sort_values(ascending=False)
    print(corr_cluster.head(10))

    # --- UMAP軸との相関を確認 ---
    # 2次元のUMAP座標をDataFrame化
    df_umap = pd.DataFrame(X_2d, columns=['UMAP-1', 'UMAP-2'])
    # 数値データのみを抽出してインデックスをリセット
    features_numeric = df.drop(columns=['cluster', 'motion_name'], errors='ignore').select_dtypes(include=[np.number]).reset_index(drop=True)
    df_umap = df_umap.reset_index(drop=True)
    # 結合して相関係数を計算
    umap_corr = pd.concat([features_numeric, df_umap], axis=1).corr()[['UMAP-1', 'UMAP-2']]
    
    print("\n=== Top 10 Features correlating with UMAP-1 ===")
    # 自身(UMAP-1)との相関1.0を除いて表示
    print(umap_corr['UMAP-1'].abs().sort_values(ascending=False).iloc[1:11])
    
    print("\n=== Top 10 Features correlating with UMAP-2 ===")
    # 自身(UMAP-2)との相関1.0を除いて表示
    print(umap_corr['UMAP-2'].abs().sort_values(ascending=False).iloc[1:11])
