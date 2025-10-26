# check_cache.py
import os, pickle, json, numpy as np, time

BASE = ".bliga_cache"
V = "v1"

def mtime(p): 
    return time.ctime(os.path.getmtime(p)) if os.path.exists(p) else "NO_EXISTE"

npz_path = os.path.join(BASE, f"content_model_{V}.npz")
pkl_path = os.path.join(BASE, f"content_index_{V}.pkl")
npy_path = os.path.join(BASE, f"user_means_{V}.npy")
meta_path = os.path.join(BASE, "meta.json")

print("Archivos:")
for p in [npz_path, pkl_path, npy_path, meta_path]:
    print(f"  {p}  | existe={os.path.exists(p)}  | mtime={mtime(p)}  | size={os.path.getsize(p) if os.path.exists(p) else '-'}")

# Cargar content model desde disco (solo lectura)
assert os.path.exists(npz_path) and os.path.exists(pkl_path), "Faltan archivos del content model"
npz = np.load(npz_path)
with open(pkl_path, "rb") as f:
    idx = pickle.load(f)

item_vecs = npz["item_vecs"]
idf = npz["idf"] if "idf" in npz.files else None
vocab = idx["vocab"]
item_index = idx["item_index"]

print("\nContentModel OK:")
print("  item_vecs shape:", item_vecs.shape)       # [N_items, Vocab]
print("  len(vocab):", len(vocab))
print("  len(item_index):", len(item_index))
assert item_vecs.shape[1] == len(vocab), "Vocab y columnas no coinciden"
if idf is not None and idf.size > 0:
    assert idf.shape[0] == len(vocab), "IDF y vocab no coinciden"

# Cargar user_means si existe
if os.path.exists(npy_path):
    um = np.load(npy_path)
    print("\nuser_means OK: shape", um.shape)
else:
    print("\nuser_means no encontrado (no se calculó/guardó)")

# Mostrar meta si existe
if os.path.exists(meta_path):
    with open(meta_path) as f: 
        meta = json.load(f)
    print("\nmeta.json:", meta)
