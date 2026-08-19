"""Гиперграф семейств товаров: рёбра = SHA картинки / нормализованный текст /
dHash с допуском Хэмминга. Компоненты связности -> группы для GroupKFold
и retrieval-памяти. Меряем размер, чистоту меток, влияние на валидацию."""
import sys, hashlib, time, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("/Users/ashotmirzoyan/Documents/quality-control")
OUT = Path(__file__).parent / "families.csv"

# --- union-find ---
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

def dhash256(path, size=16):
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits(a[:, 1:] > a[:, :-1])  # 32 байта = 256 бит

df = pd.read_csv(ROOT / "data.csv").drop(columns=["Unnamed: 0"], errors="ignore")
n = len(df)
dsu = DSU(n)
t0 = time.time()

# --- ребро 1: нормализованный текст ---
norm = (df["name"].fillna("").astype(str) + "||" + df["description"].fillna("").astype(str))
norm = norm.str.lower().str.replace(r"[^0-9a-zа-яё]+", "", regex=True)
by_text = defaultdict(list)
for i, k in enumerate(norm):
    if k != "||" and k:
        by_text[k].append(i)
text_edges = 0
for ids in by_text.values():
    for j in ids[1:]:
        dsu.union(ids[0], j); text_edges += 1
print(f"текстовых рёбер: {text_edges} ({time.time()-t0:.0f}c)")

# --- ребро 2: SHA-1 главной картинки (побайтовая копия) ---
t1 = time.time()
sha_of = {}
hashes_raw = []
valid_rows = []
for i, pid in enumerate(df["id"]):
    p = ROOT / "images" / str(pid) / "0.jpg"
    if not p.exists():
        continue
    data = p.read_bytes()
    sha_of.setdefault(hashlib.sha1(data).hexdigest(), []).append(i)
    valid_rows.append(i)
sha_edges = 0
for ids in sha_of.values():
    for j in ids[1:]:
        dsu.union(ids[0], j); sha_edges += 1
print(f"SHA-рёбер: {sha_edges} ({time.time()-t1:.0f}c)")

# --- ребро 3: dHash-256 главной картинки, Хэмминг <= 6 ---
t2 = time.time()
hs = np.zeros((len(valid_rows), 32), dtype=np.uint8)
for k, i in enumerate(valid_rows):
    pid = df["id"].iloc[i]
    try:
        hs[k] = dhash256(ROOT / "images" / str(pid) / "0.jpg")
    except Exception:
        pass
print(f"dHash-256 посчитан для {len(valid_rows)} ({time.time()-t2:.0f}c)")

t3 = time.time()
# брутфорс по Хэммингу на numpy: блоками, XOR + аппаратный popcount
# (np.bitwise_count = инструкция POPCNT/CNT, ~5x быстрее lookup-таблицы)
THRESH = 6
pairs = 0
B = 512
for s in range(0, len(hs), B):
    blk = hs[s:s+B]
    # (B, N, 32) слишком жирно -> сравниваем блок со всеми след. строками кусками
    for s2 in range(s, len(hs), B):
        blk2 = hs[s2:s2+B]
        d = np.bitwise_count(blk[:, None, :] ^ blk2[None, :, :]).sum(axis=2, dtype=np.uint16)
        ii, jj = np.where(d <= THRESH)
        for a, b in zip(ii, jj):
            ga, gb = valid_rows[s + a], valid_rows[s2 + b]
            if ga < gb:
                dsu.union(ga, gb); pairs += 1
print(f"dHash-рёбер (Хэмминг<={THRESH}): {pairs} ({time.time()-t3:.0f}c)")

# --- компоненты ---
comp = defaultdict(list)
for i in range(n):
    comp[dsu.find(i)].append(i)
sizes = sorted((len(v) for v in comp.values()), reverse=True)
fams = {len(v) > 1: 0 for v in comp.values()}
multi = [v for v in comp.values() if len(v) > 1]
in_fam = sum(len(v) for v in multi)
print(f"\nкомпонент всего: {len(comp)}; семейств (>=2): {len(multi)}; товаров в семействах: {in_fam} ({in_fam/n:.0%})")
print(f"крупнейшие семейства: {sizes[:8]}")

# чистота меток
mixed = sum(1 for v in multi if df['label'].iloc[v].nunique() > 1)
print(f"семейств со смешанными метками: {mixed} ({mixed/len(multi):.1%})")

# редкий класс
fl = df["category"] == "Легковоспламеняющиеся"
pos_in_fam = sum(1 for v in multi for i in v if fl.iloc[i] and df['label'].iloc[i] == 1)
print(f"позитивов редкого класса внутри семейств: {pos_in_fam} из {int((fl & (df.label==1)).sum())}")

# сохранить group id
group_id = np.zeros(n, dtype=np.int64)
for g, (root, ids) in enumerate(comp.items()):
    for i in ids:
        group_id[i] = g
pd.DataFrame({"id": df["id"], "family": group_id}).to_csv(OUT, index=False)
print(f"\nгруппы сохранены: {OUT}")
print(f"итого {time.time()-t0:.0f}c")
