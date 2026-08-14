"""Exact SHA-256 and perceptual dHash retrieval with an optional C++ kernel."""

from __future__ import annotations

import ctypes
import hashlib
import io
import math
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


INDEX_FORMAT_VERSION = 1
DHASH_WORDS = 4
DHASH_BITS = 256
_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_POPCOUNT8 = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint8)


def image_paths_for_id(images_root: str | Path, product_id: object) -> list[Path]:
    directory = Path(images_root) / str(product_id)
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in _VALID_EXTENSIONS
    ]


def _dhash256_from_image(image: Image.Image) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    gray = image.convert("L").resize((17, 16), resampling)
    pixels = np.asarray(gray, dtype=np.uint8)
    packed = np.packbits(pixels[:, 1:] > pixels[:, :-1], bitorder="big")
    return np.ascontiguousarray(packed).view(np.uint64).reshape(DHASH_WORDS).copy()


def fingerprint_file(path: str | Path) -> tuple[bytes, np.ndarray] | None:
    """Read an image once and return exact and perceptual fingerprints."""

    try:
        data = Path(path).read_bytes()
        exact = hashlib.sha256(data).digest()
        with Image.open(io.BytesIO(data)) as image:
            perceptual = _dhash256_from_image(image)
        return exact, perceptual
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def fingerprint_paths(paths: Iterable[str | Path]) -> tuple[tuple[bytes, ...], np.ndarray]:
    exact: list[bytes] = []
    perceptual: list[np.ndarray] = []
    for path in paths:
        result = fingerprint_file(path)
        if result is None:
            continue
        sha256, dhash = result
        exact.append(sha256)
        perceptual.append(dhash)
    exact_unique = tuple(sorted(set(exact)))
    if perceptual:
        hashes = np.unique(np.vstack(perceptual), axis=0)
    else:
        hashes = np.empty((0, DHASH_WORDS), dtype=np.uint64)
    return exact_unique, hashes


def fingerprint_products(
    image_paths: Sequence[Sequence[str | Path]],
    *,
    workers: int = 4,
) -> list[tuple[tuple[bytes, ...], np.ndarray]]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(fingerprint_paths, image_paths))


def _native_library_path() -> Path | None:
    root = Path(__file__).resolve().parents[1] / "native"
    system = platform.system().lower()
    machine = platform.machine().lower()
    aliases = {"amd64": "x86_64", "aarch64": "arm64"}
    machine = aliases.get(machine, machine)
    suffix = ".dylib" if system == "darwin" else ".so" if system == "linux" else ""
    if not suffix:
        return None
    candidate = root / f"libhashmatch_{system}_{machine}{suffix}"
    return candidate if candidate.is_file() else None


class HammingMatcher:
    """Batch top-k Hamming search, native when available and NumPy otherwise."""

    def __init__(self, *, allow_native: bool = True) -> None:
        self._function = None
        self.backend = "numpy"
        path = _native_library_path() if allow_native else None
        if path is None:
            return
        try:
            library = ctypes.CDLL(str(path))
            function = library.hm_topk256
            function.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_uint16),
            ]
            function.restype = ctypes.c_int32
            self._library = library
            self._function = function
            self.backend = "cpp"
        except (AttributeError, OSError):
            self._function = None

    @staticmethod
    def _matrix(values: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.uint64)
        if matrix.size == 0:
            return np.empty((0, DHASH_WORDS), dtype=np.uint64)
        if matrix.ndim != 2 or matrix.shape[1] != DHASH_WORDS:
            raise ValueError(f"dHash matrix must have shape (n, {DHASH_WORDS})")
        return np.ascontiguousarray(matrix)

    def topk(
        self,
        train_hashes: Sequence[Sequence[int]] | np.ndarray,
        query_hashes: Sequence[Sequence[int]] | np.ndarray,
        *,
        max_distance: int = 6,
        top_k: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        train = self._matrix(train_hashes)
        query = self._matrix(query_hashes)
        if not 0 <= max_distance <= DHASH_BITS:
            raise ValueError("max_distance must be in 0..256")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        indices = np.full((len(query), top_k), -1, dtype=np.int32)
        distances = np.full((len(query), top_k), np.iinfo(np.uint16).max, dtype=np.uint16)
        if not len(train) or not len(query):
            return indices, distances

        if self._function is not None:
            status = self._function(
                train.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                len(train),
                query.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                len(query),
                max_distance,
                top_k,
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                distances.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            )
            if status == 0:
                return indices, distances

        # Fail-safe path.  It is intentionally simple and exact; only speed is
        # sacrificed if the evaluator cannot load the native ELF.
        for query_index, value in enumerate(query):
            xor = np.bitwise_xor(train, value)
            byte_view = np.ascontiguousarray(xor).view(np.uint8).reshape(len(train), 32)
            all_distances = _POPCOUNT8[byte_view].sum(axis=1, dtype=np.uint16)
            candidates = np.flatnonzero(all_distances <= max_distance)
            if not len(candidates):
                continue
            order = np.lexsort((candidates, all_distances[candidates]))[:top_k]
            chosen = candidates[order]
            indices[query_index, : len(chosen)] = chosen.astype(np.int32)
            distances[query_index, : len(chosen)] = all_distances[chosen]
        return indices, distances


def _text_digest(value: object) -> bytes:
    from src.rule_features import normalize_text

    normalized = normalize_text(value)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def build_retrieval_index(dataframe, fingerprints) -> dict[str, np.ndarray]:
    if len(dataframe) != len(fingerprints):
        raise ValueError("dataframe and fingerprints must align")
    categories = dataframe["category"].astype(str).tolist()
    category_names = tuple(sorted(set(categories)))
    category_to_code = {name: index for index, name in enumerate(category_names)}

    sha_keys: list[bytes] = []
    sha_products: list[int] = []
    dhashes: list[np.ndarray] = []
    dhash_products: list[int] = []
    for product_index, (exact, perceptual) in enumerate(fingerprints):
        for digest in exact:
            sha_keys.append(digest)
            sha_products.append(product_index)
        for digest in perceptual:
            dhashes.append(digest)
            dhash_products.append(product_index)

    sha_array = np.asarray(sha_keys, dtype="S32")
    sha_product_array = np.asarray(sha_products, dtype=np.int32)
    if len(sha_array):
        order = np.argsort(sha_array, kind="stable")
        sha_array = sha_array[order]
        sha_product_array = sha_product_array[order]

    return {
        "format_version": np.asarray(INDEX_FORMAT_VERSION, dtype=np.int16),
        "category_names": np.asarray(category_names, dtype="U64"),
        "product_labels": dataframe["label"].astype(np.int8).to_numpy(),
        "product_categories": np.asarray(
            [category_to_code[value] for value in categories], dtype=np.int8
        ),
        "product_name_digest": np.asarray(
            [_text_digest(value) for value in dataframe["name"]], dtype="S16"
        ),
        "sha_keys": sha_array,
        "sha_products": sha_product_array,
        "dhashes": (
            np.ascontiguousarray(np.vstack(dhashes), dtype=np.uint64)
            if dhashes
            else np.empty((0, DHASH_WORDS), dtype=np.uint64)
        ),
        "dhash_products": np.asarray(dhash_products, dtype=np.int32),
    }


def save_retrieval_index(path: str | Path, index: Mapping[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **index)


def load_retrieval_index(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        index = {name: archive[name] for name in archive.files}
    if int(index.get("format_version", -1)) != INDEX_FORMAT_VERSION:
        raise ValueError("unsupported image retrieval index")
    required = {
        "category_names",
        "product_labels",
        "product_categories",
        "product_name_digest",
        "sha_keys",
        "sha_products",
        "dhashes",
        "dhash_products",
    }
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"retrieval index misses arrays: {missing}")
    return index


class ImageRetrievalIndex:
    def __init__(self, index: Mapping[str, np.ndarray], *, allow_native: bool = True) -> None:
        self.index = dict(index)
        self.matcher = HammingMatcher(allow_native=allow_native)
        self.category_to_code = {
            str(name): position for position, name in enumerate(self.index["category_names"])
        }
        products = self.index["dhash_products"]
        product_categories = self.index["product_categories"]
        self._dhash_by_category = {
            code: np.flatnonzero(product_categories[products] == code)
            for code in range(len(self.category_to_code))
        }

    def _exact_products(self, digests: Sequence[bytes], category_code: int) -> set[int]:
        keys = self.index["sha_keys"]
        postings = self.index["sha_products"]
        product_categories = self.index["product_categories"]
        found: set[int] = set()
        for digest in set(digests):
            key = np.asarray(digest, dtype="S32")
            left = int(np.searchsorted(keys, key, side="left"))
            right = int(np.searchsorted(keys, key, side="right"))
            for product in postings[left:right]:
                product_index = int(product)
                if int(product_categories[product_index]) == category_code:
                    found.add(product_index)
        return found

    def query(
        self,
        *,
        category: object,
        name: object,
        exact_hashes: Sequence[bytes],
        dhashes: np.ndarray,
        max_distance: int = 6,
        top_k: int = 24,
    ) -> dict[str, object]:
        category_code = self.category_to_code.get(str(category))
        if category_code is None:
            return {"exact_products": (), "perceptual_products": (), "backend": self.matcher.backend}

        exact_products = self._exact_products(exact_hashes, category_code)
        rows = self._dhash_by_category[category_code]
        perceptual_distance: dict[int, int] = {}
        if len(rows) and len(dhashes):
            indices, distances = self.matcher.topk(
                self.index["dhashes"][rows],
                dhashes,
                max_distance=max_distance,
                top_k=top_k,
            )
            products = self.index["dhash_products"]
            for local_index, distance in zip(indices.ravel(), distances.ravel()):
                if local_index < 0:
                    continue
                product = int(products[rows[int(local_index)]])
                value = int(distance)
                perceptual_distance[product] = min(value, perceptual_distance.get(product, 257))

        name_digest = _text_digest(name)
        name_key = np.asarray(name_digest, dtype="S16")
        train_name_digest = self.index["product_name_digest"]
        labels = self.index["product_labels"]

        def records(products: Iterable[int], distance: Mapping[int, int] | None = None):
            return tuple(
                (
                    int(product),
                    int(labels[product]),
                    None if distance is None else int(distance[product]),
                    bool(train_name_digest[product] == name_key),
                )
                for product in sorted(set(products))
            )

        return {
            "exact_products": records(exact_products),
            "perceptual_products": records(perceptual_distance, perceptual_distance),
            "backend": self.matcher.backend,
        }


def _unanimous_label(
    records: Sequence[tuple[int, int, int | None, bool]],
    *,
    require_name: bool = False,
) -> int | None:
    selected = [record for record in records if not require_name or record[3]]
    labels = {record[1] for record in selected}
    return int(next(iter(labels))) if len(labels) == 1 else None


def retrieval_decision(category: object, evidence: Mapping[str, object]) -> tuple[int | None, str]:
    """Conservative asymmetric policy chosen from leave-one-product-out diagnostics."""

    exact_records = evidence.get("exact_products", ())
    perceptual_records = evidence.get("perceptual_products", ())
    exact_label = _unanimous_label(exact_records)
    perceptual_label = _unanimous_label(
        perceptual_records, require_name=True
    )
    # Mixed exact-image labels are a real annotation/product-family conflict.
    # A weaker perceptual source must never break that tie.
    if exact_records and exact_label is None:
        return None, "exact_sha256_conflict"
    # A conflict between independent image signals is more informative than
    # either vote.  Abstaining lets the text model keep control.
    if exact_label is not None and perceptual_label is not None and exact_label != perceptual_label:
        return None, "image_conflict"
    category_text = str(category)
    if exact_label is not None:
        if category_text == "БАД":
            # Five 70/30 deployment simulations showed that an unconditional
            # exact-image override is only marginally useful.  Requiring two
            # independent train products and at least one identical normalized
            # title isolates the stable positive subset (36 right vs 7 wrong
            # flips against the text model in the audit).
            if len(exact_records) >= 2 and any(record[3] for record in exact_records):
                return exact_label, "exact_sha256_multi_name"
            return None, "exact_sha256_observe"
        # Despite 99.7% neighbour accuracy, exact-image overrides hurt the
        # already tuned rare-class text model on 70/30 stress folds.  Keep the
        # evidence observable but do not change that category yet.
        return None, "exact_sha256_observe"

    # On leave-one-product-out diagnostics after removing every exact-SHA
    # match, BAD dHash positives were 39/40 correct while its negatives were
    # weaker.  Flammable name+dHash decisions were 78/78 correct in both
    # directions.  Keep exactly those empirically supported directions.
    if category_text == "БАД" and perceptual_label == 1:
        return 1, "dhash_name_positive"
    if category_text == "Легковоспламеняющиеся" and perceptual_label is not None:
        return perceptual_label, f"dhash_name_{'positive' if perceptual_label else 'negative'}"
    return None, "image_abstain"


def apply_image_retrieval(
    dataframe,
    probabilities: Sequence[float],
    predictions: Sequence[int],
    index: ImageRetrievalIndex,
    *,
    workers: int = 4,
    max_distance: int = 4,
    top_k: int = 128,
) -> dict[str, object]:
    """Fingerprint test images and apply high-precision retrieval decisions."""

    if not (len(dataframe) == len(probabilities) == len(predictions)):
        raise ValueError("retrieval inputs must align")
    fingerprints = fingerprint_products(dataframe["image_paths"].tolist(), workers=workers)
    output_probabilities = np.asarray(probabilities, dtype=np.float64).copy()
    output_predictions = np.asarray(predictions, dtype=np.int8).copy()
    reasons = np.asarray(["text_model"] * len(dataframe), dtype=object)
    evidence_rows = []
    for position, (_, row) in enumerate(dataframe.iterrows()):
        exact, perceptual = fingerprints[position]
        evidence = index.query(
            category=row.get("category", ""),
            name=row.get("name", ""),
            exact_hashes=exact,
            dhashes=perceptual,
            max_distance=max_distance,
            top_k=top_k,
        )
        decision, reason = retrieval_decision(row.get("category", ""), evidence)
        evidence_rows.append(evidence)
        if decision is None:
            continue
        output_predictions[position] = decision
        output_probabilities[position] = 1.0 - 1e-5 if decision else 1e-5
        reasons[position] = reason
    return {
        "probabilities": output_probabilities,
        "predictions": output_predictions,
        "reasons": reasons,
        "evidence": evidence_rows,
        "fingerprints": fingerprints,
        "backend": index.matcher.backend,
    }
