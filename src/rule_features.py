"""Детерминированные rule-признаки (перенесены из ultra, где дали 0.8 на LB).

38 лексических признаков: маркировка БАД и отрицания, спортпит, пиротехника,
топливо/устройства, комплектность. Обобщаются на новые бренды - в отличие от
TF-IDF-лексики, на которой редкий класс переобучился (OOF 0.871 vs LB 0.687).
Регулярки не менять без замера: они выверены на лидерборде.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from typing import Iterable, Sequence

import numpy as np


BAD_CATEGORY = "БАД"
FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"


_HTML_TAG_RE = re.compile(r"<[^>]*>")
_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^0-9a-zа-я]+", re.IGNORECASE)


def safe_text(value: object) -> str:
    """Return a stable string for CSV values, treating NaN/None as empty."""

    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def normalize_text(value: object) -> str:
    """Normalize noisy marketplace text without erasing model/SKU numbers."""

    text = html.unescape(safe_text(value))
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = _HTML_TAG_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalized_name(name: object) -> str:
    return normalize_text(name)


def normalized_full_text(name: object, description: object) -> str:
    return f"{normalize_text(name)}\x1f{normalize_text(description)}"


def build_model_text(name: object, description: object) -> str:
    """Build field-aware input, intentionally giving the title extra weight."""

    title = normalize_text(name)
    body = normalize_text(description)
    return f"название {title} название {title} описание {body}"


def build_model_texts(names: Iterable[object], descriptions: Iterable[object]) -> list[str]:
    return [build_model_text(name, description) for name, description in zip(names, descriptions)]


# The rule bank is intentionally compact.  It adds relational signals that a
# bag-of-ngrams can underweight, while leaving the final decision to a trained
# linear model and category-specific threshold.
_BAD_MARKER_RE = re.compile(
    r"(?:\bбад\b|биологическ\w*\s+активн\w*\s+добавк\w*|dietary\s+supplement|food\s+supplement)"
)
_BAD_NEGATION_RE = re.compile(
    r"(?:не\s+явля\w*\s+(?:биологическ\w*\s+активн\w*\s+добавк\w*|бад)|"
    r"не\s+(?:является\s+)?бад\b|not\s+(?:a\s+)?dietary\s+supplement)"
)
_STRICT_BAD_MARKER_RE = re.compile(
    r"биологическ\w*\s+активн\w*\s+добавк\w*\s+к\s+пище"
)
_BACH_FLOWER_RE = re.compile(r"(?:цвет\w*\s+баха|bach\s+flower)")
_EMPTY_SUPPLEMENT_CONTAINER_RE = re.compile(
    r"(?:таблетниц\w*|органайзер\w*\s+для\s+лекарств\w*|пуст\w*\s+капсул\w*)"
)
_SPORTS_RE = re.compile(
    r"(?:спортивн\w*\s+питан\w*|\bпротеин\w*|\bгейнер\w*|\bbcaa\b|"
    r"аминокислот\w*|л[ -]?карнитин\w*|\bcarnitine\b|\bcreatine\b|\bкреатин\w*|"
    r"предтрен\w*|изотоник\w*|сывороточн\w*\s+белок)"
)

_FLAME_STANDALONE_RE = re.compile(
    r"(?:\bспичк\w*|\bзажигалк\w*|\bхлопушк\w*|\bсалют\w*|\bфейерверк\w*|"
    r"бенгальск\w*\s+(?:огн\w*|свеч\w*)|дым\w*\s+шашк\w*|цветн\w*\s+дым\w*|"
    r"свеч\w*[- ]?фонтан\w*|пиротехн\w*)"
)
_FLAME_FUEL_RE = re.compile(
    r"(?:\bбутан\w*|\bпропан\w*|\bгаз\w*\s+баллон\w*|баллон\w*\s+с\s+газ\w*|"
    r"газ\w*\s+для\s+(?:зажигал\w*|плит\w*)|топлив\w*\s+для\s+зажигал\w*|"
    r"сух\w*\s+горюч\w*|\bрастопк\w*|ролл\w*\s+для\s+розжиг\w*|"
    r"угол\w*\s+(?:древесн\w*|каменн\w*)|брикет\w*\s+для\s+(?:грил\w*|розжиг\w*))"
)
_FLAME_DEVICE_RE = re.compile(
    r"(?:\bмангал\w*|\bгрил\w*|\bпеч\w*|\bкамин\w*|\bплит\w*|"
    r"\bгорелк\w*|\bтаган\w*|\bразогревател\w*|\bрозжиг\w*)"
)
_INCLUDED_RE = re.compile(
    r"(?:в\s+комплект\w*|комплект\w*\s+с|в\s+набор\w*|набор\w*\s+с|"
    r"вход\w*\s+в\s+комплект|поставля\w*\s+с|вместе\s+с|укомплектован\w*)"
)
_ABSENT_RE = re.compile(
    r"(?:без\s+(?:газ\w*|баллон\w*|топлив\w*|угл\w*|спич\w*|брик\w*|растопк\w*)|"
    r"не\s+(?:вход\w*|включен\w*|постав\w*)|приобрета\w*\s+отдельно|"
    r"баллон\w*\s+отдельно|(?:газ\w*|баллон\w*|топлив\w*|угол\w*|растопк\w*)\s+"
    r"(?:не\s+вход\w*|не\s+прилага\w*|приобрета\w*\s+отдельно)|"
    r"не\s+комплекту\w*\s+(?:газ\w*|баллон\w*|топлив\w*))"
)
_REFILLABLE_RE = re.compile(r"(?:перезаправ\w*|заправляем\w*|газ\s+заправляется|резервуар\w*)")
_ABSENCE_VETO_RE = re.compile(
    r"(?:без\s+(?:газ|баллон|топлив|угл|спич)|не\s+(?:входит|включен|постав)|"
    r"приобрет\w*\s+отдельно|баллон\w*\s+отдельно)"
)
_DESIGNED_FOR_RE = re.compile(
    r"(?:(?:насадк\w*|переходник\w*)\s+(?:на|для)\s+баллон\w*|"
    r"для\s+(?:газ\w*\s+)?баллон\w*|совместим\w*\s+с\s+баллон\w*)"
)
_BUILT_IN_RE = re.compile(
    r"(?:пьезо(?:электрическ\w*)?(?:поджиг\w*)?|встроенн\w*\s+(?:поджиг\w*|зажиг\w*))"
)
_COMPONENT_ONLY_RE = re.compile(
    r"(?:огнеупорн\w*|огнестойк\w*|термостойк\w*|негорюч\w*|"
    r"чехол\w*|сумк\w*|опахал\w*|аксессуар\w*|запчаст\w*)"
)
_STRICT_FLAME_POSITIVE_RE = re.compile(
    r"(?:гост\s+1820\s+2001|дымова\w*\s+шашк\w*|шашк\w*\s+дымов\w*|"
    r"pyrofx|a2tech|бдш\s+тип|joker\s+fireworks|maxsem\s+smoking|"
    r"газ\s+для\s+заправк\w*\s+зажигал\w*|газ\s+для\s+портативн\w*\s+плит\w*|"
    r"свеч\w*\s+фонтан\w*|фонтан\w*\s+(?:в|для)\s+торт\w*|фонтаны\s+в\s+торт\w*)"
)


def _strict_flame_positive_family(name_text: str, all_text: str) -> bool:
    """Return the shared high-precision flame-family flag.

    Keeping this helper shared prevents the learned rule feature, the VLM gate
    and the human-readable explanation from silently using different rule
    banks.
    """

    return bool(
        _STRICT_FLAME_POSITIVE_RE.search(all_text)
        or re.search(r"хлопушк\w*.*артикул\s+тр", name_text)
        or (
            "мангал одноразов" in name_text
            and re.search(r"(?:комплект\w*\s+с|с)\s+угл", name_text)
        )
        or (
            re.search(r"интерактивн\w*\s+открытк", name_text)
            and "свеч" in all_text
            and "спич" in all_text
        )
    )


RULE_FEATURE_NAMES: tuple[str, ...] = (
    "marker_any",
    "marker_name",
    "marker_description",
    "explicit_not_bad",
    "sports_any",
    "sports_name",
    "marker_x_not_bad",
    "marker_x_sports",
    "strict_bad_marker",
    "strict_bad_marker_no_sports",
    "bach_flower",
    "empty_supplement_container",
    "standalone_ignition",
    "fuel_or_material",
    "device",
    "included",
    "content_absent",
    "refillable",
    "absence_non_refill",
    "empirical_absence_veto",
    "designed_for",
    "built_in_ignition",
    "component_or_accessory",
    "fuel_x_included",
    "device_x_included",
    "device_x_absent",
    "device_x_designed_for",
    "standalone_x_absent",
    "name_standalone",
    "name_fuel",
    "name_device",
    "name_included",
    "name_absent",
    "description_included",
    "description_absent",
    "strict_positive_family",
    "log_name_length",
    "log_description_length",
)


def _hit(pattern: re.Pattern[str], text: str) -> float:
    return float(pattern.search(text) is not None)


def rule_feature_vector(name: object, description: object, category: object) -> np.ndarray:
    """Return a fixed, category-agnostic vector of lexical rule evidence."""

    name_text = normalize_text(name)
    desc_text = normalize_text(description)
    all_text = f"{name_text} {desc_text}".strip()

    marker = _hit(_BAD_MARKER_RE, all_text)
    marker_name = _hit(_BAD_MARKER_RE, name_text)
    marker_desc = _hit(_BAD_MARKER_RE, desc_text)
    not_bad = _hit(_BAD_NEGATION_RE, all_text)
    sports = _hit(_SPORTS_RE, all_text)
    sports_name = _hit(_SPORTS_RE, name_text)
    strict_bad_marker = _hit(_STRICT_BAD_MARKER_RE, all_text)
    bach_flower = _hit(_BACH_FLOWER_RE, all_text)
    empty_container = _hit(_EMPTY_SUPPLEMENT_CONTAINER_RE, all_text)

    standalone = _hit(_FLAME_STANDALONE_RE, all_text)
    fuel = _hit(_FLAME_FUEL_RE, all_text)
    device = _hit(_FLAME_DEVICE_RE, all_text)
    included = _hit(_INCLUDED_RE, all_text)
    absent = _hit(_ABSENT_RE, all_text)
    refillable = _hit(_REFILLABLE_RE, all_text)
    empirical_absence_veto = _hit(_ABSENCE_VETO_RE, all_text) * (1.0 - refillable)
    designed_for = _hit(_DESIGNED_FOR_RE, all_text)
    built_in = _hit(_BUILT_IN_RE, all_text)
    component = _hit(_COMPONENT_ONLY_RE, all_text)
    strict_positive = float(_strict_flame_positive_family(name_text, all_text))

    values = (
        marker,
        marker_name,
        marker_desc,
        not_bad,
        sports,
        sports_name,
        marker * not_bad,
        marker * sports,
        strict_bad_marker,
        strict_bad_marker * (1.0 - sports),
        bach_flower,
        empty_container,
        standalone,
        fuel,
        device,
        included,
        absent,
        refillable,
        absent * (1.0 - refillable),
        empirical_absence_veto,
        designed_for,
        built_in,
        component,
        fuel * included,
        device * included,
        device * absent,
        device * designed_for,
        standalone * absent,
        _hit(_FLAME_STANDALONE_RE, name_text),
        _hit(_FLAME_FUEL_RE, name_text),
        _hit(_FLAME_DEVICE_RE, name_text),
        _hit(_INCLUDED_RE, name_text),
        _hit(_ABSENT_RE, name_text),
        _hit(_INCLUDED_RE, desc_text),
        _hit(_ABSENT_RE, desc_text),
        strict_positive,
        math.log1p(len(name_text)),
        math.log1p(len(desc_text)),
    )
    return np.asarray(values, dtype=np.float32)


def rule_feature_matrix(
    names: Sequence[object],
    descriptions: Sequence[object],
    categories: Sequence[object],
) -> np.ndarray:
    if not (len(names) == len(descriptions) == len(categories)):
        raise ValueError("names, descriptions and categories must have equal length")
    if not names:
        return np.empty((0, len(RULE_FEATURE_NAMES)), dtype=np.float32)
    return np.vstack(
        [
            rule_feature_vector(name, description, category)
            for name, description, category in zip(names, descriptions, categories)
        ]
    )


def evidence_flags(name: object, description: object) -> dict[str, bool]:
    """Human-readable evidence switches used by deterministic explanations."""

    name_text = normalize_text(name)
    desc_text = normalize_text(description)
    text = f"{name_text} {desc_text}".strip()
    return {
        "bad_marker": bool(_BAD_MARKER_RE.search(text)),
        "not_bad": bool(_BAD_NEGATION_RE.search(text)),
        "sports": bool(_SPORTS_RE.search(text)),
        "strict_bad_marker": bool(_STRICT_BAD_MARKER_RE.search(text)),
        "strict_bad_marker_no_sports": bool(
            _STRICT_BAD_MARKER_RE.search(text) and not _SPORTS_RE.search(text)
        ),
        "bach_flower": bool(_BACH_FLOWER_RE.search(text)),
        "empty_supplement_container": bool(_EMPTY_SUPPLEMENT_CONTAINER_RE.search(text)),
        "standalone": bool(_FLAME_STANDALONE_RE.search(text)),
        "fuel": bool(_FLAME_FUEL_RE.search(text)),
        "device": bool(_FLAME_DEVICE_RE.search(text)),
        "included": bool(_INCLUDED_RE.search(text)),
        "absent": bool(_ABSENT_RE.search(text)),
        "refillable": bool(_REFILLABLE_RE.search(text)),
        "absence_non_refill": bool(_ABSENT_RE.search(text) and not _REFILLABLE_RE.search(text)),
        "empirical_absence_veto": bool(
            _ABSENCE_VETO_RE.search(text) and not _REFILLABLE_RE.search(text)
        ),
        "strict_positive_family": _strict_flame_positive_family(name_text, text),
        "designed_for": bool(_DESIGNED_FOR_RE.search(text)),
        "built_in": bool(_BUILT_IN_RE.search(text)),
    }
