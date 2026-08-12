"""Grounded deterministic explanations for ultra predictions."""

from __future__ import annotations

import re
from typing import Mapping

from src.ultra_features import BAD_CATEGORY, FLAMMABLE_CATEGORY, safe_text


MIN_COMMENT_LENGTH = 50
MAX_COMMENT_LENGTH = 300
_TAGS_RE = re.compile(r"</?(?:комментарий|вердикт)>", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_VARIANT_RE = re.compile(r"\s*([AB])\s*", re.ASCII)


def _clean_comment(text: object) -> str:
    comment = _TAGS_RE.sub(" ", safe_text(text))
    comment = _SPACE_RE.sub(" ", comment).strip()
    if len(comment) > MAX_COMMENT_LENGTH:
        cut = comment.rfind(" ", 0, MAX_COMMENT_LENGTH + 1)
        comment = comment[: cut if cut >= MIN_COMMENT_LENGTH else MAX_COMMENT_LENGTH].rstrip(" ,;:-")
    if len(comment) < MIN_COMMENT_LENGTH:
        comment = (
            comment.rstrip(" .")
            + ". Решение принято по совокупности доступных данных карточки товара."
        )
    if len(comment) > MAX_COMMENT_LENGTH:
        comment = comment[:MAX_COMMENT_LENGTH].rstrip()
    return comment


def build_comment(
    *,
    category: object,
    prediction: int,
    flags: Mapping[str, bool],
    reason: object,
) -> str:
    category_text = str(category)
    reason_text = str(reason)

    if category_text == BAD_CATEGORY:
        if prediction == 1:
            if flags.get("strict_bad_marker") or flags.get("bad_marker"):
                comment = (
                    "В карточке прямо указана маркировка БАД или биологически активной "
                    "добавки, поэтому товар соответствует правилу категории."
                )
            elif reason_text == "vlm_visual_rescue":
                comment = (
                    "Мультимодальная модель уверенно относит карточку к пищевым добавкам; "
                    "в доступных данных нет надёжного запрещающего основания."
                )
            else:
                comment = (
                    "Совокупность данных карточки относит товар к биологически активным "
                    "добавкам; признака исключённого спортивного питания не найдено."
                )
        else:
            if flags.get("bach_flower"):
                comment = (
                    "Карточка описывает средство из группы цветов Баха, а не товар с "
                    "подтверждённой маркировкой биологически активной добавки."
                )
            elif flags.get("empty_supplement_container"):
                comment = (
                    "Продаётся таблетница, органайзер либо пустая капсула, а не сама "
                    "биологически активная добавка с обязательной маркировкой."
                )
            elif flags.get("not_bad"):
                comment = (
                    "В карточке есть прямое указание, что продукт не является БАД; "
                    "достаточного противоположного подтверждения в данных товара нет."
                )
            elif flags.get("sports"):
                comment = (
                    "Карточка описывает продукт спортивного питания, а явная подтверждённая "
                    "маркировка биологически активной добавки отсутствует."
                )
            else:
                comment = (
                    "В тексте и доступной маркировке не подтверждено явное указание БАД "
                    "или dietary supplement, необходимое по правилу этой категории."
                )
    elif category_text == FLAMMABLE_CATEGORY:
        if prediction == 1:
            if flags.get("fuel") and flags.get("included"):
                comment = (
                    "Карточка указывает, что газ, топливо либо горючий материал входит в "
                    "продаваемый комплект, поэтому товар соответствует категории."
                )
            elif flags.get("standalone") or flags.get("strict_positive_family"):
                comment = (
                    "Товар является самостоятельным средством розжига, пиротехническим "
                    "изделием либо горючим расходным материалом, а не пустым устройством."
                )
            else:
                comment = (
                    "Совокупность данных карточки относит товар к горючему компоненту или "
                    "самостоятельному источнику огня, входящему в комплект."
                )
        else:
            if flags.get("empirical_absence_veto") or flags.get("absent"):
                comment = (
                    "В карточке указано, что газ, топливо или иной горючий расходник не входит "
                    "в комплект и должен приобретаться отдельно."
                )
            elif flags.get("designed_for"):
                comment = (
                    "Продаётся устройство или насадка, предназначенная для отдельного баллона; "
                    "сам газ либо другой горючий материал в комплект не включён."
                )
            elif flags.get("built_in"):
                comment = (
                    "Упомянутый поджиг является встроенной частью устройства, а отдельный "
                    "источник огня или горючий расходник в комплекте не подтверждён."
                )
            else:
                comment = (
                    "Карточка описывает устройство, конструкцию или аксессуар; наличие "
                    "самостоятельного горючего товара в продаваемом комплекте не подтверждено."
                )
    else:
        comment = (
            "Вердикт получен по совокупному анализу данных карточки товара с учётом правил "
            "указанной категории и проверенных модельных сигналов."
        )
    return _clean_comment(comment)


def format_result(comment: str, prediction: int) -> str:
    verdict = "не бан" if int(prediction) == 1 else "бан"
    return f"<комментарий>{_clean_comment(comment)}<вердикт>{verdict}"


def build_comment_variants(
    *,
    category: object,
    prediction: int,
    flags: Mapping[str, bool],
    reason: object,
) -> tuple[str, str]:
    """Return two prevalidated wordings with exactly the same factual claim."""

    direct = build_comment(
        category=category,
        prediction=prediction,
        flags=flags,
        reason=reason,
    )
    context_first = _clean_comment(
        "По результатам проверки карточки: "
        + direct[0].lower()
        + direct[1:]
    )
    return direct, context_first


def choose_comment_variant(candidate: object, variants: tuple[str, str]) -> str:
    """Map a strict LLM A/B choice to safe text; raw generation is never used."""

    choice = safe_text(candidate)
    match = _VARIANT_RE.fullmatch(choice)
    return variants[1] if match and match.group(1) == "B" else variants[0]
