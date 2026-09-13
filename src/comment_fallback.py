"""Валидатор LLM-комментариев и детерминированный фолбэк.

Формулировки перенесены из ecup_quality_ultra/src/ultra_explain.py: на каждую
комбинацию (категория x вердикт x флаги правил) - готовое объяснение в терминах
политики площадки. Фолбэк по построению не может противоречить вердикту,
потому что собирается из тех же флагов, из которых складывается решение.

Валидатор ловит брак LLM с высокой точностью и низкой полнотой (лучше
пропустить средний комментарий, чем заменить хороший): пустоту, слова
«бан»/«не бан» в тексте и прямые логические инверсии.
"""

from __future__ import annotations

import re
import sys
from typing import Mapping

MIN_LEN = 50
MAX_LEN = 300

_BAN_WORD_RE = re.compile(r"(?<![а-яa-zё])не\s+бан|(?<![а-яa-zё])бан(?![а-яa-zё])", re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"не\s+относится|не\s+подходит\s+под\s+категори", re.IGNORECASE)
_POSITIVE_RE = re.compile(
    r"отнесён\s+к\s+категории\s+верно|действительно\s+относится|"
    r"соответствует\s+(правилу\s+)?категори", re.IGNORECASE)


def _log(message: str) -> None:
    print(f"[fallback] {message}", file=sys.stderr, flush=True)


def comment_is_valid(comment: object, prediction: int) -> bool:
    text = " ".join(str(comment or "").split())
    if len(text) < MIN_LEN:
        return False
    if _BAN_WORD_RE.search(text):
        return False
    if prediction == 1 and _NEGATIVE_RE.search(text):
        return False
    if prediction == 0 and _POSITIVE_RE.search(text):
        return False
    return True


def _fit_length(comment: str) -> str:
    comment = " ".join(comment.split()).strip()
    if len(comment) > MAX_LEN:
        cut = comment.rfind(" ", 0, MAX_LEN + 1)
        comment = comment[: cut if cut >= MIN_LEN else MAX_LEN].rstrip(" ,;:-")
    if len(comment) < MIN_LEN:
        comment = (comment.rstrip(" .")
                   + ". Решение принято по совокупности данных карточки товара.")
    return comment[:MAX_LEN].rstrip()


def build_fallback_comment(category: object, prediction: int,
                           flags: Mapping[str, bool]) -> str:
    category_text = str(category)
    if category_text == "БАД":
        if prediction == 1:
            if flags.get("strict_bad_marker") or flags.get("bad_marker"):
                comment = ("В карточке прямо указана маркировка БАД или биологически "
                           "активной добавки, поэтому товар соответствует правилу категории.")
            else:
                comment = ("Совокупность данных карточки относит товар к биологически "
                           "активным добавкам; признака исключённого спортивного питания "
                           "не найдено.")
        else:
            if flags.get("bach_flower"):
                comment = ("Карточка описывает средство из группы цветов Баха, а не товар "
                           "с подтверждённой маркировкой биологически активной добавки.")
            elif flags.get("empty_supplement_container"):
                comment = ("Продаётся таблетница, органайзер либо пустая капсула, а не сама "
                           "биологически активная добавка с обязательной маркировкой.")
            elif flags.get("not_bad"):
                comment = ("В карточке есть прямое указание, что продукт не является БАД; "
                           "достаточного противоположного подтверждения в данных товара нет.")
            elif flags.get("sports"):
                comment = ("Карточка описывает продукт спортивного питания, а явная "
                           "подтверждённая маркировка биологически активной добавки "
                           "отсутствует.")
            else:
                comment = ("В тексте и доступной маркировке не подтверждено явное указание "
                           "БАД или dietary supplement, необходимое по правилу этой категории.")
    elif category_text == "Легковоспламеняющиеся":
        if prediction == 1:
            if flags.get("fuel") and flags.get("included"):
                comment = ("Карточка указывает, что газ, топливо либо горючий материал "
                           "входит в продаваемый комплект, поэтому товар соответствует "
                           "категории.")
            elif flags.get("standalone") or flags.get("strict_positive_family"):
                comment = ("Товар является самостоятельным средством розжига, "
                           "пиротехническим изделием либо горючим расходным материалом, "
                           "а не пустым устройством.")
            else:
                comment = ("Совокупность данных карточки относит товар к горючему "
                           "компоненту или самостоятельному источнику огня, входящему "
                           "в комплект.")
        else:
            if flags.get("empirical_absence_veto") or flags.get("absent"):
                comment = ("В карточке указано, что газ, топливо или иной горючий расходник "
                           "не входит в комплект и должен приобретаться отдельно.")
            elif flags.get("designed_for"):
                comment = ("Продаётся устройство или насадка, предназначенная для отдельного "
                           "баллона; сам газ либо другой горючий материал в комплект не включён.")
            elif flags.get("built_in"):
                comment = ("Упомянутый поджиг является встроенной частью устройства, а "
                           "отдельный источник огня или горючий расходник в комплекте "
                           "не подтверждён.")
            else:
                comment = ("Карточка описывает устройство, конструкцию или аксессуар; наличие "
                           "самостоятельного горючего товара в продаваемом комплекте "
                           "не подтверждено.")
    else:
        comment = ("Вердикт получен по совокупному анализу данных карточки товара с учётом "
                   "правил указанной категории и проверенных модельных сигналов.")
    return _fit_length(comment)


def repair_comments(comments, dataframe) -> list:
    """LLM-комментарии с заменой брака на детерминированные. Любая ошибка
    деградирует до фолбэка, никогда не роняет прогон."""
    try:
        from src.rule_features import evidence_flags
    except Exception:
        evidence_flags = None

    repaired, replaced = [], 0
    rows = list(dataframe.iterrows())
    for position, (_, row) in enumerate(rows):
        raw = comments[position] if position < len(comments) else ""
        prediction = int(row.get("pred", 0) or 0)
        if comment_is_valid(raw, prediction):
            repaired.append(str(raw))
            continue
        try:
            flags = evidence_flags(row.get("name"), row.get("description")) if evidence_flags else {}
        except Exception:
            flags = {}
        repaired.append(build_fallback_comment(row.get("category"), prediction, flags))
        replaced += 1
    if replaced:
        _log(f"заменено детерминированным фолбэком: {replaced} из {len(rows)}")
    return repaired
