from typing import List

from src.constants import (
    MIN_COMMENT_LEN,
    MAX_COMMENT_LEN,
    MIN_COMMENT_FILLER,
    MISSING_COMMENT_PLACEHOLDER,
)

# Patch a single comment to fit within [min_len, max_len] 
def _patch_comment(raw_comment: str, min_len: int, max_len: int, filler: str, placeholder: str) -> str:
    
    if not isinstance(raw_comment, str) or not raw_comment.strip():
        return placeholder

    comment = raw_comment.strip()

    if len(comment) < min_len:
        comment += filler
        if len(comment) < min_len:
            comment += " " * (min_len - len(comment))
        return comment

    if len(comment) > max_len:
        trim_idx = comment.rfind(" ", 0, max_len)
        comment = comment[:trim_idx] if trim_idx != -1 else comment[:max_len]
        return comment

    return comment

# Format comments and verdicts in a single pass
def format_results(raw_comments, crisp_verdicts) -> List[str]:
    
    n = len(raw_comments)
    if n == 0:
        return []

    results = []
    for i in range(n):
        verdict = "не бан" if crisp_verdicts[i] else "бан"
        comment = _patch_comment(
            raw_comments[i],
            MIN_COMMENT_LEN,
            MAX_COMMENT_LEN,
            MIN_COMMENT_FILLER,
            MISSING_COMMENT_PLACEHOLDER,
        )
        results.append(f"<комментарий>{comment}<вердикт>{verdict}")

    return results
