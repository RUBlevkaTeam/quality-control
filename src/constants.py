# Common constants for the quality predictor pipeline

# Размерность эмбеддинга Qwen3-VL-Embedding-2B (столько же ждёт классификатор)
EMBEDDING_DIM = 2048

# Presets are specific to Qwen3VL models
PIXEL_PRESETS = {
    "S": 100352,
    "M": 261120,
    "L": 401408,
}

# Явный потолок длины промпта: без него truncation опирается на
# model_max_length, который у Qwen может быть выставлен в огромный сентинел
MAX_PROMPT_TOKENS = 4096

# Default inference batch sizes tuned for H100 80GB
# Если будем запускать на макбуках, то надо будет уменьшить
DEFAULT_EMBED_BATCH_SIZE = 128
DEFAULT_LLM_BATCH_SIZE = 64

# Classifier artifact path (relative to submit root)
_ARTIFACT_REL = "baseline_qwen3vl_bf16.joblib"

# Models directory (matches evaluator's SHARED_MODELS_PATH convention)
SHARED_MODELS_DIR = "SHARED_MODELS_PATH"
DEFAULT_MODELS_DIR = "/shared_models"

# Embedding model path
EMBED_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

# LLM model path
LLM_MODEL_NAME = "Qwen/Qwen3.5-4B"

# Соответствие метки и вердикта.
# label=1 означает "товар относится к категории" -> вердикт "не бан".
# Проверено эмпирически: сборка ecup_quality_ultra с этой же конвенцией
# даёт 0.8 на лидерборде, при инверсии F1 редкого класса был бы около нуля.
VERDICT_FOR_POSITIVE = "не бан"   # pred == 1
VERDICT_FOR_NEGATIVE = "бан"      # pred == 0

MIN_COMMENT_LEN = 50
MAX_COMMENT_LEN = 300
MIN_COMMENT_FILLER = " Вердикт основан на данных карточки товара."
MISSING_COMMENT_PLACEHOLDER = (
    "Вердикт вынесен по названию, описанию и изображениям товара "
    "в соответствии с правилами его категории."
)