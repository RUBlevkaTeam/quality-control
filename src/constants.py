# Common constants for the quality predictor pipeline

# Presets are specific to Qwen3VL models
PIXEL_PRESETS = {
    "S": 100352,
    "M": 261120,
    "L": 401408,
}
DEFAULT_PIXEL_PRESET = "S"

# Default inference batch sizes tuned for H100 80GB
DEFAULT_EMBED_BATCH_SIZE = 128
DEFAULT_LLM_BATCH_SIZE = 64

# Conservative defaults for Apple Silicon and CPU development machines.
DEFAULT_LOCAL_EMBED_BATCH_SIZE = 2
DEFAULT_LOCAL_LLM_BATCH_SIZE = 1

# Model paths used by the evaluator
DEFAULT_SHARED_MODELS_PATH = "/shared_models"
EMBED_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
LLM_MODEL_NAME = "Qwen/Qwen3.5-4B"

# Comment formatting constants
MIN_COMMENT_LEN = 50
MAX_COMMENT_LEN = 300
MIN_COMMENT_FILLER = " Решение основано на признаках и описании карточки товара."
MISSING_COMMENT_PLACEHOLDER = (
    "Комментарий не был сгенерирован; вердикт определён классификатором "
    "по данным карточки товара."
)
