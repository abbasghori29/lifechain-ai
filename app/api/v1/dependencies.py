"""
FastAPI Dependencies for Translation
"""

from typing import Optional
from fastapi import Query
from app.services.translation import translate_response

async def get_translation_language(
    lang: Optional[str] = Query(
        "en",
        description="Language code for translation (en=English, ur=Urdu, ar=Arabic, hi=Hindi, bn=Bengali)"
    )
) -> str:
    """
    Get translation language from query parameter
    
    Args:
        lang: Language code (default: "en" for no translation)
    
    Returns:
        Language code
    """
    valid_languages = ["en", "ur", "ar", "hi", "bn"]
    if lang not in valid_languages:
        return "en"  # Default to English if invalid
    return lang

from typing import Any

async def apply_translation(
    data: Any,
    model_type: str,
    language: str
) -> Any:
    """
    Apply translation to response data if language is not English
    
    Args:
        data: Response data
        model_type: Type of model (e.g., "visit", "symptom", "diagnosis")
        language: Target language code
    
    Returns:
        Translated data if language is not "en", otherwise original data
    """
    if language == "en" or not data:
        return data
    
    return await translate_response(data, model_type, language)

