"""
Translation Service for Medical Terms
Uses LangChain Groq to translate medical terminology for doctors who don't speak English
"""

import json
from typing import Dict, Any, Optional, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.core.config import get_settings

# Initialize LLM and parser (singleton pattern)
_llm: Optional[ChatGroq] = None
_parser: Optional[JsonOutputParser] = None

def get_translation_llm() -> ChatGroq:
    """Get or create the Groq LLM instance"""
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = ChatGroq(
            model="groq/compound",  # Using versatile model for better translation
            temperature=0,
            groq_api_key=settings.GROQ_API_KEY
        )
    return _llm

def get_translation_parser() -> JsonOutputParser:
    """Get or create the JSON parser instance"""
    global _parser
    if _parser is None:
        _parser = JsonOutputParser()
    return _parser

# Translation schema for medical terms
TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translated_text": {
            "type": "string",
            "description": "The translated text in the target language"
        },
        "original_text": {
            "type": "string",
            "description": "The original English text"
        },
        "language": {
            "type": "string",
            "description": "The target language code (e.g., 'ur' for Urdu, 'ar' for Arabic)"
        }
    },
    "required": ["translated_text", "original_text", "language"]
}

async def translate_text(
    text: str,
    target_language: str = "ur",
    context: str = "medical"
) -> str:
    """
    Translate medical text to target language
    
    Args:
        text: English text to translate
        target_language: Target language code (default: "ur" for Urdu)
        context: Context of translation (default: "medical")
    
    Returns:
        Translated text
    """
    if not text or not text.strip():
        return text
    
    try:
        llm = get_translation_llm()
        parser = get_translation_parser()
        
        language_names = {
            "ur": "Urdu",
            "ar": "Arabic",
            "hi": "Hindi",
            "bn": "Bengali"
        }
        language_name = language_names.get(target_language, target_language)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a Medical Translation Expert. Translate the English medical text to {language_name}.

CRITICAL: Return ONLY valid JSON. No explanations, no reasoning, no markdown code blocks, no extra text.

Required JSON format:
{{
    "translated_text": "translated text in {language_name}",
    "original_text": "original English text",
    "language": "{target_language}"
}}

Rules:
- Maintain medical accuracy
- Use proper medical terminology in {language_name}
- Preserve numbers, measurements, codes
- Keep format and structure
- Return ONLY the JSON object, nothing else
            """),
            ("human", "Translate this medical text to {language_name}:\n\n{text}\n\nReturn ONLY the JSON object with no additional text.")
        ])
        
        chain = prompt | llm | parser
        
        result = await chain.ainvoke({
            "text": text,
            "language_name": language_name,
            "target_language": target_language,
            "context": context
        })
        
        # Extract translated text from result
        if isinstance(result, dict):
            translated = result.get("translated_text", text)
        else:
            # Fallback: try to extract JSON from string if parser failed
            translated = _extract_json_from_text(str(result), text)
        
        return translated
    
    except Exception as e:
        # If translation fails, return original text
        print(f"Translation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return text

def translate_dict_fields(
    data: Dict[str, Any],
    fields_to_translate: List[str],
    target_language: str = "ur"
) -> Dict[str, Any]:
    """
    Translate specific fields in a dictionary
    
    Args:
        data: Dictionary containing data to translate
        fields_to_translate: List of field names to translate
        target_language: Target language code
    
    Returns:
        Dictionary with translated fields
    """
    if not data or not isinstance(data, dict):
        return data
    
    translated_data = data.copy()
    
    for field in fields_to_translate:
        if field in translated_data and translated_data[field]:
            if isinstance(translated_data[field], str):
                # For async context, we'll need to handle this differently
                # For now, return the field as-is and translate in the endpoint
                translated_data[field] = translated_data[field]
            elif isinstance(translated_data[field], list):
                # Handle list of items
                translated_data[field] = [
                    translate_dict_fields(item, fields_to_translate, target_language)
                    if isinstance(item, dict) else item
                    for item in translated_data[field]
                ]
            elif isinstance(translated_data[field], dict):
                # Recursively translate nested dictionaries
                translated_data[field] = translate_dict_fields(
                    translated_data[field],
                    fields_to_translate,
                    target_language
                )
    
    return translated_data

def _extract_json_from_text(text: str, fallback: str) -> str:
    """
    Try to extract JSON from text that may contain explanations.
    Falls back to original text if JSON cannot be extracted.
    """
    import re
    import json
    
    # Try to find JSON object in the text
    json_pattern = r'\{[^{}]*"translated_text"[^{}]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        # Try to parse the last match (usually the final answer)
        for match in reversed(matches):
            try:
                # Clean up the match
                match = match.strip()
                # Remove markdown code blocks if present
                match = re.sub(r'```json\s*', '', match)
                match = re.sub(r'```\s*', '', match)
                parsed = json.loads(match)
                if "translated_text" in parsed:
                    return parsed["translated_text"]
            except:
                continue
    
    # If no JSON found, return fallback
    return fallback

def _model_to_dict(model_instance: Any) -> Optional[Dict[str, Any]]:
    """
    Convert a model instance (SQLAlchemy or Pydantic) to a dictionary.
    Handles both Pydantic v1 and v2, and SQLAlchemy models.
    """
    if model_instance is None:
        return None
    
    if isinstance(model_instance, dict):
        return model_instance
    
    # Try Pydantic v2 method first
    if hasattr(model_instance, "model_dump"):
        return model_instance.model_dump()
    
    # Try Pydantic v1 method
    if hasattr(model_instance, "dict"):
        return model_instance.dict()
    
    # Try SQLAlchemy model - convert to dict manually
    if hasattr(model_instance, "__table__"):
        result = {}
        for column in model_instance.__table__.columns:
            value = getattr(model_instance, column.name, None)
            # Handle enum values
            if hasattr(value, 'value'):
                result[column.name] = value.value
            else:
                result[column.name] = value
        return result
    
    # If it's a simple object with __dict__, use that
    if hasattr(model_instance, "__dict__"):
        return model_instance.__dict__
    
    return None

# Fields that need translation in different models
# NOTE: Enum fields (like visit_type, report_type) should NOT be translated
# as they must match database enum values and schema patterns
TRANSLATION_FIELDS = {
    "visit": ["chief_complaint", "doctor_notes"],  # Removed visit_type - it's an enum
    "symptom": ["symptom_name", "notes"],
    "diagnosis": ["disease_name", "notes"],
    "prescription": ["medication_name", "instructions"],
    "lab_report": ["test_name"],  # Removed report_type - it's an enum
    "lab_test_result": ["test_name"],
    "patient": ["disease_name"],  # For disease history
    "family_disease_history": ["disease_name", "notes"]
}

async def translate_response(
    data: Any,
    model_type: str,
    target_language: str = "ur"
) -> Any:
    """
    Translate response data based on model type
    
    Args:
        data: Response data (dict, list, or model instance)
        model_type: Type of model (e.g., "visit", "symptom", "diagnosis")
        target_language: Target language code
    
    Returns:
        Translated data
    """
    if data is None:
        return data
    
    fields_to_translate = TRANSLATION_FIELDS.get(model_type, [])
    
    if not fields_to_translate:
        return data
    
    if isinstance(data, list):
        # Translate each item in the list
        translated_items = []
        for item in data:
            if isinstance(item, dict):
                translated_item = await translate_dict_async(
                    item,
                    fields_to_translate,
                    target_language
                )
                translated_items.append(translated_item)
            else:
                # If it's a model instance (SQLAlchemy or Pydantic), convert to dict first
                item_dict = _model_to_dict(item)
                if item_dict:
                    translated_dict = await translate_dict_async(
                        item_dict,
                        fields_to_translate,
                        target_language
                    )
                    translated_items.append(translated_dict)
                else:
                    translated_items.append(item)
        return translated_items
    
    elif isinstance(data, dict):
        return await translate_dict_async(data, fields_to_translate, target_language)
    
    else:
        # Convert model instance to dict, translate, and return
        data_dict = _model_to_dict(data)
        if data_dict:
            translated_dict = await translate_dict_async(
                data_dict,
                fields_to_translate,
                target_language
            )
            return translated_dict
        return data

async def translate_dict_async(
    data: Dict[str, Any],
    fields_to_translate: List[str],
    target_language: str = "ur"
) -> Dict[str, Any]:
    """
    Asynchronously translate specific fields in a dictionary
    
    Args:
        data: Dictionary containing data to translate
        fields_to_translate: List of field names to translate
        target_language: Target language code
    
    Returns:
        Dictionary with translated fields
    """
    if not data or not isinstance(data, dict):
        return data
    
    translated_data = data.copy()
    
    for field in fields_to_translate:
        if field in translated_data and translated_data[field] is not None:
            field_value = translated_data[field]
            
            # Handle enum values - convert to string first
            if hasattr(field_value, 'value'):
                field_value = field_value.value
            elif not isinstance(field_value, str):
                field_value = str(field_value)
            
            if isinstance(field_value, str) and field_value.strip():
                translated_data[field] = await translate_text(
                    field_value,
                    target_language,
                    context="medical"
                )
            elif isinstance(translated_data[field], list):
                # Handle list of items
                translated_list = []
                for item in translated_data[field]:
                    if isinstance(item, dict):
                        translated_item = await translate_dict_async(
                            item,
                            fields_to_translate,
                            target_language
                        )
                        translated_list.append(translated_item)
                    elif isinstance(item, str):
                        translated_list.append(
                            await translate_text(item, target_language, "medical")
                        )
                    else:
                        translated_list.append(item)
                translated_data[field] = translated_list
            elif isinstance(translated_data[field], dict):
                # Recursively translate nested dictionaries
                translated_data[field] = await translate_dict_async(
                    translated_data[field],
                    fields_to_translate,
                    target_language
                )
    
    return translated_data

