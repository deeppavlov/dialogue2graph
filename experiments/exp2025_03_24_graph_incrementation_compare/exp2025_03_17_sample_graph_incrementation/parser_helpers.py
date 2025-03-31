import re
from langchain_core.tracers.schemas import Run


def remove_backticks_and_format(input_string):
    # Регулярное выражение:
    # ^`+ - удаляет ` в начале
    # (json\s*\{)? - опционально захватывает "json{", возможно с пробелами/переводами строк
    # (.*?) - захватывает основное содержимое (нежадно)
    # (\}?`+$) - опционально захватывает "}" перед закрывающими `
    pattern = r'^`+(json\s*)?(.*?)(`+)?$'
    
    # Заменяем на вторую и третью группу (основное содержимое)
    processed_string = re.sub(pattern, r'\2', input_string, flags=re.DOTALL)
    
    return processed_string