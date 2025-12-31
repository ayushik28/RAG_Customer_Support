import re

def redact_pii(text: str) -> str:
    text = re.sub(r'\b\d{10}\b', '[REDACTED_PHONE]', text)
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b', '[REDACTED_CARD]', text)
    return text
