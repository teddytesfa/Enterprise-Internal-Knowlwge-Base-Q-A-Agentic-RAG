import re

def markdown_to_text(md: str) -> str:
    """Simple markdown -> plain text conversion.
    """
    text = md

    # remove fenced code blocks
    text = re.sub(r"```(?:.|\n)*?```", " ", text)
    # remove inline code
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # remove images: ![alt](url)
    text = re.sub(r"!	[.*?]	(.*?)}", " ", text)
    # convert links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # remove headings # ...
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # remove bold/italic markers
    text = re.sub(r"(\*\*|\*|__|_)(.*?)\1", r"\2", text)
    # remove html tags
    text = re.sub(r"<[^>]+>", " ", text)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
