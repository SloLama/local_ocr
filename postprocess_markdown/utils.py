import re


def extract_content_from_tag(model_output, tag):
    """
    Extracts the content from a specific XML-like tag within a string.

    Args:
        model_output: The string containing the tagged content.
        tag: The name of the tag (without the angle brackets).

    Returns:
        The content within the specified tag, or None if the tag is not found.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, model_output, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
