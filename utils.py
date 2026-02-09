import ast

def is_valid_python(code):
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def extract_python_code_robust(text, include_preface=True):
    """
    Robustly extracts Python code from text, removing trailing comments and examples.

    This function:
    1. Extracts code from markdown blocks (```python...```) or uses raw text
    2. Uses AST parsing to extract only functions and their preface (imports, globals)
    3. Removes trailing comments like "# Example usage:" sections

    Args:
        text: The text containing Python code (with or without markdown blocks)
        include_preface: If True, includes imports/globals before functions. Default True.

    Returns:
        str: Clean Python code containing only the preface and function definitions
    """
    from base.code import TextFunctionProgramConverter

    # First, try to extract code from markdown blocks
    if "```python" in text:
        start_idx = text.find("```python") + len("```python")
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            code = text[start_idx:end_idx].strip()
        else:
            code = text[start_idx:].strip()
    else:
        code = text.strip()

    # Use AST-based parsing to extract only the program structure
    # This automatically removes trailing comments and example usage
    program = TextFunctionProgramConverter.text_to_program(code)

    if program is None:
        # If AST parsing fails, fall back to the original code
        return code

    # Convert the program back to string (this excludes trailing comments)
    if include_preface:
        return str(program)
    else:
        # Return only the functions without preface
        return '\n'.join([str(f) for f in program.functions])