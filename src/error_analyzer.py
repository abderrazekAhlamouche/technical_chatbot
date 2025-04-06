CXX_ERRORS = {
    "segmentation fault": {
        "solution": "Check for null pointer dereference or out-of-bounds array access",
        "docs_ref": "Memory Management section"
    },
    "undefined reference": {
        "solution": "Verify linking of all object files and correct header inclusions",
        "docs_ref": "Linking section"
    },
    "template error": {
        "solution": "Ensure all template parameters are properly specified when used",
        "docs_ref": "Templates section"
    },
    "syntax error": {
        "solution": "Check for missing semicolons, parentheses or incorrect scope delimiters",
        "docs_ref": "Basic Syntax section"
    }
}

def analyze_error(error_code):
    """Analyze C++ specific error messages"""
    error_code = error_code.lower().strip()
    for pattern, info in CXX_ERRORS.items():
        if pattern in error_code:
            return (f"ERROR: {error_code}\n"
                   f"SOLUTION: {info['solution']}\n"
                   f"DOCS REFERENCE: {info['docs_ref']}")
    return "Not a recognized C++ compilation/runtime error"

if __name__ == "__main__":
    print(analyze_error("segmentation fault"))