import language_tool_python
import textstat

tool = language_tool_python.LanguageTool('en-US')

def score_quality(text):
    matches = tool.check(text)
    words = len(text.split())
    errors = 0
    important_errors = 0
    suggestions = []
    for match in matches:
        if match.ruleId.startswith('TYPOS'):
            errors += 0.5
        else:
            errors += 1
            important_errors += 1
        suggestions.append(f"{match.message} (at position {match.offset})")
    quality_score = round(max(0, 1 - errors / words) * 100, 2) if words > 0 else 0
    readability = textstat.flesch_reading_ease(text) if words > 0 else 0
    return {
        "quality_score": quality_score,
        "error_count": len(matches),
        "important_error_count": important_errors,
        "readability": readability,
        "suggestions": suggestions
    }
