## OS
Assume Ubuntu latest stable release.

## Languages
For simple shell routines, use bash. For more complex code, use Julia unless otherwise specified. For visualization, prefer GLMakie (if in Julia) or ggplot2 (if in R). For docs, use Markdown.

## Formatting
In Python, follow black formatting. In other languages, follow standard best practices for that language. **Except** as noted below:
- always use tabs for indentation
- always use snake case for new variable or constant definitions, and for naming files and folders, even in cases of acronyms

## Code style
Prefer switch statements to long if/elseif statements when practical.

Prefer terenary if/else instead of a code block where practical (but not if it violates black formatting).

Never use semicolons to cram multiple statements into one line, except in the case of constructions like `for x in y; do` in bash.

Never use comments.
