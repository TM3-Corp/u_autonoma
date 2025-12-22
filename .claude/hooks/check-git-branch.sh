#!/bin/bash
#
# Claude Code Hook: Branch Protection
#
# This hook runs BEFORE Claude executes git commit/push commands.
# It ensures interns only work on their assigned branches.
#
# How it works:
# 1. Claude Code calls this hook before running Bash commands
# 2. The hook receives the command via stdin (JSON format)
# 3. If it's a git commit/push, we check the branch
# 4. Exit 0 = allow, Exit 2 = block
#

# Read the tool input from stdin
INPUT=$(cat)

# Extract the command from JSON input
# The input format is: {"command": "git commit -m 'message'", ...}
COMMAND=$(echo "$INPUT" | grep -oP '"command"\s*:\s*"\K[^"]+' | head -1)

# If we couldn't parse it, try alternate method
if [[ -z "$COMMAND" ]]; then
    COMMAND=$(echo "$INPUT" | sed 's/.*"command"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
fi

# Check if this is a git commit or push command
if [[ "$COMMAND" =~ ^git[[:space:]]+(commit|push) ]]; then

    # Get current branch
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null)

    # Get allowed branch from git config (set during setup)
    ALLOWED_BRANCH=$(git config --get user.allowed-branch)

    # Protected branches that no one should commit/push to directly
    PROTECTED_BRANCHES=("main" "develop")

    # Check if on a protected branch
    for protected in "${PROTECTED_BRANCHES[@]}"; do
        if [[ "$CURRENT_BRANCH" == "$protected" ]]; then
            echo "BLOCKED: Cannot commit/push directly to '$protected'"
            echo ""
            echo "This branch is protected. Please use your feature branch:"
            echo "  git checkout feature/eda-YOUR_NAME"
            echo ""
            echo "Then create a Pull Request on GitHub to merge your changes."
            exit 2
        fi
    done

    # If an allowed branch is set, enforce it
    if [[ -n "$ALLOWED_BRANCH" ]]; then
        if [[ "$CURRENT_BRANCH" != "$ALLOWED_BRANCH" ]]; then
            echo "BLOCKED: You can only commit/push to your assigned branch"
            echo ""
            echo "  Current branch:  $CURRENT_BRANCH"
            echo "  Your branch:     $ALLOWED_BRANCH"
            echo ""
            echo "Please switch to your branch first:"
            echo "  git checkout $ALLOWED_BRANCH"
            exit 2
        fi
    fi

    # All checks passed
    echo "Branch check passed: $CURRENT_BRANCH"
fi

# Allow the command
exit 0
