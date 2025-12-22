#!/bin/bash
#
# Setup script for interns
# Run this after cloning the repository
#
# Usage: ./scripts/setup_intern.sh
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  U_AUTONOMA - Intern Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Step 1: Get intern name
echo -e "${YELLOW}Step 1: Who are you?${NC}"
echo ""
echo "  1) Vicente"
echo "  2) Sebastian"
echo "  3) Paul (admin - no restrictions)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        INTERN_NAME="Vicente"
        BRANCH_NAME="feature/eda-vicente"
        ;;
    2)
        INTERN_NAME="Sebastian"
        BRANCH_NAME="feature/eda-sebastian"
        ;;
    3)
        INTERN_NAME="Paul"
        BRANCH_NAME=""
        echo ""
        echo -e "${GREEN}Admin mode: No branch restrictions applied.${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

# Step 2: Configure Git user (if not set)
echo ""
echo -e "${YELLOW}Step 2: Configuring Git user...${NC}"

CURRENT_NAME=$(git config --get user.name)
CURRENT_EMAIL=$(git config --get user.email)

if [[ -z "$CURRENT_NAME" ]]; then
    read -p "Enter your full name for Git: " git_name
    git config user.name "$git_name"
    echo -e "  Set user.name to: ${GREEN}$git_name${NC}"
else
    echo -e "  user.name already set: ${GREEN}$CURRENT_NAME${NC}"
fi

if [[ -z "$CURRENT_EMAIL" ]]; then
    read -p "Enter your email for Git: " git_email
    git config user.email "$git_email"
    echo -e "  Set user.email to: ${GREEN}$git_email${NC}"
else
    echo -e "  user.email already set: ${GREEN}$CURRENT_EMAIL${NC}"
fi

# Step 3: Set allowed branch (for interns only)
echo ""
echo -e "${YELLOW}Step 3: Setting branch permissions...${NC}"

if [[ -n "$BRANCH_NAME" ]]; then
    git config user.allowed-branch "$BRANCH_NAME"
    echo -e "  Your assigned branch: ${GREEN}$BRANCH_NAME${NC}"
    echo -e "  ${YELLOW}Claude Code will only allow commits/pushes to this branch.${NC}"
else
    git config --unset user.allowed-branch 2>/dev/null
    echo -e "  ${GREEN}Admin: Full access to all branches.${NC}"
fi

# Step 4: Claude Code hooks info
echo ""
echo -e "${YELLOW}Step 4: Claude Code hooks...${NC}"
echo -e "  Claude Code hooks are configured in: ${GREEN}.claude/settings.json${NC}"
echo -e "  Branch protection hook: ${GREEN}.claude/hooks/check-git-branch.sh${NC}"
echo ""
echo -e "  When you use Claude Code to commit or push, the hook will:"
echo "  - Check if you're on your assigned branch"
echo "  - Block the action if you're on main, develop, or another intern's branch"
echo "  - Show a helpful message explaining what to do"

# Step 5: Switch to correct branch
echo ""
echo -e "${YELLOW}Step 5: Switching to your branch...${NC}"

if [[ -n "$BRANCH_NAME" ]]; then
    git checkout "$BRANCH_NAME" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo -e "  Switched to: ${GREEN}$BRANCH_NAME${NC}"
    else
        echo -e "  ${YELLOW}Branch not found locally. Fetching...${NC}"
        git fetch origin
        git checkout "$BRANCH_NAME" 2>/dev/null || git checkout -b "$BRANCH_NAME" origin/"$BRANCH_NAME"
        echo -e "  Switched to: ${GREEN}$BRANCH_NAME${NC}"
    fi
else
    echo -e "  Admin: Staying on current branch ($(git branch --show-current))"
fi

# Step 6: Create Python virtual environment
echo ""
echo -e "${YELLOW}Step 6: Setting up Python environment...${NC}"

if [[ ! -d "venv" ]]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip > /dev/null
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter requests > /dev/null
    echo -e "  ${GREEN}Virtual environment created and packages installed.${NC}"
else
    echo -e "  Virtual environment already exists."
fi
echo -e "  Activate with: ${CYAN}source venv/bin/activate${NC}"

# Done!
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SETUP COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  Welcome, ${CYAN}$INTERN_NAME${NC}!"
echo ""
if [[ -n "$BRANCH_NAME" ]]; then
    echo -e "  ${YELLOW}Your branch:${NC} $BRANCH_NAME"
    echo ""
    echo "  Workflow with Claude Code:"
    echo "  1. Ask Claude to make changes to files"
    echo "  2. Ask Claude to commit: \"commit my changes\""
    echo "  3. Ask Claude to push: \"push to GitHub\""
    echo "  4. Create a Pull Request on GitHub to 'develop'"
    echo ""
    echo -e "  ${RED}The Claude hook will block you if you try to commit${NC}"
    echo -e "  ${RED}to main, develop, or someone else's branch.${NC}"
fi
echo ""
echo "  To start Claude Code:"
echo -e "  ${CYAN}claude${NC}"
echo ""
echo "  Happy coding!"
echo ""
