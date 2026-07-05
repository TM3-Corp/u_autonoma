# Contributing to U_Autonoma

Welcome to the Early Warning System project! This guide will help you get started with Claude Code.

## For Interns (Vicente & Sebastian)

### First-Time Setup

After cloning the repository, run the setup script:

```bash
# Clone the repo
git clone https://github.com/TM3-Corp/u_autonoma.git
cd u_autonoma

# Run setup (this configures your branch and permissions)
chmod +x scripts/setup_intern.sh
./scripts/setup_intern.sh
```

The setup script will:
1. Assign you to your branch (`feature/eda-vicente` or `feature/eda-sebastian`)
2. Configure Claude Code hooks for branch protection
3. Set up the Python environment

### Working with Claude Code

Start Claude Code in the project directory:

```bash
cd u_autonoma
claude
```

Then you can ask Claude to:
- **Explore data:** "Show me the structure of student_features.csv"
- **Create notebooks:** "Create a new EDA notebook analyzing time patterns"
- **Make changes:** "Add a visualization showing grade distribution"
- **Commit work:** "Commit my changes with message 'Add grade histogram'"
- **Push to GitHub:** "Push my changes"

### Your Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. START CLAUDE CODE                                       │
│     $ claude                                                │
├─────────────────────────────────────────────────────────────┤
│  2. ASK CLAUDE TO MAKE CHANGES                              │
│     "Create a notebook that analyzes page_views vs grades"  │
├─────────────────────────────────────────────────────────────┤
│  3. ASK CLAUDE TO COMMIT                                    │
│     "Commit these changes"                                  │
├─────────────────────────────────────────────────────────────┤
│  4. ASK CLAUDE TO PUSH                                      │
│     "Push to GitHub"                                        │
├─────────────────────────────────────────────────────────────┤
│  5. CREATE PULL REQUEST (on GitHub website)                 │
│     - Go to: github.com/TM3-Corp/u_autonoma                 │
│     - Click "Compare & pull request"                        │
│     - Set base branch to: develop                           │
│     - Request review from: Paul                             │
└─────────────────────────────────────────────────────────────┘
```

### Branch Rules

| Branch | Who Can Push | Purpose |
|--------|--------------|---------|
| `main` | Nobody directly | Production code (merged via PR) |
| `develop` | Nobody directly | Integration branch (merged via PR) |
| `feature/eda-vicente` | Vicente only | Vicente's EDA work |
| `feature/eda-sebastian` | Sebastian only | Sebastian's EDA work |

---

## Understanding Claude Code Hooks

This project uses **Claude Code hooks** to enforce branch discipline and code quality. These are different from traditional Git hooks!

### What Are Claude Code Hooks?

Claude Code hooks are scripts that run at specific events during a Claude session. They receive JSON via stdin with context about the tool call and can allow, block, or provide feedback.

### Our Hooks

| Hook | Event | Matcher | Purpose |
|------|-------|---------|---------|
| `protect-main.sh` | `PreToolUse` | `Bash` | Blocks `git commit` on `main` branch |
| `lint-check.sh` | `PostToolUse` | `Write\|Edit` | Runs ESLint (TS/JS) or ruff (Python) after file changes |

**protect-main.sh** — Reads the Bash command from JSON stdin. If it's a `git commit` and the current branch is `main`, it exits with code 2 (blocked). Otherwise passes through.

**lint-check.sh** — Reads the file path from JSON stdin after a Write or Edit. Runs the appropriate linter (ESLint for .ts/.tsx/.js/.jsx, ruff for .py). Reports issues but doesn't block.

### Example: What Happens When Blocked

If Vicente tries to commit while on `main`:

```
You: "commit my changes"

Claude: I'll commit your changes.
[Runs: git commit -m "..."]

BLOCKED: Direct commits to 'main' are not allowed. Switch to 'dev' or a feature branch.
```

### How Hooks Are Configured

The hook configuration lives in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/protect-main.sh"}]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/lint-check.sh"}]
      }
    ]
  }
}
```

Valid hook events: `PreToolUse` (before a tool runs), `PostToolUse` (after a tool runs). The `matcher` filters which tool triggers the hook.

### Agents (`.claude/agents/`)

Specialized read-only reviewers you can delegate to:

| Agent | Reviews |
|-------|---------|
| `schema-reviewer` | Drizzle schema, migrations, TimescaleDB |
| `api-reviewer` | NestJS endpoints, auth, Swagger, DTOs |
| `ml-reviewer` | Python ML code, reproducibility, feature registry |
| `pr-reviewer` | General PR review, delegates to domain agents |

All agents are read-only — they can read and search code but cannot write, edit, or execute commands.

---

## Getting Updates from Main

When Paul merges updates to `develop` that you need:

```
You: "Get the latest changes from develop"

Claude will:
1. git fetch origin
2. git merge origin/develop
3. Handle any conflicts
4. Explain what changed
```

Or manually:

```bash
git fetch origin
git merge origin/develop
```

---

## Project Structure

```
u_autonoma/
├── .claude/                 # Claude Code configuration
│   ├── settings.json        # Hook configuration
│   ├── hooks/               # Hook scripts
│   │   ├── protect-main.sh  # Branch protection
│   │   └── lint-check.sh    # Post-write linting
│   ├── agents/              # Specialized reviewers
│   ├── rules/               # Contextual rules by file glob
│   └── skills/              # Task-oriented workflows
├── data/                    # Data files
│   ├── early_warning/       # Processed features
│   └── baseline/            # Model results
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks (your main work area!)
├── scripts/                 # Python scripts
│   └── setup_intern.sh      # Run this first!
└── CLAUDE.md                # API documentation
```

### For Your EDA Work

Create new notebooks in `notebooks/`:

```
notebooks/
├── 01_grade_prediction.ipynb           # Existing
├── 02_early_warning_visualization.ipynb # Existing
├── eda_vicente_01_exploration.ipynb     # Your work
├── eda_sebastian_01_analysis.ipynb      # Your work
```

**Naming convention:** `eda_YOURNAME_##_description.ipynb`

---

## Commit Message Format

When asking Claude to commit, be descriptive:

```
"Commit with message: Add correlation analysis between page views and grades"
```

Good commit messages:
- `Add: new visualization for time-of-day patterns`
- `Fix: incorrect calculation in activity score`
- `Update: improved data loading in notebook`
- `Analyze: grade distribution by course section`

---

## Questions?

- **Stuck with Git?** Ask Claude: "Help me understand what branch I'm on"
- **Hook blocked you?** Read the message - it tells you exactly what to do
- **Need data help?** Check `CLAUDE.md` for API documentation
- **Want examples?** Look at existing notebooks in `notebooks/`

Happy analyzing!
