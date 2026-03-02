#!/bin/bash

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    ███╗   ███╗██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗         ║
║    ████╗ ████║██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝         ║
║    ██╔████╔██║██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗           ║
║    ██║╚██╔╝██║██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝           ║
║    ██║ ╚═╝ ██║██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗         ║
║    ╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝         ║
║                                                                      ║
║              H O W   T O :   M B R I D G E   T E S T I N G         ║
╚══════════════════════════════════════════════════════════════════════╝

  MBridge unit tests run automatically on every PR. To also trigger
  functional tests, attach the label and re-run the workflow step.

  ┌─────────────────────────────────────────────────────────────────┐
  │  DEFAULT  │  Unit tests run on every PR (no action needed)      │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │    Every PR  ──►  cicd-mbridge-testing  ──►  unit tests only   │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 1  │  Attach the label to your PR (for functional tests)  │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │    PR Labels  ──►  [ + Add label ]  ──►  "Run MBridge tests"   │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 2  │  Re-run this workflow step                           │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │    Actions  ──►  [ Re-run jobs ]  ──►  Re-run failed jobs      │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  RESULT  │  Unit + functional tests run!                        │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │         cicd-mbridge-testing  ◄── unit + functional tests      │
  │                                                                  │
  │         Tests run against MBridge using the merge commit       │
  │         SHA of your pull request.                              │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘

                ┌────────────────────────────────────┐
                │  Label present?     NO   → unit    │
                │  Label present?     YES  → unit +  │
                │                           functional│
                └────────────────────────────────────┘

  NOTE: The label must be present BEFORE the re-run is triggered.
        The CI checks for "Run MBridge tests" at runtime.

  NOTE: All MBridge test results are optional — failures do not
        block merging your PR.
EOF
