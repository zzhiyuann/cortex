# Forge Example Session

This walkthrough demonstrates creating a CSV-to-JSON converter tool using Forge.

## CLI Session

```
$ forge create "convert CSV to JSON with filtering"

╭─ Forge ───────────────────────────────────────╮
│ Creating tool: convert CSV to JSON with       │
│ filtering                                     │
│ Output type: python                           │
╰───────────────────────────────────────────────╯

Step 1: Clarification
  Found 4 question(s):

  ? What is the input format? (e.g., file path, string, stdin, JSON object)
    Options: file_path, string, stdin, json_object, other
  > file_path

  ? What should the output format be?
    Options: stdout, file, return_value, json, other
  > stdout

  ? What criteria should be used for filtering/searching?
  > Filter rows where any column value contains the query string

  ? How should the tool handle errors?
    Default: Print error message and exit with non-zero code
  > [enter]

  All questions answered.

Step 2: Generating tool...
  Code generated. (847 chars)
  Tests passed! (3 tests, 0 failures)

╭─ Success ─────────────────────────────────────╮
│ Tool created successfully!                    │
│                                               │
│ Name: convert_csv_to_json                     │
│ Tests: 3 passed                               │
│ Iterations: 0                                 │
╰───────────────────────────────────────────────╯
```

## MCP Session (via Claude Code)

```
User: I need a tool that converts CSV to JSON with filtering

Claude: Let me create that with Forge.
[calls forge_create("convert CSV to JSON with filtering", "python")]

Forge returns:
  session_id: "a1b2c3d4e5f6"
  questions: [
    {id: "q_1", question: "What is the input format?", ...},
    {id: "q_2", question: "What should the output format be?", ...},
    ...
  ]

Claude: I have a few questions:
1. What is the input format? (file path, string, stdin, etc.)
2. What should the output format be?

User: File path input, stdout output

Claude:
[calls forge_answer("a1b2c3d4e5f6", {"q_1": "file_path", "q_2": "stdout"})]
[calls forge_generate("a1b2c3d4e5f6")]

Forge returns:
  success: true
  tests: {passed: true, total: 3}

Claude: Your tool is ready! All 3 tests passed. Want me to install it?

User: Yes, install as MCP tool

Claude:
[calls forge_install("a1b2c3d4e5f6", "mcp")]

Forge returns:
  success: true
  message: "Installed MCP tool 'convert_csv_to_json'. Restart Claude Code to load."

Claude: Done! The tool is installed as an MCP server. Restart Claude Code to use it.
```

## Listing Tools

```
$ forge list

            Forge Tools
┌───────────────────┬────────┬──────────────┬───────┬───────────┬────────────┐
│ Name              │ Type   │ Description  │ Tests │ Installed │ Created    │
├───────────────────┼────────┼──────────────┼───────┼───────────┼────────────┤
│ convert_csv_to_j… │ python │ Convert CSV… │ PASS  │ Yes       │ 2026-02-22 │
│ fetch_url_links   │ cli    │ Download a … │ PASS  │ Yes       │ 2026-02-21 │
└───────────────────┴────────┴──────────────┴───────┴───────────┴────────────┘
```
