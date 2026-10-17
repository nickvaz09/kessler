"""Kessler AI assistant package.

Server-side agentic tool loop against the Claude API:
- prompt.py  - frozen system prompt (prompt-cache safe)
- tools.py   - tool schemas + executor (DATA / UI-ACTION / create_report)
- service.py - the chat loop itself (client injectable for tests)
- report.py  - standalone HTML report renderer for create_report
"""
