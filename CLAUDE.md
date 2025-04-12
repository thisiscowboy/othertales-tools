# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Run all tests: `python -m pytest tests/`
- Run specific test module: `python -m pytest tests/core/test_git_service.py`
- Run specific test class: `python -m pytest tests/core/test_git_service.py::TestGitService`
- Run specific test: `python -m pytest tests/core/test_git_service.py::TestGitService::test_get_status`
- Lint code: `python -m flake8`
- Format code: `python -m black .`
- Check types: `python -m mypy .`

## Project Structure
- `app/api/` - API endpoints and routers
- `app/core/` - Core business logic services
- `app/models/` - Pydantic data models
- `app/utils/` - Utility functions and helpers
- `tests/` - Unit and integration tests

## Scheduler and Verification Services
The application includes:
- Scheduled web scraping
- GitHub repository integration
- Content verification for knowledge graph completeness
- Document processing and embedding

## Code Style Guidelines
- Line length: 100 characters max
- Formatting: Use black with default settings
- Imports: Use isort with black profile, grouped by stdlib, third-party, local
- Types: Static typing preferred but not strictly enforced
- Naming: snake_case for functions/variables, PascalCase for classes
- Error handling: Use specific exception types, avoid bare except
- Avoid unnecessary comments - code should be self-documenting
- Follow existing patterns in nearby files when adding new code
- Use async/await for I/O-bound operations
- Handle platform-specific path issues with proper normalization