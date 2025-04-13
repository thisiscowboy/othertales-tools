# othertales unified openapi tools server
A comprehensive server that provides document storage, memory, Git versioning, and web scraping capabilities for LLMs via OpenWebUI.
## Features
- **Document Management**: Store and retrieve documents with Git versioning
- **Knowledge Graph**: Store structured data and track user preferences
- **Git Integration**: Version control for documents
- **Web Scraping**: Extract content from websites and convert to Markdown
- **OpenWebUI Integration**: Fully compatible with OpenWebUI
## Getting Started
### Installation
#### Using Docker
```bash
docker build -t unified-tools-server .
docker run -p 8000:8000 -v $(pwd)/data:/app/data unified-tools-server
```
#### Without Docker
```bash
# Install dependencies
pip install -r requirements.txt
# Install Playwright browsers
playwright install chromium
# Run the server
python main.py
```
### Configuration
Edit the `.env` file to configure the server:
```
# Server settings
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEV_MODE=False
# Storage settings
ALLOWED_DIRS=./data,~/documents
MEMORY_FILE_PATH=./data/memory.json
# Git settings
DEFAULT_COMMIT_USERNAME=UnifiedTools
DEFAULT_COMMIT_EMAIL=tools@example.com
```
## API Documentation
When running, documentation is available at:
- OpenAPI JSON: http://localhost:8000/openapi.json
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
## Usage with OpenWebUI
In OpenWebUI, add a new Tool with the following URL:
```
http://localhost:8000/openapi.json
```

### API Key Authentication
The serper service supports Bearer token authentication:

1. Configure your API key in the `.env` file:
   ```
   API_KEY=your_secure_api_key_here
   ```

2. When using OpenWebUI, add the API key in the Bearer token field:
   - When adding the tool, expand the "Authorization" section
   - Select "Bearer" as the authentication type
   - Enter your API key in the token field

The API key info endpoint (`/api-key-info`) provides information about whether API key authentication is enabled.
## License
SEE EULA