# ai-legal
## Setup Instructions (Linux)

# Clone the repository
git clone https://github.com/PriyanshuRanjann/ai-legal.git

# Change directory to the project folder
cd ai-legal

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Docker qdrant image
docker pull qdrant/qdrant

# Run Docker container (example: for PostgreSQL)
docker run -d --name ai-legal-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Start Ollama and pull a model (e.g., llama2)
ollama pull phi:latest

# Run the model with Ollama
ollama run phi:latest

# Run the main application
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Run the streamlit application
streamlit run app.py

