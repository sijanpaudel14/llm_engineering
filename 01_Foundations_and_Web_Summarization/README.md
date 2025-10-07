# Week 1: Foundation - AI Web Summarizer

## Project Title

**AI-Powered Web Content Summarizer** - Your First LLM Application

## Problem Statement

Information overload on the web makes it difficult to quickly extract key insights from lengthy articles and documents. Traditional reading requires significant time investment, while manual summarization is subjective and time-consuming.

## Solution & Key Features

Built an intelligent web summarizer that transforms any URL into concise, actionable insights using OpenAI's GPT models.

### Core Features:

- **URL-to-Summary Pipeline**: Automated web scraping and content extraction
- **Intelligent Summarization**: Context-aware summarization using frontier LLMs
- **Multiple Provider Support**: OpenAI integration with fallback to local Ollama models
- **Interactive Jupyter Interface**: User-friendly notebook-based interface
- **Error Handling**: Robust error management and graceful failures

## Technical Stack

- **Frontend**: Jupyter Notebooks with interactive widgets
- **LLM Integration**: OpenAI API (GPT-4o-mini, GPT-4o)
- **Web Scraping**: Python requests library with content parsing
- **Local Alternative**: Ollama integration for free local inference
- **Development**: Python 3.x, dotenv for configuration management

## Key Achievement

🏆 **Delivered a production-ready LLM application in first week** - Demonstrates rapid prototyping skills and ability to integrate multiple technologies into a cohesive solution.

## Setup & Usage Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (or Ollama for local development)
- Required packages (see requirements.txt)

### Environment Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sijanpaudel14/llm_engineering.git
   cd llm_engineering/week1
   ```

2. **Install dependencies**:

   ```bash
   pip install -r ../requirements.txt
   ```

3. **Configure API keys**:
   ```bash
   cp ../.env.example ../.env
   # Edit .env file with your OpenAI API key
   ```

### Running the Project

1. **Start Jupyter Lab**:

   ```bash
   jupyter lab
   ```

2. **Open the main notebook**: `day1.ipynb`

3. **Execute cells sequentially** to see the web summarizer in action

### Alternative: Local Development with Ollama

For free local development without API costs:

1. **Install Ollama**: Download from [ollama.com](https://ollama.com)
2. **Pull a model**: `ollama pull llama3.2`
3. **Use local configuration** in notebooks (examples provided)

## Project Structure

```
week1/
├── README.md                    # This file
├── day1.ipynb                   # Main web summarizer project
├── day2 EXERCISE.ipynb          # Additional exercises and challenges
├── day5.ipynb                   # Extended functionality
├── Guide to Jupyter.ipynb      # Jupyter tutorial for beginners
├── Intermediate Python.ipynb   # Python skills refresher
├── troubleshooting.ipynb       # Common issues and solutions
└── solutions/                  # Reference implementations
```

## Skills Demonstrated

### Technical Skills

- **API Integration**: RESTful API consumption and error handling
- **Web Technologies**: HTTP requests, content parsing, URL processing
- **Python Programming**: Object-oriented design, error handling, environment management
- **LLM Applications**: Prompt engineering, response processing, model selection

### Professional Skills

- **Problem Solving**: Breaking down complex requirements into manageable components
- **Documentation**: Clear code documentation and user guides
- **Testing**: Edge case handling and error recovery
- **User Experience**: Creating intuitive interfaces for non-technical users

## Next Steps & Extensions

- Add support for PDFs and other document formats
- Implement caching for frequently accessed URLs
- Add batch processing capabilities
- Create a web interface using Gradio or Streamlit
- Integrate with browser extension for one-click summarization

---

**Learning Outcome**: This project establishes the foundation for all subsequent LLM engineering work, demonstrating the complete pipeline from problem identification to deployed solution.
