# CrimsonAI-ADK

A sophisticated course recommendation system built with Google ADK (Agent Development Kit) that provides personalized course recommendations based on user career goals and academic history.

## Overview

CrimsonAI-ADK is an intelligent course recommendation system that leverages Google's Agent Development Kit to provide personalized academic guidance. The system uses multiple specialized AI agents to analyze user queries, understand career goals, and recommend relevant courses from a comprehensive database.

## Architecture

### Multi-Agent System
The application uses a sophisticated multi-agent architecture with specialized agents:

- **DecisionAgent**: Classifies user queries into recommendation, inquiry, or clarification actions
- **RecommendationAgent**: Provides personalized course recommendations based on career goals
- **InquiryAgent**: Handles specific course information requests
- **ClarificationAgent**: Generates clarifying questions for vague queries
- **CourseRecommendationWorkflow**: Coordinates the entire workflow between agents

### Technology Stack
- **Google ADK**: Agent Development Kit for AI agent orchestration
- **Google Generative AI**: Gemini 2.0 Flash model for natural language processing
- **PostgreSQL**: Cloud SQL database for course and user data
- **Sentence Transformers**: For semantic search and embeddings
- **Google Cloud Platform**: Logging and monitoring

## Prerequisites

- Python 3.8+
- Google Cloud Platform account
- PostgreSQL Cloud SQL instance
- Google ADK access

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/srikarpk150/CrimsonAI-ADK.git
   cd CrimsonAI-ADK
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the MainAgent directory:
   ```env
   # Google Cloud Configuration
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
   GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id
   GOOGLE_CLOUD_LOCATION=your_google_cloud_location
   
   # LLM Configuration
   LLM_MODEL=gemini-2.0-flash
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   
   # Database Configuration
   DATABASE_URL=your_postgresql_connection_string
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   DB_NAME=your_database_name
   DB_USER=your_database_username
   DB_PASSWORD=your_database_password
   ```

5. **Configure Google Cloud credentials**
   - Place your `crimsonai-key.json` file in the project root
   - Ensure the service account has necessary permissions for:
           - Google ADK
      - Cloud SQL
      - Logging and Monitoring

## Database Setup

The system uses PostgreSQL Cloud SQL with Google ADK integration. The complete database schema is documented in `db_metadata.txt`.

### Core Tables

**User Management:**
- **login**: User authentication and profile information
- **student_profile**: Academic profiles with degree type, major, GPA, and credit tracking

**Course Data:**
- **course_details**: Comprehensive course information with semantic embeddings for search
- **course_trends**: Course popularity metrics, enrollment data, and performance statistics
- **completed_courses**: User course completion history with grades and credits

### Key Features

- **Semantic Search**: Course details include embeddings for intelligent course matching
- **Performance Tracking**: Course trends track enrollment, ratings, GPA, and time spent
- **User Context**: Student profiles include time availability and upcoming semester info

### Database Schema Documentation

For detailed table structures, field types, and relationships, refer to `db_metadata.txt` in the project root. This file contains the complete PostgreSQL schema including:
- Field types and constraints
- Primary and foreign key relationships
- Default values and indexes
- JSONB field structures

## Usage

### Running the Main Agent

To start the multi-agent course recommendation system:

```bash
adk web
```

This command serves as the entry point to the multi-agent setup, launching the course recommendation workflow with all specialized agents (Decision, Recommendation, Inquiry, and Clarification agents).



## Configuration

### Environment Variables
**Google Cloud Configuration:**
- `GOOGLE_API_KEY`: Your Google API key for accessing Google services
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud service account credentials
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID
- `GOOGLE_CLOUD_LOCATION`: Your Google Cloud region/location

**LLM Configuration:**
- `LLM_MODEL`: The LLM model to use (default: gemini-2.0-flash)
- `EMBEDDING_MODEL`: The sentence transformer model for semantic search (default: sentence-transformers/all-MiniLM-L6-v2)

**Database Configuration:**
- `DATABASE_URL`: PostgreSQL connection string
- `DB_HOST`: Database host address
- `DB_PORT`: Database port number
- `DB_NAME`: Database name
- `DB_USER`: Database username
- `DB_PASSWORD`: Database password for authentication

### Model Configuration
The system uses sentence transformers for semantic search:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Cached in: `MainAgent/model_cache/`

## Features

- **Personalized Recommendations**: Based on career goals and academic history
- **Semantic Search**: Advanced course matching using embeddings
- **Multi-turn Conversations**: Maintains context across interactions
- **Intelligent Query Classification**: Automatically routes queries to appropriate agents
- **Course Trend Analysis**: Considers popularity and performance metrics

## Development

### Project Structure
```
CrimsonAI-ADK/
├── MainAgent/
│   ├── agent.py          # Main agent orchestration
│   ├── models.py         # Pydantic models
│   ├── prompts.py        # Agent prompts and instructions
│   ├── tools.py          # Database tools and functions
│   └── model_cache/      # Cached sentence transformers
├── requirements.txt      # Python dependencies
├── db_metadata.txt      # Database schema documentation
└── README.md           # This file
```

### Adding New Agents
1. Define the agent in `agent.py`
2. Add corresponding prompts in `prompts.py`
3. Update the workflow in `course_recommendation_workflow`

### Database Tools
The `tools.py` file contains functions for:
- Course recommendations (`get_course_recommendations`)
- Course details (`get_course_details`)
- Database connections and queries



## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request



## Support

For issues and questions:
- Check the database metadata in `db_metadata.txt`
- Review agent configurations in `MainAgent/agent.py`
- Ensure all environment variables are properly set

---

**Note**: This system requires Google ADK access and proper Google Cloud Platform setup. Ensure all credentials and permissions are correctly configured before running the application. 
