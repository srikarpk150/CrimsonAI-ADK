import os
import psycopg2
import json
import logging
from typing import List, Dict, Any, Optional
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# Sentence Transformer configuration
MODEL_NAME = os.getenv("EMBEDDING_MODEL")

# Cache directory configuration
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Global model cache
_model_cache = None



def get_db_connection():
    """Create and return a database connection"""
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        logger.info("DB Connected Successfully")
        return connection
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def get_sentence_transformer():
    """Get or create the cached sentence transformer model"""
    global _model_cache
    if _model_cache is None:
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(CACHE_DIR, exist_ok=True)
            
            # Check if model is already cached
            model_cache_path = os.path.join(CACHE_DIR, MODEL_NAME.replace("/", "_"))
            
            if os.path.exists(model_cache_path):
                logger.info(f"Loading cached model from: {model_cache_path}")
                _model_cache = SentenceTransformer(model_cache_path)
            else:
                logger.info(f"Downloading and caching model: {MODEL_NAME}")
                _model_cache = SentenceTransformer(MODEL_NAME)
                
                # Cache the model for future use
                _model_cache.save(model_cache_path)
                logger.info(f"Model cached to: {model_cache_path}")
            
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            return None
    return _model_cache

def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for text using cached sentence transformer"""
    try:
        model = get_sentence_transformer()
        if model is None:
            logger.error("Sentence transformer model not available")
            return None
        
        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def get_course_recommendations(query: str, user_id: Optional[str] = None, top_n: int = 10) -> str:
    """
    Get course recommendations based on the user's query.
    
    Args:
        query (str): User's query about courses or career goals
        user_id (str): Optional user ID for personalized recommendations
        top_n (int): Number of recommendations to return
    
    Returns:
        str: JSON string with course recommendations
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        logger.info(f'user_id: {user_id}')
        if not query_embedding:
            return json.dumps({"error": "Failed to generate embedding for query"})
        
        connection = get_db_connection()
        if not connection:
            return json.dumps({"error": "Failed to connect to database"})
        
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            # Build the recommendation query based on db_metadata schema
            recommendation_query = """
            WITH user_info AS (
                SELECT 
                    user_id,
                    remaining_credits,
                    upcoming_semester
                FROM iu_catalog.student_profile 
                WHERE user_id = %s
            ),
            completed_courses AS (
                SELECT DISTINCT course_id 
                FROM iu_catalog.completed_courses 
                WHERE user_id = %s
            ),
            eligible_courses AS (
                SELECT 
                    cd.course_id,
                    cd.course_name,
                    cd.department,
                    cd.min_credits,
                    cd.max_credits,
                    cd.offered_semester,
                    cd.prerequisites,
                    cd.course_title,
                    cd.course_description,
                    cd.embedding,
                    COALESCE(1 - (cd.embedding <=> %s::vector), 0) as similarity
                FROM iu_catalog.course_details cd
                LEFT JOIN user_info ui ON true
                LEFT JOIN completed_courses cc ON cd.course_id = cc.course_id
                WHERE 
                    -- Not already completed
                    cc.course_id IS NULL
                    -- Credits check (if user info available)
                    AND (ui.user_id IS NULL OR cd.max_credits <= ui.remaining_credits)
                    -- Semester check (if user info available)
                    AND (ui.user_id IS NULL OR cd.offered_semester ILIKE '%%' || ui.upcoming_semester || '%%')
                    -- Has embedding
                    AND cd.embedding IS NOT NULL
            )
            SELECT 
                course_id,
                course_name,
                department,
                min_credits,
                max_credits,
                offered_semester,
                prerequisites,
                course_title,
                course_description,
                similarity
            FROM eligible_courses
            ORDER BY similarity DESC
            LIMIT %s;
            """
            
            # Execute the query
            cursor.execute(recommendation_query, (user_id, user_id, query_embedding, top_n))
            results = cursor.fetchall()
            
            # Convert results to list of dictionaries
            courses = []
            for row in results:
                course = {
                    "course_id": row["course_id"],
                    "course_name": row["course_name"],
                    "department": row["department"],
                    "min_credits": row["min_credits"],
                    "max_credits": row["max_credits"],
                    "offered_semester": row["offered_semester"],
                    "prerequisites": row["prerequisites"] if row["prerequisites"] else [],
                    "course_title": row["course_title"],
                    "course_description": row["course_description"],
                    "similarity": float(row["similarity"])
                }
                courses.append(course)
            

            
            # Prepare response
            response = {
                "query": query,
                "user_id": user_id,
                "recommendations": courses,
                "total_recommendations": len(courses),
                "embedding_dimensions": len(query_embedding) if query_embedding else 0
            }
            logger.info('Successfully generated course recommendations')
            return json.dumps(response, indent=2)
            
    except Exception as e:
        logger.error(f"Error in get_course_recommendations: {e}")
        return json.dumps({
            "error": f"Failed to get course recommendations: {str(e)}",
            "query": query,
            "user_id": user_id
        })
    
    finally:
        if connection:
            connection.close()

def get_course_details(course_id: str) -> str:
    """
    Get detailed information about a specific course.
    
    Args:
        course_id (str): The course ID to get details for
    
    Returns:
        str: JSON string with course details
    """
    try:
        connection = get_db_connection()
        if not connection:
            return json.dumps({"error": "Failed to connect to database"})
        
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get course details
            course_query = """
            SELECT 
                cd.course_id,
                cd.course_name,
                cd.department,
                cd.min_credits,
                cd.max_credits,
                cd.offered_semester,
                cd.prerequisites,
                cd.course_title,
                cd.course_description,
                cd.course_details
            FROM iu_catalog.course_details cd
            WHERE cd.course_id = %s;
            """
            
            cursor.execute(course_query, (course_id,))
            result = cursor.fetchone()
            
            if result:
                course_info = dict(result)
                return json.dumps(course_info, indent=2)
            else:
                return json.dumps({"error": f"Course {course_id} not found"})
                
    except Exception as e:
        logger.error(f"Error in get_course_details: {e}")
        return json.dumps({"error": f"Failed to get course details: {str(e)}"})
    
    finally:
        if connection:
            connection.close()

def get_user_profile(user_id: str) -> str:
    """
    Get user profile information.
    
    Args:
        user_id (str): The user ID to get profile for
    
    Returns:
        str: JSON string with user profile
    """
    try:
        connection = get_db_connection()
        if not connection:
            return json.dumps({"error": "Failed to connect to database"})
        
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get user profile
            profile_query = """
            SELECT 
                sp.user_id,
                sp.degree_type,
                sp.major,
                sp.enrollment_type,
                sp.gpa,
                sp.total_credits,
                sp.completed_credits,
                sp.remaining_credits,
                sp.time_availability,
                sp.upcoming_semester,
                l.first_name,
                l.last_name,
                l.email
            FROM iu_catalog.student_profile sp
            LEFT JOIN iu_catalog.login l ON sp.user_id = l.user_id
            WHERE sp.user_id = %s;
            """
            
            cursor.execute(profile_query, (user_id,))
            result = cursor.fetchone()
            
            if result:
                profile = dict(result)
                return json.dumps(profile, indent=2)
            else:
                return json.dumps({"error": f"User {user_id} not found"})
                
    except Exception as e:
        logger.error(f"Error in get_user_profile: {e}")
        return json.dumps({"error": f"Failed to get user profile: {str(e)}"})
    
    finally:
        if connection:
            connection.close()

# Export functions for ADK
__all__ = [
    "get_course_recommendations",
    "get_course_details", 
    "get_user_profile"
]