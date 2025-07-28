"""
Prompts for Course Recommendation System
========================================

This file contains all the prompt templates used in the course recommendation system.
The prompts are organized by functionality and include detailed documentation.
"""

# =============================================================================
# DECISION AGENT PROMPTS
# =============================================================================

DECISION_PROMPT = """
You are a highly intelligent Course Recommendation Assistant.

Your primary task is to analyze user queries and classify them into a structured JSON format with one of the following actions:

1. "recommendation" — If the user mentions a **career goal** (e.g., "I want to become a data scientist"), extract the goal and set the action to "recommendation". The query should then be handled by the **recommendation_agent**.

2. "inquiry" — If the user is asking about a **specific course** (e.g., course description, timing, instructor), extract the course name and set the action to "inquiry". The query should then be handled by the **inquiry_agent**.

3. "clarification_needed" — If the user's message is vague, lacks a clear goal or course, or is playful/unserious, set the action to "clarification_needed". The query should then be handled by the **clarification_agent**.

Classification Rules:
- If both a career goal and a course name are mentioned, prioritize **"recommendation"**.
- If a new career goal is introduced in a later query, treat it as a **new recommendation**.
- Do **not** assume a career goal unless the user clearly implies it (e.g., "I want to become...", "I'm thinking of becoming...", "I want to switch to...").
- If the user expresses dissatisfaction, frustration, or feeling lost **without a clear next step**, classify the query as **"clarification_needed"**.
- Do **not** infer meaning from jokes, metaphors, or playful language. Only classify based on explicit educational or career-related intent.


"""

# =============================================================================
# RECOMMENDATION AGENT PROMPTS
# =============================================================================

RECOMMENDATION_PROMPT = """
You are an Advanced Course Recommendation Agent with deep expertise in academic and career path guidance. Your primary objective is to provide personalized, strategic course recommendations based on a student's:
- Career aspirations
- Academic interests
- Skill development needs
- Holistic educational growth

Core Recommendation Principles:
1. Contextualize Recommendations
- Go beyond surface-level course matching
- Consider long-term career trajectory
- Identify skill gaps and development opportunities
- Create a strategic learning pathway

2. Recommendation Framework
- Provide 5 course recommendations
- Include detailed description for each recommendation
- Connect courses to:
  * Long-term career goals
  * Skill development
  * Interdisciplinary learning

3. Recommendation Structure
For each recommended course, provide:
- course_name and course_title
- course_description
- Skill Development Alignment
- Career Impact

4. Analytical Depth
- Analyze course descriptions
- Consider prerequisite knowledge
- Evaluate potential learning outcomes
- Assess interdisciplinary connections

Additional Instructions:
- Extract the student's explicit career goal from the query.
- Use the list of available courses returned by the tool to select appropriate recommendations.

"""

# =============================================================================
# INQUIRY AGENT PROMPTS
# =============================================================================

INQUIRY_PROMPT = """
You are a course information expert at a university. You provide detailed and accurate information about specific courses when students inquire about them.

Please provide comprehensive information about the requested course, including:
- Course description
- Prerequisites (if any)
- Credit hours
- Typical semester offerings
- Related courses that might also interest the student

"""

# =============================================================================
# CLARIFICATION AGENT PROMPTS
# =============================================================================

CLARIFICATION_PROMPT = """
You are a helpful academic advisor assistant. The user has provided a query that needs clarification before you can assist them properly.

Please generate a clarifying question that will help determine what the user is looking for. Your response should be friendly, helpful, and guide the user toward providing the information needed to assist them with course selection or academic information.

"""


# =============================================================================
# OUTPUT SCHEMAS AND EXAMPLES
# =============================================================================

# Decision Output Schema Examples
DECISION_OUTPUT_EXAMPLES = {
    "recommendation_example": {
        "action": "recommendation",
        "career_goal": ["UX Designer"],
        "course_name": [],
        "course_work": ["User Interface Design", "Human-Computer Interaction", "UX Research Methods"],
        "original_query": "I want to become a UX Designer",
        "reasoning": "The user clearly expresses a career goal using 'I want to become', which aligns with the recommendation action."
    },
    "inquiry_example": {
        "action": "inquiry",
        "career_goal": [],
        "course_name": ["Advanced Database Concepts"],
        "course_work": ["SQL Optimization", "NoSQL Systems", "Database Design"],
        "original_query": "What topics are covered in Advanced Database Concepts?",
        "reasoning": "The user is directly asking about the content of a specific course, which fits the inquiry action."
    },
    "clarification_example": {
        "action": "clarification_needed",
        "career_goal": [],
        "course_name": [],
        "course_work": [],
        "original_query": "I'm so lost. This semester has been overwhelming.",
        "reasoning": "The user is expressing emotional distress without providing a clear goal or inquiry. It's unclear what help they need, so clarification is required."
    }
}

# Course Recommendation Schema Examples
COURSE_RECOMMENDATION_EXAMPLES = {
    "machine_learning_example": {
        "course_id": "096191",
        "course_code": "CS401",
        "course_title": "Machine Learning",
        "course_description": "Introduction to machine learning algorithms and applications",
        "skill_development": ["Python", "Statistical Analysis", "Algorithm Design"],
        "career_alignment": "Core skill for data scientists",
        "relevance_reasoning": "This course directly addresses the user's interest in becoming a data scientist by providing foundational skills in machine learning algorithms, which are essential for processing and analyzing large datasets to extract meaningful insights - a primary responsibility of data scientists."
    },
    "ux_design_example": {
        "course_id": "084226",
        "course_code": "BUEX-V 594",
        "course_title": "User Experience Design",
        "course_description": "Principles and methods of designing digital products focused on the user experience",
        "skill_development": ["Wireframing", "User Testing", "Prototyping"],
        "career_alignment": "Essential for UX Designer roles",
        "relevance_reasoning": "This course directly addresses the user's goal of becoming a UX Designer by teaching core user experience methodologies. The hands-on prototyping projects will build a portfolio that employers look for, and the user testing components align perfectly with the user-centered research mentioned in the query."
    }
}

# Recommendation Output Schema Examples
RECOMMENDATION_OUTPUT_EXAMPLES = {
    "complete_recommendation_example": {
        "recommended_courses": [
            {
                "course_id": "096191",
                "course_code": "CS401",
                "course_title": "Machine Learning",
                "course_description": "Introduction to machine learning algorithms and applications",
                "skill_development": ["Python", "Statistical Analysis", "Algorithm Design"],
                "career_alignment": "Core skill for data scientists",
                "relevance_reasoning": "This course directly addresses your interest in becoming a data scientist by providing foundational skills in machine learning algorithms, which are essential for processing and analyzing large datasets to extract meaningful insights - a primary responsibility of data scientists."
            },
            {
                "course_id": "084226",
                "course_code": "BUEX-V 594",
                "course_title": "User Experience Design",
                "course_description": "Principles and methods of designing digital products focused on the user experience",
                "skill_development": ["Wireframing", "User Testing", "Prototyping"],
                "career_alignment": "Essential for UX Designer roles",
                "relevance_reasoning": "Based on your interest in human-computer interaction, this course will provide you with practical skills in UX design that complement your technical background. The combination of technical knowledge and user experience skills is highly valued in the industry."
            }
        ],
        "recommendation_strategy": "These recommendations focus on building your technical skills in data science while also expanding your knowledge in user experience design, which aligns with your expressed career interests and complements your computer science background.",
        "additional_guidance": "Consider taking these courses in sequence, with Machine Learning first to build your technical foundation, followed by User Experience Design to diversify your skill set."
    }
}

# General Inquiry Output Schema Examples
GENERAL_INQUIRY_EXAMPLES = {
    "course_inquiry_example": {
        "course_information": {
            "course_id": "CS301",
            "course_name": "Advanced Programming",
            "prerequisites": ["CS101", "CS201"]
        },
        "professor_name": "Dr. Jane Smith",
        "total_strength": 120
    }
}

# Clarification Output Schema Examples
CLARIFICATION_EXAMPLES = {
    "career_guidance_clarification": {
        "clarification_question": "Could you share what career path you're interested in exploring?",
        "possible_intents": ["Career guidance", "Course selection", "Degree requirements"]
    },
    "emotional_support_clarification": {
        "clarification_question": "I understand you might be feeling overwhelmed. Could you share more about what's making you consider dropping out?",
        "possible_intents": ["Academic struggles", "Financial concerns", "Personal challenges", "Lack of interest in current program"]
    }
}

# =============================================================================
# PROMPT UTILITY FUNCTIONS
# =============================================================================

def get_decision_prompt_with_examples():
    """
    Returns the decision prompt with embedded examples for better understanding.
    """
    return DECISION_PROMPT

def get_recommendation_prompt_with_examples():
    """
    Returns the recommendation prompt with embedded examples for better understanding.
    """
    return RECOMMENDATION_PROMPT

def get_inquiry_prompt_with_examples():
    """
    Returns the inquiry prompt with embedded examples for better understanding.
    """
    return INQUIRY_PROMPT

def get_clarification_prompt_with_examples():
    """
    Returns the clarification prompt with embedded examples for better understanding.
    """
    return CLARIFICATION_PROMPT

def get_supervisor_prompt_with_examples():
    """
    Returns the supervisor prompt with embedded examples for better understanding.
    """
    return SUPERVISOR_PROMPT

def get_title_prompt_with_examples():
    """
    Returns the title generation prompt with embedded examples for better understanding.
    """
    return TITLE_PROMPT



# Export all prompts for easy access
__all__ = [
    "DECISION_PROMPT",
    "RECOMMENDATION_PROMPT", 
    "INQUIRY_PROMPT",
    "CLARIFICATION_PROMPT",
    "SUPERVISOR_PROMPT",
    "TITLE_PROMPT",
    "DECISION_OUTPUT_EXAMPLES",
    "COURSE_RECOMMENDATION_EXAMPLES",
    "RECOMMENDATION_OUTPUT_EXAMPLES",
    "GENERAL_INQUIRY_EXAMPLES",
    "CLARIFICATION_EXAMPLES",
    "get_decision_prompt_with_examples",
    "get_recommendation_prompt_with_examples", 
    "get_inquiry_prompt_with_examples",
    "get_clarification_prompt_with_examples",
    "get_supervisor_prompt_with_examples",
    "get_title_prompt_with_examples"
]
