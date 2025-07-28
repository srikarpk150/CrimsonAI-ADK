from google.adk.agents import LlmAgent, LoopAgent
import os
import sys
import os.path

# Add the current directory to Python path to find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompts import (
    DECISION_PROMPT,
    RECOMMENDATION_PROMPT,
    INQUIRY_PROMPT,
    CLARIFICATION_PROMPT
)

from tools import get_course_recommendations, get_course_details

user_id = 'vpothapr'
user_context = 'The current user_id is ' + user_id

# Get model name from environment variable
LLM_MODEL = os.getenv("LLM_Model", "gemini-2.0-flash")

decision_agent = LlmAgent(
    name="DecisionAgent",
    model=LLM_MODEL,
    instruction=DECISION_PROMPT,
    description="Analyzes user queries and classifies them into recommendation, inquiry, or clarification_needed actions",
    output_key="decision_result",
)

recommendation_agent = LlmAgent(
    name="RecommendationAgent",
    model=LLM_MODEL,
    instruction=f"{user_context}\n\n{RECOMMENDATION_PROMPT}",
    description="Provides personalized course recommendations based on career goals and available courses",
    output_key="recommendation_result",
    tools=[get_course_recommendations]

)

inquiry_agent = LlmAgent(
    name="InquiryAgent",
    model=LLM_MODEL,
    instruction=INQUIRY_PROMPT,
    description="Handles specific course information requests and provides detailed course details",
    output_key="inquiry_result",
    tools=[get_course_details]

)

clarification_agent = LlmAgent(
    name="ClarificationAgent",
    model=LLM_MODEL,
    instruction=CLARIFICATION_PROMPT,
    description="Generates clarifying questions for vague or unclear user queries",
    output_key="clarification_result",

)


course_recommendation_workflow = LlmAgent(
    name="CourseRecommendationWorkflow",
    model=LLM_MODEL,
    instruction="""
    You are the Course Recommendation Workflow.

    **Important: Never return the classification result from DecisionAgent to the user.**

    Steps:
    1. Always call DecisionAgent to classify.
    2. Based on the classification, call exactly one of: RecommendationAgent, InquiryAgent, or ClarificationAgent.
    3. Only return the output from the second agent to the user. The intermediate JSON from DecisionAgent must be ignored.
        
        Always maintain conversation context and ensure smooth transitions between agents.
    """,
    description="Coordinates the course recommendation workflow between specialized agents",
    sub_agents=[
        decision_agent,
        recommendation_agent,
        inquiry_agent,
        clarification_agent,
    ],
)

root_agent = course_recommendation_workflow 