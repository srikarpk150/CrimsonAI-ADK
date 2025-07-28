from pydantic import BaseModel, Field

class DecisionOutput(BaseModel):
    action: str = Field(description="The action to be taken based on the user's query")
    career_goal: str = Field(description="The career goal of the user")


