import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool, RunContextWrapper 
from dataclasses import dataclass

load_dotenv(find_dotenv())
set_tracing_disabled(True)

gemini_api_key=os.getenv("GEMINI_API_KEY")

external_client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=external_client
)

@dataclass
class StudentInfo:
    name: str
    
    
@function_tool
async def fetch_student_info(wrapper: RunContextWrapper[StudentInfo]) -> str:
    return f"Student {wrapper.context.name} roll number 23 is a student of I.C.S."
   


async def main():
    # Create your context object
    student_info=StudentInfo(name="Muhammad Javed")
    # define an agent that will use the tool above
    
    agent=Agent[StudentInfo](
        name="Assistant",
        instructions="You are a helpful assistant for retrieving student information. ",
        tools=[fetch_student_info],
        model=model
    )
    
    # Run the agent, passing in the local context
    result= await Runner.run(
        starting_agent=agent,
        input="tell me about student",
        context=student_info,
    )
    print(result.final_output)
    
    

if __name__ == "__main__":
    asyncio.run(main()) 
