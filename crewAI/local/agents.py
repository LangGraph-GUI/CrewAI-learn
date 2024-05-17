import os
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

from crewai_tools import BaseTool

#llama3 = Ollama(model= 'dolphin-llama3:8b-256k')

llama3 = Ollama(model= 'phi3')


class FileWriterTool(BaseTool):
    name: str = "FileWriter"
    description: str = "Writes given content to a specified file."

    def _run(self, filename: str, content: str) -> str:
        # Open the specified file in write mode and write the content
        with open(filename, 'w') as file:
            file.write(content)
        return f"Content successfully written to {filename}"


# Assuming the FileWriterTool has been imported and set up as described above
file_writer = FileWriterTool()

# Define your agents with roles, goals, and the new tool
researcher = Agent(
    role='Knowledge Article Writer',
    goal='Create content of professional domains longer than 1000 words',
    backstory="Write articles about Game Design.",
    verbose=True,
    allow_delegation=True,
    llm = llama3,
    tools=[file_writer]  # Include the file writer tool
)

writer = Agent(
    role='SEO Writer',
    goal='Convert article into SEO version',
    backstory="Define keywords and titles for better SEO.",
    verbose=True,
    allow_delegation=True,
    llm = llama3,
    tools=[file_writer]  # Include the file writer tool
)

# Create tasks for your agents
task1 = Task(
    description="Write several articles.",
    expected_output="At least 4 topics saved to files",
    agent=researcher,
    tools=[file_writer],
    function_args={'filename': '<articles>.md', 'content': 'Example article content'}  # Adjust as necessary
)

task2 = Task(
    description="Make articles into SEO version",
    expected_output="Convert into 8 SEO pages saved to files. the files name start with seo_",
    agent=writer,
    tools=[file_writer],
    function_args={'filename': 'seo_<articles>.md', 'content': 'SEO optimized article content'}  # Adjust as necessary
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2
)

# Get your crew to work!
result = crew.kickoff()
print("######################")
print(result)
