import os
from pyowm.owm import OWM
from dotenv import load_dotenv
import json
from langgraph.graph import Graph
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from a .env file
load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("OPENWEATHERMAP_API_KEY")

# Initialize the OpenWeatherMap API
owm = OWM(api_key)
mgr = owm.weather_manager()

# Specify the local language model
local_llm = "mistral"

# Initialize the ChatOllama model with desired parameters
llm = ChatOllama(model=local_llm, format="json", temperature=0)

def Agent(question):
    # Define the prompt template
    template = """
    Question: {question} Let's think step by step.
    The question might contain a city name with a typo. Please identify the correct city name and respond with the weather information in the following JSON format:
    {{
        "city": "",
        "status": "",
        "temperature": ""
    }}
    """
    
    prompt = PromptTemplate.from_template(template.strip())

    # Format the prompt with the input variable
    formatted_prompt = prompt.format(question=question)

    llm_chain = prompt | llm | StrOutputParser()
    generation = llm_chain.invoke(formatted_prompt)
    
    return generation

def Tool(input):
    print("Tool Stage input:" + input)
    # Parse the JSON input
    data = json.loads(input)
    city_name = data.get("city", "")
    # Fetch weather data
    weather_data = get_weather(city_name)
    if isinstance(weather_data, str):
        # If there is an error string, return it directly
        return weather_data
    else:
        # Prepare the output content
        content = f"Weather in {weather_data['city']}: {weather_data['status']}\n"
        content += f"Temperature: {weather_data['temperature']}°C\n"
        return content

def get_weather(city_name):
    try:
        # Fetch weather data for the given city
        observation = mgr.weather_at_place(city_name)
        weather = observation.weather
        
        # Extract relevant weather information
        temperature = weather.temperature('celsius')
        status = weather.detailed_status
        return {
            "city": city_name,
            "status": status,
            "temperature": temperature['temp']
        }
    except Exception as e:
        return f"Error fetching weather data: {e}"

# Define a Langchain graph
workflow = Graph()

workflow.add_node("agent", Agent)
workflow.add_node("tool", Tool)

workflow.add_edge('agent', 'tool')

workflow.set_entry_point("agent")
workflow.set_finish_point("tool")

app = workflow.compile()

# Example invocation with potential typo correction
question = "What's the temperature in the city of angle?"
result = app.invoke(question)
print(result)
