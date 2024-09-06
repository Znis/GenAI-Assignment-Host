import streamlit as st
import requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents import tool
from langchain_core.output_parsers import StrOutputParser
import os


WEATHER_API_KEY=st.secrets['WEATHER_API_KEY']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
LANGCHAIN_API_KEY = st.secrets['LANGCHAIN_API_KEY']


os.environ["LANGCHAIN_TRACING_V2"]="true" # enables the tracing
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=os.getenv(LANGCHAIN_API_KEY)
os.environ["LANGCHAIN_PROJECT"]="assignment-4" #project name in the LangSmith platform

@tool
def get_weather_data(city: str) -> str:
    """Calls the Weather API and return the weather data
    Args:
        city: str
    Returns:
        str
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    return str(response.json())


@tool
def get_city_name(location: str) -> str:
    """Calls the Location API and returns the address data
    Args:
        location: str
    Returns:
        str
    """
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"

    headers = {
    'User-Agent': 'MyGeocodingApp/1.0 (your-email@example.com)'
}

    response = requests.get(url, headers=headers)
    if(len(response.json()) > 0):
        return response.json()[0]

    return "City not found"

def extract_from_stream(output_stream):
    target_key = 'output'  
    for chunk in output_stream:
        if isinstance(chunk, dict):
            if target_key:
                if target_key in chunk:
                    yield chunk[target_key]

def init_agent():

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key= OPENAI_API_KEY)

    tools = [ get_city_name, get_weather_data ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are very powerful assistant equipped with multiple tools whose responsibility
                is to provide weather data or answer the question related to weather.
                Here is the detailed instruction:
                1. Call the Weather API to get the weather data of the city.
                2. If the Weather API returns valid response with weather data then, return the weather data in given output format.
                3. If the Weather API returns no weather data then, call the Location API to get the city name.
                4. If the Location API returns a valid address then, extract only the city name from it.
                5. Call the Weather API again to get the weather data of the extracted city.
                6. If the Weather API returns valid response with weather data for the extracted city then, return the weather data in given output format for the extracted city.
                7. If the Location API returns no valid city name or the Weather API cant get the weather data of the city then, return the response
                saying the weather data of the city is not available.
                If the query is not related to asking weather then, respond politely asking for weather related questions.
                The desired output format for different scenarios are given below:
                ###
                #Scenario where the weather data of the city or extracted city is available:
                Here is the weather data of the city <city_name>:
                <weather_data_in_bullet_form (dash separated)>
                #
                #Scenario where the weather data of the city or extracted city is not available, or city name is not valid:
                The weather data of the city <city_name> is not available.
                #
                ###
                """,

            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # sequence of messages that contain the previous agent tool invocations and the corresponding tool outputs
        ]
    )


    llm_with_tools = llm.bind_tools(tools)


    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor






agent_executor = init_agent()

# Show title and description.
st.title("💬 Weather")
st.write(
    "This is a simple chatbot that provides you weather data of the any given city. "
    "Ask the weather of any city in the input field."
)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if len(st.session_state) <= 0:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("Message"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    output = agent_executor.stream({"input": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(extract_from_stream(output))
    st.session_state.messages.append({"role": "assistant", "content": response})






