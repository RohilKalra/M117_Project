from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import yaml

# Read openai_key from ../secrets.yaml file
with open("../secrets.yaml", "r") as stream:
    secrets = yaml.safe_load(stream)
    openai_key = secrets["openai_key"]


# Your existing tool functions remain the same
def suggest_destination(preferences):
    return "Paris, France"


def recommend_hotel(destination):
    return "Paris has only camping sites available."


def suggest_activities(destination):
    return "Visit the Louvre, Eiffel Tower, and Montmartre."


def get_weather(destination):
    return "Sunny with temperatures around 20°C."


def calculate_budget(input_string):
    """Calculate budget based on combined hotel and activities information"""
    try:
        data = json.loads(input_string)
        hotel = data["hotel"]
        activities = data["activities"]
        return f"Estimated budget: $2000 for {hotel} with activities: {activities}"
    except json.JSONDecodeError:
        return "Error: Please provide input as valid JSON with 'hotel' and 'activities' keys"
    except KeyError:
        return "Error: JSON must contain both 'hotel' and 'activities' keys"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)

# Create LLMChains with your prompt templates
destination_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["preferences"],
        template="Suggest a vacation destination based on: {preferences}",
    ),
)

hotel_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["destination"], template="Recommend hotels in {destination}."
    ),
)

activity_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["destination"], template="Suggest activities in {destination}."
    ),
)

weather_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["destination"], template="Provide weather for {destination}."
    ),
)

budget_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["hotels_and_activities"],
        template="Calculate a budget based on {hotels_and_activities}.",
    ),
)

# Define tools with the chains
destination_tool = Tool(
    name="DestinationTool",
    func=lambda p: destination_chain.run(preferences=p),
    description="Suggests a travel destination based on preferences.",
)

hotel_tool = Tool(
    name="HotelTool",
    func=lambda d: hotel_chain.run(destination=d),
    description="Recommends hotels in a destination.",
)

activity_tool = Tool(
    name="ActivityTool",
    func=lambda d: activity_chain.run(destination=d),
    description="Suggests activities for a destination.",
)

weather_tool = Tool(
    name="WeatherTool",
    func=lambda d: weather_chain.run(destination=d),
    description="Provides weather information for a destination.",
)

budget_tool = Tool(
    name="BudgetTool",
    func=lambda h, a: budget_chain.run(hotel=h, activities=a),
    description="Calculates travel budget based on hotel and activities.",
)

# Initialize agents with the tools
destination_agent = initialize_agent(
    tools=[destination_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True,
)

hotel_agent = initialize_agent(
    tools=[hotel_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True,
)

activity_agent = initialize_agent(
    tools=[activity_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True,
)

weather_agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True,
)

budget_agent = initialize_agent(
    tools=[budget_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True,
)


# Run the travel planning pipeline
def plan_travel(preferences):
    # Get destination recommendation
    destination = destination_agent.run(
        f"What would be a good travel destination for someone who wants: {preferences}"
    )

    # Get hotel recommendations
    hotels = hotel_agent.run(f"Find hotels in {destination}")

    # Get activity suggestions
    activities = activity_agent.run(f"What activities are available in {destination}?")

    # Get weather information
    weather = weather_agent.run(f"What's the weather like in {destination}?")

    # Combine hotel and activities info for budget calculation
    combined_info = f"hotel: {hotels} | activities: {activities}"
    budget = budget_agent.run(f"Calculate budget for {combined_info}")

    return {
        "destination": destination,
        "hotels": hotels,
        "activities": activities,
        "weather": weather,
        "budget": budget,
    }


# Example usage
preferences = "family-friendly, cultural sites, comfortable stay"
travel_plan = plan_travel(preferences)

for key, value in travel_plan.items():
    print(f"{key.capitalize()}: {value}")
