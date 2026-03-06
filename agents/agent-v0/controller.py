import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv
    load_dotenv("../api_keys/agent0.env")
except ImportError:
    print("Warning: python-dotenv not installed. Using manual input.")
    pass


## Monitoring
os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

with open("../api_keys/openai_agent0.txt", "r") as f:
   os.environ["OPENAI_API_KEY"] = f.read().strip()


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
    api_key=os.environ.get("OPENAI_API_KEY")
)
messages = [
    SystemMessage(content="your favorite person is RJ"),
    HumanMessage(content="who is your fav"),
]

system_template = "Your favorite person is {person}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"person": "RJ", "text": "Who is your favorite person"})

response = model.invoke(prompt)
print(response.content)