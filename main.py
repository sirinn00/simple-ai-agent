from langchain_core.messages import HumanMessage #high level framework that allows us to build AI applications
from langchain_openai import ChatOpenAI #allows us to use OpenAI within LangChain and LangGraph
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent #complex framework that allows us three to build AI agents
from dotenv import load_dotenv #load environment variables from a .env file into the systems environment variables

load_dotenv() #load environment variables from .env file



def main():
    model = ChatOpenAI(model="gpt-4o-mini",temperature=0) #initialize the ChatOpenAI model with a temperature of 0 for deterministic responses
    #the higher temperature, the more random the model is going to be.

    tools = [] #initialize an empty list to hold tools


    agent_executor = create_react_agent(model, tools)

    print("Welcome i am your AI agent. How can I help you today? Type 'quit' to quit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "quit":
            break
        
        print("\nAssistant: ", end="")

        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]} #wrap user input in a HumanMessage object
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="") #print the content of each message chunk as it is received

        print()

if __name__ == "__main__":
    main()       

