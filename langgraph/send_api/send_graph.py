# Standard library
import operator
import os
from typing import Annotated, TypedDict

# Third-party
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# Make sure the GOOGLE_API_KEY environment variable is set. If not, prompt the user to enter it.
if "GOOGLE_API_KEY" not in os.environ:
    print("Please set the GOOGLE_API_KEY environment variable.")

# Initialize the Gemini 3.0+ model with the desired parameters. You can adjust these parameters as needed.
model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,  
    max_tokens=None,
    timeout=None,
    max_retries=2,

)

# Define the state of our graph. In this case, we have a list of questions and a list of answers. The answers list is annotated with operator.add, which means that when we return a new state with an 'answers' key, it will be added to the existing list of answers rather than replacing it.
class MessagesState(TypedDict):
    questions: list[str]
    answers: Annotated[list[str], operator.add]

# Define the logic for our graph. The generate_answers node will take the list of questions and return a new state with the same list of questions. The send_logic function will create a Send object for each question, which will trigger the answer_questions node for each question. The answer_questions node will call the LLM to get an answer for the question and return a new state with the answer added to the list of answers.
def generate_answers(state: MessagesState) -> MessagesState:
    # Dummy node to trigger the send logic. It just returns the same list of questions that it receives.
    return {'questions': state['questions']}
    
# The send_logic function takes the current state and returns a list of Send objects. Each Send object specifies the node to send to (in this case, "answer_questions") and the data to send (in this case, the question). This will trigger the answer_questions node for each question in the list.
def send_logic(state: MessagesState) -> MessagesState:
    return [Send("answer_questions",{"question": s}) for s in state['questions']]

# The answer_questions function takes the current state (which includes a single question) and calls the LLM to get an answer. It then returns a new state with the answer added to the list of answers. In this example, we create a list of messages that includes a system message to set the context for the LLM and a human message that contains the question. We then call model.invoke(messages) to get the response from the LLM and return it in the state.
def answer_questions(state: MessagesState) -> MessagesState:
    # This is where you would call your LLM to get an answer to the question.
    messages = [SystemMessage(content="You are a helpful assistant that answers questions about the world."), HumanMessage(content=state['question'])]
    response = model.invoke(messages)

    return {'answers': [response.text]}

# Build the graph. 
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("generate_answers", generate_answers)
graph_builder.add_node("answer_questions", answer_questions)
graph_builder.add_edge(START,"generate_answers")
# The add_conditional_edges function takes a node name, a function that generates a list of Send objects, and a list of node names that the Send objects can send to. In this case, we specify that the generate_answers node should use the send_logic function to generate Send objects that send to the answer_questions node.
graph_builder.add_conditional_edges("generate_answers", send_logic, ["answer_questions"])
graph_builder.add_edge("answer_questions", END)
graph = graph_builder.compile()

# Invoke the graph with an initial state that includes a list of questions. The graph will process the questions in parallel and return a final state that includes the answers.
result = graph.invoke({"questions": ["What is the capital of France?", "What is the largest mammal?"]})

print(result)