######### Langraph code generation based on AlpaCodium ##########
import os
import logging
import subprocess
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, TypedDict
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langgraph.graph import END, StateGraph, START


####### Config ###########
# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create a logger instance for the application
logger = logging.getLogger("app")

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.', '.env')

# Check if the path is correct
if not os.path.isfile(env_path):
        logger.error(f"File not found: {env_path}")
        exit(1)

load_dotenv(dotenv_path=env_path, override=True)

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
	raise RuntimeError("OPENAI_API_KEY is not set in the environment")



################# Setup ############

# Generation prompt
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in Python. \n 
    Answer the user question based on your knowledge of Python. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Also, generate at least 3 tests to demonstrate code functionality. \n
    Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)


# Code generation chain
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    tests: List[str] = Field(description="Code block tests")
    
expt_llm = "gpt-4o-mini"
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model=expt_llm)
code_gen_chain = code_gen_prompt | llm.with_structured_output(code)
question = "How do I determine if a string is a palindrome?"
solution = code_gen_chain.invoke(
    {"messages": [("user", question)]}
)

# Print helper function
def print_multiline(input_string: str):
    lines = input_string.splitlines()
    for line in lines:
        print(line)


#print_multiline(solution.prefix + solution.imports + solution.code + "\n".join(solution.tests))


############ Graph State ################

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: List
    generation: str
    iterations: int

#################### Graph ##########################

### Parameter

# Max tries/
max_iterations = 3
# Reflect
flag = 'reflect'
# flag = "do not reflect"

# Nodes
def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]

    # Solution
    code_solution = code_gen_chain.invoke(
        {"messages": messages}
    )
    messages += [
        (
            "assistant",
            f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code} \n Tests: {"\n".join(solution.tests)}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code
    tests = code_solution.tests

    #print_multiline(imports + "\n" + code + "\n" + "\n".join(tests))

    # Check execution
    try:
        subprocess.run(
			['poetry', 'run', 'python', imports + "\n" + code + "\n" + "\n".join(tests)],
			capture_output=True, 
			text=True, 
			timeout=10
        )
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        print(f"Exception {e}")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


def reflect(state: GraphState):
    """
    Reflect on errors

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]
    error = state["error"]

    # Prompt reflection

    code_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced software engineer with great debugging skills \n 
    Here is the code generated:  \n ------- \n  {code_solution} \n ------- \n. The code was executed 
    and the errors recorded in {error} \n. Please review the error, and the code to suggest the following: \n
    1. Code quality and adherence to best practices \n
    2. Potential bugs or edge cases \n
    3. Performance optimizations \n
    4. Readability and maintainability \n
    5. Any security concerns \n""",
        ),
        ("placeholder", "{messages}"),
    ]
)

    # Add reflection
    reflections = code_gen_chain.invoke(
        {"messages": messages, "code_solution": code_solution, "error": error}
    )
    messages += [("assistant", f"Here are reflections on the error: {reflections}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


### Edges


def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"
        

########## Setup and invoke graph #############
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code", code_check)  # check code
workflow.add_node("reflect", reflect)  # reflect

# Build graph
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "reflect": "reflect",
        "generate": "generate",
    },
)
workflow.add_edge("reflect", "generate")
app = workflow.compile()

question = "How do I determine if a string is a palindrome?"
solution = app.invoke({"messages": [("user", question)], "iterations": 0, "error": ""})

print_multiline(solution["generation"].prefix + solution["generation"].imports + solution["generation"].code)
print(f"error {solution["error"]}")

