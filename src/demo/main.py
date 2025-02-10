#!/usr/bin/env python
import os
import sys
import warnings


from demo.crew import Demo


os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ['LITELLM_LOG'] = 'DEBUG'

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")



def run():
    description_path = "src/description"  # "/app/crewai/src/description"  # //os.environ["DESCRIPTION_PATH"]
    issue_description = open(description_path + "/issue_description.txt", "r").read()
    """
    Run the crew.
    """
    inputs = {
        "issue_description": issue_description
    }

    Demo().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Demo().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Demo().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Demo().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
