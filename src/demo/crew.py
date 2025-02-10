import glob
import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.project import CrewBase, agent, task, after_kickoff, crew

llm_base_url =  os.environ['LLM_BASE_URL'] #"http://localhost:11434" # "http://localhost:4000"
MODEL_NAME = os.environ['LLM_MODEL_NAME'] # "ollama/llama3.1" # "litellm_proxy/gigachat-custom-model" #
EMBED_MODEL_URL =  os.environ['EMBED_MODEL_URL'] # "http://localhost:8000/embed"

llm = LLM(
    model=MODEL_NAME,
    base_url=llm_base_url
)

embedder_config = {
    "provider": "huggingface",
    "config": {
        "api_url": EMBED_MODEL_URL
    }
}

files = glob.glob("knowledge/*", recursive=True)
files = [file.replace("knowledge/", "", 1) for file in files]

text_source = TextFileKnowledgeSource(file_paths=files)


@CrewBase
class Demo():
    """Demo crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @after_kickoff  # Optional hook to be executed after the crew has finished
    def log_results(self, output):
        # Example of logging results, dynamically changing the output
        print(f"Results: {output}")
        print(text_source.content )
        return output

    @agent
    def developer(self) -> Agent:
        return Agent(
            config=self.agents_config['developer'],
            llm=llm,
            knowledge_sources=[text_source],
            embedder=embedder_config,
            verbose=True
        )

    @task
    def solve_issue(self) -> Task:
        return Task(
            config=self.tasks_config['solve_issue'],
            output_file='report.txt'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Demo crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            llm=llm,
        )
