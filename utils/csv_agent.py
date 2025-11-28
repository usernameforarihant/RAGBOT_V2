"""
CSV Agent Handler
Uses LangChain Experimental CSV Agent for CSV files (LangChain v0.3.x)
"""

from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from typing import Dict
import os


class CSVAgent:
    """Handles CSV queries using LangChain CSV Agent."""
    
    # Prompt prefix and suffix for better CSV analysis
    CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns, get the column names, then answer the question.
"""
    
    CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
- FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result, reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "

Explanation:
".
In the explanation, mention the column names that you used to get
to the final answer.
"""
    
    def __init__(self, csv_file_path: str, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize CSV Agent.
        
        Args:
            csv_file_path: Path to CSV file
            model_name: OpenAI model to use
            temperature: Temperature for generation
        """
        self.csv_file_path = csv_file_path
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Create CSV agent
        self.agent = create_csv_agent(
            llm=self.llm,
            path=csv_file_path,
            verbose=True,
            allow_dangerous_code=True,  # Required for CSV agent to execute code
            agent_type="openai-tools"  # Use OpenAI tools for better performance
        )
    
    def query(self, question: str) -> Dict[str, str]:
        """
        Query the CSV using the agent.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer
        """
        try:
            # Construct full query with prefix and suffix
            full_query = self.CSV_PROMPT_PREFIX + question + self.CSV_PROMPT_SUFFIX
            
            # Run agent
            response = self.agent.invoke({"input": full_query})
            
            # Extract answer from response
            if isinstance(response, dict):
                answer = response.get("output", str(response))
            else:
                answer = str(response)
            
            return {
                "answer": answer,
                "source": "csv_agent",
                "file": self.csv_file_path
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing CSV query: {str(e)}",
                "source": "csv_agent",
                "file": self.csv_file_path,
                "error": str(e)
            }
    
    def query_simple(self, question: str) -> str:
        """
        Simple query without prompt engineering.
        
        Args:
            question: User question
            
        Returns:
            Answer string
        """
        try:
            response = self.agent.invoke({"input": question})
            
            if isinstance(response, dict):
                return response.get("output", str(response))
            return str(response)
            
        except Exception as e:
            return f"Error: {str(e)}"