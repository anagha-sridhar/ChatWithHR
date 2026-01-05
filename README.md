Project Overview

This project aims to develop a Generative AI-powered agent using Python and the Llama Large Language Model (LLM) to help in exploring and analyzing Human Resource data within companies.
By translating plain English questions into structured data operations, the agent will facilitate evidence-based decision-making, allowing non-technical users (e.g., HR managers, business leaders) to perform complex data analysis without writing code or SQL queries. 

System Architecture 

Input Data:
Users upload HR data in CSV or Excel format via the web interface.
The Python backend uses the pandas library for efficient data loading, cleaning, and the initial calculation of essential Key Performance Indicators (KPIs)


LLM model:
The Llama model is integrated into the environment using Ollama, a lightweight, accessible runtime.
User questions are fed to the LLM. The Llama model is prompted to translate this into a structured, executable JSON instruction. This instruction specifies the required filters, metrics, aggregation logic, and other comparisons.


Output Data:
The engine dynamically executes the structured JSON instruction and performs required data manipulations using pandas library. 
The output will dynamically generate analytical results, such as comparisons across years, countries, or organizational groups.


Run the ChatWithHR.py file and when the agentic model appears, upload the file_v2.xlsx file to it. Then proceed to ask it questions. 
