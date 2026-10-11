from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from datetime import date

today = date.today().isoformat()

load_dotenv()

search = SerperDevTool()

researcher = Agent(
    role="Tech News Researcher",
    goal="Find today's top 3 important technology news stories.",
    backstory="You are a technology news researcher who searches the web and collects accurate, recent news with source links.",
    tools=[search],
    verbose=True
)

writer = Agent(
    role="Tech Newsletter Writer",
    goal="Turn researched technology news into a clear, engaging 5-line newsletter.",
    backstory="You are a concise technology newsletter writer. You use only the research provided to you and never perform web searches yourself.",
    verbose=True
)

find = Task(
    description=f"Search the web for the latest technology news as of {today}. Find exactly 3 recent stories. For each story, provide a headline, short summary, publication date, and direct source URL. Use the current date {today} in your search queries. Do not invent news or dates.",
    expected_output="Exactly 3 recent technology news stories. Each story must include its headline, a short summary, publication date, and source URL.",
    agent=researcher
)

write = Task(
    description="Using the research provided, write a concise tech newsletter with exactly 5 lines. Include the 3 news stories and a final line explaining why they matter.",
    expected_output="Exactly 5 lines: a title 'Tech in 5 lines', three numbered news story lines, and one line starting with 'Why it matters:'. Each line must fit on a single line.",
    agent=writer,
    context=[find]
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[find, write],
    process=Process.sequential,
    verbose=True
)

print(crew.kickoff())
