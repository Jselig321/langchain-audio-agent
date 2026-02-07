import os
from constant import openai_key, assemblyai_key
import streamlit as st
import assemblyai as aai
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate

aai.settings.api_key = assemblyai_key
os.environ["OPENAI_API_KEY"] = openai_key

st.title("Audio Transcription + Summary Agent")

uploaded_file= st.file_uploader("Upload MP3", type=["mp3"])

@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribes an MP3 audio file and returns the text transcript."""
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speech_models=["universal"])
    transcript = transcriber.transcribe(file_path, config=config)
    return transcript.text

@tool
def summarize_text(text: str) -> str:
   """Summarizes a block of transcript text."""
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   response = llm.invoke(f"Summarize this transcript: \n\n{text}")
   return response.content

# Agent Setup
llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [transcribe_audio, summarize_text]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can transcribe and summarize audio."),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

if uploaded_file and st.button("Start Transcription and Summerization"):
   with st.spinner("Processing audio..."):
      with open("temp.mp3", "wb") as f:
            f.write(uploaded_file.read())

      result = agent_executor.invoke({"input": "Transcribe temp.mp3 and summarize it."})
      result = result["output"]

   st.success("Done!")

   st.subheader("Summary")
   st.write(result)