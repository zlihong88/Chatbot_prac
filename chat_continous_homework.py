# Introduction
# To reduce the number of tutors required in the practicals of GIS courses for undergrads, this AI tool was developed, which can answer students' questions pertinent to each practical.
# Compared to asking ChatGPT directly, the AI tool's responses are more specific and concise.


from openai import AsyncOpenAI
import asyncio
from datetime import datetime
import fitz  # PyMuPDF
import os

# Create the core part of the tool - the OpenAI client
client = AsyncOpenAI()

# Construct the system prompt
system_prompt_template = """You are Lucky, a virtual tutor create by Lihong. The course you are working for is {course_name}. You answer students' questions with responses that are clear, straightforward, and factually accurate, without speculation or falsehood. Given the following context, please answer each question truthfully to the best of your abilities based on the provided information. Answer each question with a brief summary followed by several bullet points. 

Example:
Tutor Lucky is happy to help you:
- bullet point 1
- bullet point 2
...

<context>
{context}
</context>
"""
# Replace context_content with text from tutorials (PDF files)

# Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            # Extract text
            text += page.get_text()
    return text

# Function to extract text from multiple PDFs
def extract_text_from_multiple_pdfs(pdf_folder_path):
    context_content = ""
    for file_name in os.listdir(pdf_folder_path):
        if file_name.endswith(".pdf"):  # Check if the file is a PDF
            pdf_path = os.path.join(pdf_folder_path, file_name)
            print(f"Processing: {file_name}")
            context_content += extract_text_from_pdf(pdf_path) + "\n"
    return context_content

# Path to the folder containing PDF files
pdf_folder_path = "/Users/lz/Desktop/Chatbot_prac/PDF_data"   # (Don't include assignment questions in the context!)

# Extract text from all PDFs in the folder
context_content = extract_text_from_multiple_pdfs(pdf_folder_path)

# Set the context in the system prompt
system_prompt = system_prompt_template.format(
    context=context_content, 
    course_name="GEOM2001 - Geographic Information System"
)

# Define a chat_fun to call the OpenAI client
# Use asynchronous programming to handle I/O-bound tasks efficiently
# Use stream=True to have streaming output
async def chat_func(history):

    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}] + history,
        max_tokens=256,
        temperature=0.5,
        stream=True,
    )

    buffer = ""
    async for r in result:
        next_token = r.choices[0].delta.content
        if next_token:
            print(next_token, flush=True, end="")  # The flush=True argument forces the output to be displayed immediately, without waiting for the buffer to fill up or for a newline (\n) character.
            buffer += next_token

    print("\n", flush=True)

    return buffer

# Define a continous_chat to save history and enable users continously asking questions
async def continous_chat():
    history = []

    # Loop to receive user input continously
    while(True):
        user_input = input("> ")
        if user_input == "exit":
            break

        history.append({"role": "user", "content": user_input})

        # notice every time we call the chat function
        # we pass all the history to the API
        bot_response = await chat_func(history)

        history.append({"role": "assistant", "content": bot_response})

asyncio.run(continous_chat())

