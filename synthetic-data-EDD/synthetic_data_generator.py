import os
import json
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from definitions import personas_data, scenarios_data # Import definitions

# --- Configuration ---
# Assumes OPENAI_API_KEY is set as an environment variable OR in .env
# MODEL = "gpt-4o-mini"

MODEL = "gemini-1.5-flash-latest"
NUM_QUESTIONS_PER_COMBO = 2 # How many questions per pair
BASE_PATH = os.path.dirname(__file__) # Directory of this script (scratch)
OUTPUT_DATA_PATH = os.path.join(BASE_PATH, 'data') # Subdirectory for output

# --- Load .env file (should be in project root, one level up) ---
dotenv_path = os.path.join(BASE_PATH, os.path.pardir, '.env') # Go up ONE directory
if os.path.exists(dotenv_path):
    print(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"Warning: .env file not found at {dotenv_path}. Relying on environment variables.")

# --- Use Imported Definitions (No longer hardcoded here) ---
# personas_data and scenarios_data are imported from definitions.py

# TODO: I don' think we need workshop_topic in our Agentic RAG. so you can remove this variable in here and in line 57 where it is referenced.
workshop_topic = "Building Reliable LLM Applications (RAG, evaluation, synthetic data, SDLC, prompt engineering, observability, tools like LlamaIndex/Gradio)"

# --- Ensure output directory exists (needed for questions.json) ---
os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

# --- Generate Questions ---
#print("Initializing OpenAI client...")
# OpenAI client now correctly reads from env vars loaded by dotenv or system env vars
#client = OpenAI() 

# --- With the Gemini client configuration and model initialization ---
print("Configuring Gemini client...")
# The API key is loaded from your .env file
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print(f"Initializing Gemini model: {MODEL}")
# Create an instance of the Gemini model
model = genai.GenerativeModel(MODEL)

generated_questions = []
print("Generating questions using imported definitions...")

for p_name, p_data in personas_data.items(): # Use imported data
    p_desc = p_data['description'] 
    for s_name, s_data in scenarios_data.items(): # Use imported data
        s_desc = s_data['description']
        print(f"- Generating: {p_name}/{s_name}")
        prompt = (
            f"Workshop Topic: {workshop_topic}\n"
            f"Persona: {p_name} ({p_desc})\n"
            f"Scenario: {s_name} ({s_desc})\n"
            f"Generate {NUM_QUESTIONS_PER_COMBO} distinct questions this user might ask. "
            f"List one question per line, no numbering/bullets."
        )
        # No error handling for API call
        #completion = client.chat.completions.create(
        #    model=MODEL,
        #    messages=[
        #        {"role": "system", "content": "Generate realistic user questions about a workshop based on persona/scenario.",},
        #        {"role": "user", "content": prompt}
        #    ],
        #    n=1, temperature=0.8
        #)
        #response_lines = completion.choices[0].message.content.strip().split('\n')
        
        # --- Here is the Gemini alternative ---
        # # Combine the system and user prompts for Gemini
        full_prompt = (
             "You are an expert at generating realistic user questions about a workshop based on a given persona and scenario.\n\n"
             f"{prompt}"
        )

        # Set generation configuration
        generation_config = genai.types.GenerationConfig(
           candidate_count=1,
           temperature=0.8
        )

        # Call the Gemini API
        response = model.generate_content(
           full_prompt,
           generation_config=generation_config
        )

        # Parse the response
        response_lines = response.text.strip().split('\n')

        count = 0
        for line in response_lines:
            if line.strip() and count < NUM_QUESTIONS_PER_COMBO:
                generated_questions.append({
                    "id": f"synth_{p_name}_{s_name}_{count+1}",
                    "question": line.strip(),
                    "user_type": p_name, 
                    "scenario": s_name
                })
                count += 1
        print(f"  -> Got {count} questions.")

# --- Save Generated Questions ---
# Note: Saving personas.json and scenarios.json is now handled by definitions.py when run directly
output_path = os.path.join(OUTPUT_DATA_PATH, 'questions.json') 
print(f"Saving {len(generated_questions)} questions to {output_path}...")
# No error handling for file save
with open(output_path, 'w') as f:
    json.dump(generated_questions, f, indent=2)

print("Finished.") 