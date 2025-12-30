#!/usr/bin/env python3
"""
Minimal LLM-as-a-judge for RAG Evaluation

A simple script that uses GPT-4 or Gemini to evaluate RAG responses using existing labeled data as examples.
"""

import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini client
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Could not configure Gemini: {e}")

def load_context_docs(context_dir):
    """Loads all .md and .txt files from the context directory."""
    context_text = ""
    if not context_dir or not os.path.exists(context_dir):
        return context_text
    
    for filename in os.listdir(context_dir):
        if filename.endswith(('.md', '.txt')):
            file_path = os.path.join(context_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    context_text += f"\n--- Source: {filename} ---\n"
                    context_text += f.read() + "\n"
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return context_text

def generate_context_summary(context_text, judge_model="openai"):
    """Generates a brief summary of the available context to define the judge's scope."""
    if not context_text:
        return "No internal documentation provided."
        
    prompt = f"""Summarize the following internal documentation in 3-5 bullet points. 
Focus on what topics are covered (e.g., local setup, cloud provisioning, specific projects).
This summary will be used to define the 'Scope of Knowledge' for an evaluation judge.

{context_text[:5000]} # Limit context for summary
"""
    
    if judge_model == "gemini":
        try:
            model = genai.GenerativeModel('gemini-3-flash-preview')
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            return "Summary unavailable (API error)."
    else:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception:
            return "Summary unavailable (API error)."

def evaluate_rag_response(question, new_response, reference_context, scope_summary, good_examples, bad_examples, judge_model="openai"):
    """Use an LLM to evaluate a RAG response using sophisticated prompt engineering."""
    
    system_prompt = f"""You are an Expert Technical Auditor for Enterprise Internal Systems. 
Your task is to evaluate RAG (Retrieval-Augmented Generation) responses based strictly on provided internal documentation.

### Scope of Knowledge:
{scope_summary}

### Evaluation Criteria (RAG Triad):
1. **Groundedness (Faithfulness)**: Every claim in the response must be directly supported by the Reference Context. If a detail is correct in the real world but NOT in the provided context, it must be flagged as a hallucination.
2. **Relevance**: The response must directly address all parts of the user's question.

### Instructions:
- Adopt a critical, analytical tone.
- Perform a step-by-step "Chain-of-Thought" analysis before reaching a verdict.
- Ignore your pre-trained public knowledge. Use ONLY the Reference Context.
"""

    user_prompt = f"""### User Question:
{question}

### Reference Context (Ground Truth):
{reference_context if reference_context else "NO CONTEXT PROVIDED"}

### System Response to Evaluate:
{new_response}

"""
    
    if good_examples or bad_examples:
        user_prompt += "### Reference Examples for Style:\n"
        if good_examples:
            for ex in good_examples:
                res = ex['response'][0] if isinstance(ex['response'], list) else str(ex['response'])
                user_prompt += f"Good Response: {res}\nReason: {ex.get('reason', 'N/A')}\n\n"
        if bad_examples:
            for ex in bad_examples:
                res = ex['response'][0] if isinstance(ex['response'], list) else str(ex['response'])
                user_prompt += f"Bad Response: {res}\nReason: {ex.get('reason', 'N/A')}\n\n"

    user_prompt += """
### Required Output Format:
Thought: (Analyze groundedness and relevance step-by-step. Compare the response to the Reference Context claim-by-claim.)
Judgment: "+1" (Pass) or "-1" (Fail)
Reason: (Brief summary of the thought process)
"""

    # Call the selected LLM
    if judge_model == "gemini":
        try:
            model = genai.GenerativeModel('gemini-3-flash-preview')
            content = model.generate_content(system_prompt + "\n\n" + user_prompt).text
        except Exception as e:
            return "error", f"Gemini API failed: {str(e)}", ""
    else:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content
        except Exception as e:
            return "error", f"OpenAI API failed: {str(e)}", ""
    
    # Parse the result
    judgment = "unknown"
    if 'Judgment: "+1"' in content or 'Judgment: +1' in content:
        judgment = "pass"
    elif 'Judgment: "-1"' in content or 'Judgment: -1' in content:
        judgment = "fail"
    
    # Extract thought and reason
    thought = ""
    if "Thought:" in content and "Judgment:" in content:
        thought = content.split("Thought:")[1].split("Judgment:")[0].strip()
        
    reason = "No reason provided"
    if "Reason:" in content:
        reason = content.split("Reason:")[1].strip()
    
    return judgment, reason, thought

def main():
    """Run a simple test of the LLM judge."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate RAG responses using an LLM judge.")
    parser.add_argument("--input-file", required=True, help="Path to the JSON file containing responses to evaluate.")
    parser.add_argument("--examples-file", default="data/evaluated_responses_20250328_190348.json",
                        help="Path to the JSON file containing labeled examples for few-shot learning.")
    parser.add_argument("--context-dir", help="Directory containing internal documentation for grounding.")
    parser.add_argument("--limit", type=int, help="Limit evaluation to the first N responses.")
    parser.add_argument("--output-prefix", default="llm_evaluated",
                        help="Prefix for the timestamped output file and the _all.json file.")
    parser.add_argument("--judge-model", default="openai", choices=["openai", "gemini"],
                        help="The LLM to use as the judge.")
    args = parser.parse_args()

    print(f"Using {args.judge_model.upper()} as the judge.")
    
    # Load context if directory provided
    reference_context = ""
    scope_summary = "General company knowledge."
    if args.context_dir:
        print(f"Loading context from {args.context_dir}...")
        reference_context = load_context_docs(args.context_dir)
        if reference_context:
            print(f"Generating scope summary using {args.judge_model}...")
            scope_summary = generate_context_summary(reference_context, judge_model=args.judge_model)
            print(f"Scope Summary: {scope_summary[:100]}...")
        else:
            print(f"Warning: No valid context files found in {args.context_dir}")

    print("Loading evaluation data...")
    
    # Load examples (for few-shot learning)
    try:
        with open(args.examples_file, 'r') as f:
            examples = json.load(f)
        print(f"Loaded {len(examples)} examples from {args.examples_file}")
    except Exception as e:
        print(f"Error loading examples file {args.examples_file}: {e}")
        print("Proceeding without few-shot examples.")
        examples = []

    # Load responses to evaluate
    try:
        with open(args.input_file, 'r') as f:
            to_evaluate = json.load(f)
        print(f"Loaded {len(to_evaluate)} responses from {args.input_file}")
    except Exception as e:
        print(f"Error loading input file {args.input_file}: {e}")
        return

    # Apply limit
    if args.limit and 0 < args.limit < len(to_evaluate):
        to_evaluate = to_evaluate[:args.limit]
        print(f"LIMIT MODE: Evaluating only the first {len(to_evaluate)} responses.")
    
    good_examples = [ex for ex in examples if ex.get('judgment') == 'pass'][:1]
    bad_examples = [ex for ex in examples if ex.get('judgment') == 'fail'][:1]

    # Evaluate
    results = []
    total = len(to_evaluate)
    print(f"Starting evaluation of {total} responses...")
    
    for i, item in enumerate(to_evaluate):
        print(f"\nEvaluating response {i+1}/{total}...")
        
        question = item['question']
        response_text = item['response'][0] if isinstance(item['response'], list) else str(item['response'])
        
        print(f"Question: {question[:50]}...")
        
        # Evaluate with prompt engineering
        judgment, reason, thought = evaluate_rag_response(
            question, 
            response_text, 
            reference_context,
            scope_summary,
            good_examples, 
            bad_examples,
            judge_model=args.judge_model
        )
        
        print(f"Judgment: {judgment}")
        print(f"Reason: {reason[:100]}...")
        
        # Save result
        item['judgment'] = judgment
        item['reason'] = reason
        item['thought'] = thought # Store the step-by-step analysis
        item['evaluation_type'] = f'llm-{args.judge_model}-pe' # pe for Prompt Engineering
        item['context_provided'] = bool(reference_context)
        results.append(item)
        
        time.sleep(1)
    
    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(args.input_file)
    base_prefix = os.path.basename(args.output_prefix)
    
    output_filename = os.path.join(output_dir, f"{base_prefix}_{timestamp}.json")
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    all_output_filename = os.path.join(output_dir, f"{base_prefix}_all.json")
    with open(all_output_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_filename}")
    print(f"All-view file updated: {all_output_filename}")

if __name__ == "__main__":
    main()