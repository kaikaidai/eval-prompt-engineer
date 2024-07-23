import streamlit as st
import textwrap
from openai import OpenAI
import re
import os


openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_prompt(input_variables, criteria, scoring_rubric, examples=None):

    if set(input_variables) == {'input', 'response', 'reference'}:
        system_prompt = """Please create an evaluation prompt based on the user specified criteria and scoring rubric (and additionally some examples if defined in the user prompt). The scoring rubric in the prompt should always be as the user defines it (e.g., the score range should be 1 - 3 if the user defines it as such rather than the 1 - 5 in the example below). You can use the below template as a guide for how it should be set up:
                '### **Precision and Conciseness Evaluation:**
                **Example Format**

                  QUESTION:

                  REFERENCE RESPONSE:

                  AI ASSISTANT ANSWER:

                  —

                  SCORE:

                  CRITIQUE:

                - **Objective: Assess responses from any AI model on a scale from 1 (worst) to 5 (best), evaluating the relevance, accuracy, directness, and conciseness of the response in direct comparison to a provided reference response..
                - **Process:**
                - **Analyze:** Rigorously evaluate the AI's response solely against the reference response for its correctness, relevance, directness, and conciseness.
                - **Scoring:**
                    - **1 (Poor):** The response is significantly inaccurate or irrelevant compared to the reference, includes much extraneous information, or fails to adhere to the format and essence of the reference.
                    - **2 (Fair):** The response, while showing some alignment with the reference, includes notable inaccuracies or irrelevant content and lacks brevity.
                    - **3 (Good):** The response aligns fairly well with the reference but includes slight inaccuracies or unnecessary details that affect its overall precision and conciseness.
                    - **4 (Very Good):** The response is almost fully aligned with the reference in terms of precision, relevance, and conciseness, with only minor discrepancies.
                    - **5 (Excellent):** The response perfectly matches the reference in all aspects—precision, relevance, directness, and conciseness—without any deviations or errors.
                - **Feedback:**
                    - Provide a concise justification for the assigned score, focusing strictly on how well the AI's response mirrors the reference in terms of precision, relevance, directness, and conciseness. Highlight even minor discrepancies in high-scoring responses to justify not achieving a perfect score.

                - **Examples for guidance:**
                [Few-shot examples given by the user]'"""
    elif set(input_variables) == {'input', 'response', 'context'}:
        system_prompt = """Please create an evaluation prompt based on the user specified criteria and scoring rubric (and additionally some examples if defined in the user prompt). The scoring rubric in the prompt should always be as the user defines it (e.g., the score range should be 1 - 3 if the user defines it as such rather than the 1 - 5 in the example below). You can use the below template as a guide for how it should be set up:

                ### **Contextual Groundedness Evaluation**

                **Example Format**

                QUESTION:

                CONTEXT:

                AI ASSISTANT ANSWER:

                —

                SCORE:

                CRITIQUE:

                You are an evaluator scoring responses from an AI assistant. You are provided with a question, the specific context it pertains to, and the AI's answer. Score the AI’s answer from 1 (worst) to 5 (best), focusing on how well it is grounded in the given context.

                **Evaluation Objective**: This evaluation measures the degree to which the AI's responses are grounded in the context provided. It assesses the relevance and adherence to the specifics of the context, highlighting the importance of accurately reflecting the nuances and details presented.

                **Process**:

                1. **Review**: Start by thoroughly examining the AI's response in relation to the context provided. Evaluate how effectively the response incorporates and aligns with key aspects of the context, enhancing its relevance and accuracy.
                2. **Scoring**:
                    - **1 (Poor)**: The response shows a significant disregard for or misinterpretation of the context, missing key details.
                    - **2 (Fair)**: The response demonstrates some connection to the context but includes notable inaccuracies or omissions.
                    - **3 (Good)**: The response is mostly consistent with the context, with only minor errors or misalignments.
                    - **4 (Very Good)**: The response is well-aligned with the context, showing a deep understanding and minimal discrepancies.
                    - **5 (Excellent)**: Represents an exemplary standard of perfect contextual adherence, which, while challenging to achieve, serves as a goal for absolute precision in contextual grounding.
                3. **Feedback**: Provide focused feedback on how well the response adheres to the context. Identify specific instances where the AI effectively used or failed to use contextual cues, discussing the impact of these cues on the accuracy and relevance of the answer. Keep the feedback succinct and targeted, directly addressing the response's strengths and areas for improvement in terms of contextual groundedness.

                - **Examples for guidance:**
                [Few-shot examples given by the user]"""
    elif set(input_variables) == {'input', 'response'}:
        system_prompt = """Please create an evaluation prompt based on the user specified criteria and scoring rubric (and additionally some examples if defined in the user prompt). The scoring rubric in the prompt should always be as the user defines it (e.g., the score range should be 1 - 3 if the user defines it as such rather than the 1 - 5 in the example below). You can use the below template as a guide for how it should be set up:

                ### **Formattings:**

                Return a score for the number of different formattings present in the model response. Recognized formattings include: list (numbered or bullet pointed), a markdown table, unformatted text, or a legal memorandum. Text that directly follows a markdown table and refers to its content should be considered as an extension of the markdown table formatting. For instance, the explanatory text following the markdown table about missing information from the table (e.g., "Please note that some information such as the names of the CEOs for CHARANGA SL, the registered office for NEPTUNE GETAFE PROPCO, S.L.U., and the capital figure for CHARANGA SL were not provided in the attachments.") should be counted as part of the markdown table and not classified as unformatted text. Conversely, text completely unrelated to the content of the markdown table or legal memorandum should be considered separately as unformatted text. If a list appears within a legal memorandum, count it as part of the legal memorandum format and not as a separate formatting type.

                **Process:**

                **Analyze:** Examine the AI's response to determine how many types of formatting are used, according to the specific formatting criteria outlined above.

                **Scoring:**

                - Allocate one point for each distinct type of formatting identified. The score should only be 0 if the response is empty — even a single character of text otherwise warrants a point.

                **Feedback:**

                - Provide feedback listing the different formats detected in the response, in order of appearance. For example: ['table', 'unformatted text']. The types of formatting used should be listed separated by commas, and the string returned should be wrapped in square brackets.

                - **Examples for guidance:**
                [Few-shot examples given by the user]"""
        
    elif set(input_variables) == {"input", "response", "context", "reference"}:
        system_prompt = """Please create an evaluation prompt based on the user specified criteria and scoring rubric (and additionally some examples if defined in the user prompt). The scoring rubric in the prompt should always be as the user defines it (e.g., the score range should be 1 - 3 if the user defines it as such rather than the 1 - 5 in the example below). You can use the below template as a guide for how it should be set up:
        '### Hallucination Detection Evaluation:
        Example Format

        QUESTION:

        CONTEXT:

        REFERENCE RESPONSE:

        AI ASSISTANT ANSWER:

        —

        SCORE:

        CRITIQUE:

        **Objective: Identify any claims in the AI response that are not supported by either the context or the reference response. Score responses from 1 (worst) to 3 (best).
        Process:
        Analyze: Carefully compare the AI's response against both the context and the reference response to detect unsupported claims (hallucinations).
        Scoring:
        1 (Poor): The response includes significant hallucinations not supported by the context or reference response.
        2 (Fair): The response has some minor hallucinations, but the overall message aligns with the context and reference response.
        3 (Good): The response contains no hallucinations and is fully supported by the context and reference response.
        Feedback:
        
        Provide a concise justification for the assigned score, highlighting any hallucinations and explaining why they affect the response's accuracy and relevance.

        Examples for guidance:
        [Few-shot examples given by the user]"""

    user_prompt = f"Evaluation criteria: {criteria}\nScoring rubric: {scoring_rubric}"

    if examples:
        user_prompt += f"\nFew shot examples: {examples}"


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},

    ]

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    if completion.choices:
        formatted_text = textwrap.fill(completion.choices[0].message.content, width=80)
        print(formatted_text)
    else:
        print("No completion found.")

    return formatted_text

def is_valid_variable_name(name):
    return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None

# List of reserved metric names
reserved_metrics = [
    "hallucination", "context_relevance", "recall", "precision",
    "logical_coherence", "recall_gpt", "atla_groundedness", "atla_precision", "atla_recall"
]

if 'custom_metrics' not in st.session_state:
    st.session_state.custom_metrics = {}

###
st.title("Evaluation Prompt Generator")


# Create two columns for metric name and suggestions
col1, col2 = st.columns([2, 1])

with col1:
    metric_name = st.text_input("Metric Name", placeholder="Enter a name for this metric...")

with col2:
    # Autocomplete suggestions
    suggestions = list(st.session_state.custom_metrics.keys())
    if metric_name:
        matching_suggestions = [s for s in suggestions if metric_name.lower() in s.lower()]
        if matching_suggestions:
            selected_suggestion = st.selectbox("Did you mean?", [""] + matching_suggestions)
            if selected_suggestion:
                metric_name = selected_suggestion
                
                

# Check if the metric name exists in custom_metrics
if metric_name in st.session_state.custom_metrics:
    # Populate fields with existing data
    metric_data = st.session_state.custom_metrics[metric_name]
    criteria = st.text_area("Criteria", value=metric_data["criteria"])
    scoring_rubric = st.text_input("Scoring Rubric", value=metric_data["scoring_rubric"])
    st.write("Select Input Variables (input and response are compulsory):")
    input_variables = st.multiselect(
        "Input variables",
        ["input", "response", "reference", "context"],
        default=metric_data["input_variables"]
    )
    examples = st.text_area("Examples", value=metric_data["examples"], height=250)
    st.subheader("Edit the Generated Prompt")
    edited_prompt = st.text_area("Edit Prompt", value=metric_data["prompt"], height=300)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save Changes"):
            st.session_state.custom_metrics[metric_name] = {
                "criteria": criteria,
                "scoring_rubric": scoring_rubric,
                "input_variables": input_variables,
                "examples": examples,
                "prompt": edited_prompt
            }
            st.success("Changes saved successfully!")
    with col2:
        if st.button("Delete Metric"):
            del st.session_state.custom_metrics[metric_name]
            st.success(f"Metric '{metric_name}' deleted successfully!")
            st.experimental_rerun()
    with col3:
        if st.button("Regenerate Prompt"):
            try:
                with st.spinner("Regenerating prompt..."):
                    generated_prompt = generate_prompt(input_variables, criteria, scoring_rubric, examples)
                if generated_prompt:
                    st.session_state.custom_metrics[metric_name]["prompt"] = generated_prompt
                    st.success("Prompt regenerated successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Failed to regenerate prompt.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

else:
    # Fields for creating a new metric
    criteria = st.text_area("Criteria", placeholder="Enter criteria here...")
    scoring_rubric = st.text_input("Scoring Rubric", placeholder="Enter scoring rubric here...")
    
    st.write("Select Input Variables (input and response are compulsory):")
    input_variables = st.multiselect(
        "Input variables",
        ["input", "response", "reference", "context"],
        default=["input", "response"]
    )
    
    compulsory_selected = set(["input", "response"]).issubset(set(input_variables))
    
    if not compulsory_selected:
        st.warning("'input' and 'response' are compulsory variables and must be selected.")
    
    examples = st.text_area("Examples", placeholder="Enter examples here...", height=250)

    if st.button("Generate Prompt", disabled=not compulsory_selected):
        if not metric_name or not is_valid_variable_name(metric_name):
            st.error("Please enter a valid metric name. It should start with a letter or underscore and contain only letters, numbers, or underscores.")
        elif metric_name.lower() in [m.lower() for m in reserved_metrics]:
            st.error("Please choose a metric name that is not one of the Atla base metrics.")
        elif not criteria or not scoring_rubric:
            st.error("Please fill in all required fields.")
        else:
            try:
                with st.spinner("Generating prompt..."):
                    generated_prompt = generate_prompt(input_variables, criteria, scoring_rubric, examples)
                if generated_prompt:
                    st.success("Prompt generated successfully!")
                    st.session_state.temp_prompt = generated_prompt
                    
                    # Store the inputs
                    st.session_state.custom_metrics[metric_name] = {
                        "criteria": criteria,
                        "scoring_rubric": scoring_rubric,
                        "input_variables": input_variables,
                        "examples": examples,
                        "prompt": generated_prompt
                    }
                    
                    st.experimental_rerun()
                else:
                    st.error("Failed to generate prompt.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Edit and Save section for newly generated prompt
    if 'temp_prompt' in st.session_state:
        st.subheader("Edit the Generated Prompt")
        edited_prompt = st.text_area("Edit Prompt", value=st.session_state.temp_prompt, height=300)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Prompt"):
                st.session_state.custom_metrics[metric_name]["prompt"] = edited_prompt
                del st.session_state.temp_prompt
                st.success(f"Prompt for '{metric_name}' saved successfully!")
                st.experimental_rerun()
        with col2:
            if st.button("Regenerate Prompt"):
                try:
                    with st.spinner("Regenerating prompt..."):
                        generated_prompt = generate_prompt(input_variables, criteria, scoring_rubric, examples)
                    if generated_prompt:
                        st.session_state.temp_prompt = generated_prompt
                        st.success("Prompt regenerated successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to regenerate prompt.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")