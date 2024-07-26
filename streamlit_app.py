import streamlit as st
import textwrap
from openai import OpenAI
import re
import os
import json


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
        user_prompt += f"\nFew shot examples: \n\n{examples}"


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},

    ]

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    if completion.choices:
        formatted_text = textwrap.fill(completion.choices[0].message.content, width=80)
        print(formatted_text)
    else:
        print("No completion found.")

    return formatted_text

def generate_example(criteria, scoring_rubric, input_variables, existing_example):
    system_prompt = """You are an AI assistant tasked with generating an example for an evaluation metric. 
    Based on the given criteria, scoring rubric, input variables, and an existing example, create a new, similar example."""

    user_prompt = f"""
    Criteria: {criteria}
    Scoring Rubric: {scoring_rubric}
    Input Variables: {input_variables}
    Existing Example: {existing_example}

    Please generate a new example in JSON format with the following fields:
    - input    
    - response
    - score
    - critique
    """

    if "reference" in input_variables:
        user_prompt += "- reference\n"
    if "context" in input_variables:
        user_prompt += "- context\n"

    user_prompt += "\nEnsure the example is similar in style but different in content from the existing example."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    if completion.choices:
        return completion.choices[0].message.content
    else:
        return None

def get_latest_example_values(metric_name, index):

    return {
        'input': st.session_state.get(f"input_{metric_name}_{index}", ""),
        'response': st.session_state.get(f"response_{metric_name}_{index}", ""),
        'score': st.session_state.get(f"score_{metric_name}_{index}", ""),
        'critique': st.session_state.get(f"critique_{metric_name}_{index}", ""),
        'reference': st.session_state.get(f"reference_{metric_name}_{index}", ""),
        'context': st.session_state.get(f"context_{metric_name}_{index}", "")
    }

def is_valid_variable_name(name):
    return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None

def clear_temp_state():
    del st.session_state.temp_prompt
    del st.session_state.editing_metric

def initialize_new_metric(metric_name):
    return {
        "criteria": "",
        "scoring_rubric": "Likert: 1 - 5",  # Default to the first option
        "input_variables": ["input", "response"],
        "prompt": "",
        "examples": [{}]
    }


reserved_metric_info = {
    "hallucination": {
        "criteria": "Assesses presence of incorrect of unrelated content in the AI’s response.",
        "scoring_rubric": "Likert: 1 - 5",
        "input_variables": ["input", "response", "reference"]
    },
    "context_relevance": {
        "criteria": "Measures how on-point the retrieved context is.",
        "scoring_rubric": "Likert: 1 - 5",
        "input_variables": ["input", "response", "context"]
    },

    "groundedness": {
        "criteria": "Determines if the response is factually based on the provided context.",
        "scoring_rubric": "Likert: 1 - 5",
        "input_variables": ["input", "response", "context"]
    },

    "precision": {
        "criteria": "Assesses the relevance of all the information in the response.",
        "scoring_rubric": "Likert: 1 - 5",
        "input_variables": ["input", "response", "reference"]
    },
    
    "logical_coherence": {
        "criteria": "Measures the logical flow, consistency, and rationality of the response.",
        "scoring_rubric": "Likert: 1 - 5",
        "input_variables": ["input", "response"]
    },
    
    "recall": {
        "criteria": "Measures how complete the response captures the key facts and details.",
        "scoring_rubric": "Likert: 1 - 5",
        "input_variables": ["input", "response", "reference"]
    }
    # Add information for other reserved metrics here
}

reserved_metrics = list(reserved_metric_info.keys())


###
st.title("Create Evaluation Metrics")

if 'custom_metrics' not in st.session_state:
    st.session_state.custom_metrics = {}

if 'show_edit_prompt' not in st.session_state:
    st.session_state.show_edit_prompt = False

# Add Eval Metrics Library to the sidebar
st.sidebar.title("Eval Metrics Library")

# Display Base Metrics
st.sidebar.subheader("Base Metrics:")
st.sidebar.text("\n".join(reserved_metrics))

# Display Custom Metrics
st.sidebar.subheader("Custom Metrics:")
custom_metrics = list(st.session_state.custom_metrics.keys())
if custom_metrics:
    st.sidebar.text("\n".join(custom_metrics))
else:
    st.sidebar.text("No custom metrics created yet.")

# Add a separator
st.sidebar.markdown("---")

#Metric name input
metric_name = st.text_input("Metric Name", key="metric_name", placeholder="Enter a name for this metric...")                

# Check if the metric name exists in custom_metrics
if metric_name in st.session_state.custom_metrics:
    # Populate fields with existing data
    metric_data = st.session_state.custom_metrics[metric_name]
    criteria = st.text_area("Criteria", value=metric_data["criteria"])
    scoring_rubric_options = ["Likert: 1 - 5", "Binary: 0 or 1", "Float: 0 - 1"]

    default_index = 0
    saved_rubric = metric_data.get("scoring_rubric", "")
    if saved_rubric in scoring_rubric_options:
        default_index = scoring_rubric_options.index(saved_rubric)
    scoring_rubric = st.selectbox("Scoring Rubric", options=scoring_rubric_options, index=default_index)

    st.write("Select Input Variables (input and response are required):")
    input_variables = st.multiselect(
        "Input variables",
        ["input", "response", "reference", "context"],
        default=metric_data["input_variables"]
    )
    
    st.subheader("Few-shot examples")

    # Button to add a new example (up to 3)
    if len(metric_data['examples']) < 3 and st.button("Add another example"):
        metric_data['examples'].append({})
        st.experimental_rerun()  # Rerun to update the selectbox options

    
    # Dropdown to select example number
    example_options = [f"Example {i}" for i in range(1, len(metric_data['examples']) + 1)]
    example_number = st.selectbox("Select example", options=example_options)

    # Extract the number from the selected option
    selected_index = int(example_number.split()[-1]) - 1

    # Display fields for the selected example
    with st.expander(example_number, expanded=True):
        example = metric_data['examples'][selected_index]
        
        example['input'] = st.text_area("Input", value=example.get('input', ''), key=f"input_{metric_name}_{selected_index}", placeholder="Enter example input here...")
        example['response'] = st.text_area("Response", value=example.get('response', ''), key=f"response_{metric_name}_{selected_index}", placeholder="Enter example response here...")

        # Conditional inputs based on selected input variables
        if "reference" in input_variables:
            example['reference'] = st.text_area("Reference", value=example.get('reference', ''), key=f"reference_{metric_name}_{selected_index}", placeholder="Enter example reference here...")
        if "context" in input_variables:
            example['context'] = st.text_area("Context", value=example.get('context', ''), key=f"context_{metric_name}_{selected_index}", placeholder="Enter example context here...")

        example['score'] = st.text_input("Score", value=example.get('score', ''), key=f"score_{metric_name}_{selected_index}", placeholder="Enter example score here...")
        example['critique'] = st.text_area("Critique", value=example.get('critique', ''), key=f"critique_{metric_name}_{selected_index}", placeholder="Enter example critique here...")

        # Update the metric data
        metric_data['examples'][selected_index] = example

    # Combine all examples to form the 'Examples' variable for the Prompt generation
    examples = ""
    for i, example in enumerate(metric_data['examples'], 1):
        examples += f"Example {i}:\n"
        examples += f"Input: {example.get('input', '')}\n\n"
        examples += f"Response: {example.get('response', '')}\n\n"
        if "reference" in input_variables:
            examples += f"Reference: {example.get('reference', '')}\n\n"
        if "context" in input_variables:
            examples += f"Context: {example.get('context', '')}\n\n"
        examples += f"Score: {example.get('score', '')}\n\n"
        examples += f"Critique: {example.get('critique', '')}\n\n"
    
    st.subheader("Edit the Generated Prompt")
    edited_prompt = st.text_area("Edit Prompt", value=metric_data["prompt"], height=300)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Deploy Metric"):
            st.session_state.custom_metrics[metric_name] = {
                "criteria": criteria,
                "scoring_rubric": scoring_rubric,
                "input_variables": input_variables,
                "prompt": edited_prompt,
                "examples": metric_data['examples']  # Save all examples
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

elif metric_name in reserved_metrics:
    # Display information for reserved metrics
    metric_info = reserved_metric_info.get(metric_name, {})
    
    st.text_area("Criteria", value=metric_info.get('criteria', 'N/A'), disabled=True)
    
    scoring_rubric_options = ["Likert: 1 - 5", "Binary: 0 or 1", "Float: 0 - 1"]
    st.selectbox("Scoring Rubric", options=scoring_rubric_options, 
                 index=scoring_rubric_options.index(metric_info.get('scoring_rubric', 'Likert: 1 - 5')), 
                 disabled=True)
    
    st.write("Select Input Variables (input and response are required):")
    st.multiselect(
        "Input variables",
        ["input", "response", "reference", "context"],
        default=metric_info.get('input_variables', ["input", "response"]),
        disabled=True
    )
    
    st.subheader("Few-shot examples")

    # Display a single, non-interactive example
    example_number = "Example 1"
    st.selectbox("Select example", options=[example_number], disabled=True)

    # Display fields for the example
    with st.expander(example_number, expanded=True):
        example = metric_info.get('examples', [{}])[0]  # Get the first (and only) example
        
        st.text_area("Input", value=example.get('input', 'N/A'), disabled=True)
        st.text_area("Response", value=example.get('response', 'N/A'), disabled=True)

        # Conditional inputs based on selected input variables
        if "reference" in metric_info.get('input_variables', []):
            st.text_area("Reference", value=example.get('reference', 'N/A'), disabled=True)
        if "context" in metric_info.get('input_variables', []):
            st.text_area("Context", value=example.get('context', 'N/A'), disabled=True)

        st.text_input("Score", value=example.get('score', 'N/A'), disabled=True)
        st.text_area("Critique", value=example.get('critique', 'N/A'), disabled=True)

    st.write("This is a base metric. You cannot modify its properties.")

else:
    # Fields for creating a new metric
    temp_metric_data = initialize_new_metric(metric_name)
    
    # Initialising temp_metric_data in the session state as well if not already
    if 'temp_metric_data' not in st.session_state: 
        st.session_state.temp_metric_data = temp_metric_data
    else:
        temp_metric_data = st.session_state.temp_metric_data

    criteria = st.text_area("Criteria", value="", placeholder="Enter criteria here...")
    scoring_rubric_options = ["Likert: 1 - 5", "Binary: 0 or 1", "Float: 0 - 1"]
    scoring_rubric = st.selectbox("Scoring Rubric", options=scoring_rubric_options, index=0)
    st.write("Select Input Variables (input and response are required):")
    input_variables = st.multiselect(
        "Input variables",
        ["input", "response", "reference", "context"],
        default=["input", "response"]
    )
    
    compulsory_selected = set(["input", "response"]).issubset(set(input_variables))
    
    if not compulsory_selected:
        st.warning("'input' and 'response' are compulsory variables and must be selected.")

    st.session_state.show_edit_prompt = False

    # Initialize examples in session state if not present
    st.subheader("Few-shot examples")

    # Check if the metric name has changed
    if 'previous_metric_name' not in st.session_state or st.session_state.previous_metric_name != metric_name:
        # Reset the number of examples
        st.session_state.selected_example = 0
        st.session_state.previous_metric_name = metric_name
        # Reset temp_metric_data
        temp_metric_data = initialize_new_metric(metric_name)
    
    temp_metric_data['examples'][0] = get_latest_example_values(metric_name, 0)
    
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = 0

    col1, col2 = st.columns(2)
    with col1:
        # Button to add a new example (up to 3)
        if len(temp_metric_data['examples']) < 3 and st.button("Add another example"):
            temp_metric_data['examples'].append({})
            st.session_state.selected_example = len(temp_metric_data['examples']) - 1
            st.session_state.temp_metric_data = temp_metric_data
            st.experimental_rerun()

    with col2:
        # Button to generate a new example (up to 3)
        if len(temp_metric_data['examples']) < 3 and st.button("Generate example"):
            try:
                with st.spinner("Generating example..."):
                    existing_example = temp_metric_data['examples'][0]
                    generated_example_json = generate_example(criteria, scoring_rubric, input_variables, str(existing_example))
                    if generated_example_json:
                        generated_example = json.loads(generated_example_json)
                        temp_metric_data['examples'].append(generated_example)
                        st.session_state.selected_example = len(temp_metric_data['examples']) - 1
                        st.session_state.temp_metric_data = temp_metric_data
                        st.success("New example generated successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to generate example.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Create example options based on the number of examples in session state
    example_options = [f"Example {i+1}" for i in range(len(temp_metric_data['examples']))]
    selected_example = st.selectbox("Select example", options=example_options, index=st.session_state.selected_example)

    # Extract the number from the selected option
    selected_index = int(selected_example.split()[-1]) - 1
    st.session_state.selected_example = selected_index
    

    with st.expander(selected_example, expanded=True):
        example = temp_metric_data['examples'][selected_index]
        
        example['input'] = st.text_area("Input", value=example.get('input', ''), key=f"input_{metric_name}_{selected_index}", placeholder="Enter example input here...")
        example['response'] = st.text_area("Response", value=example.get('response', ''), key=f"response_{metric_name}_{selected_index}", placeholder="Enter example response here...")

        # Conditional inputs based on selected input variables
        if "reference" in input_variables:
            example['reference'] = st.text_area("Reference", value=example.get('reference', ''), key=f"reference_{metric_name}_{selected_index}", placeholder="Enter example reference here...")
        if "context" in input_variables:
            example['context'] = st.text_area("Context", value=example.get('context', ''), key=f"context_{metric_name}_{selected_index}", placeholder="Enter example context here...")

        example['score'] = st.text_input("Score", value=example.get('score', ''), key=f"score_{metric_name}_{selected_index}", placeholder="Enter example score here...")
        example['critique'] = st.text_area("Critique", value=example.get('critique', ''), key=f"critique_{metric_name}_{selected_index}", placeholder="Enter example critique here...")

        # Update the temp metric data
        temp_metric_data['examples'][selected_index] = example

    # After the loop, update the session state
    st.session_state.temp_metric_data = temp_metric_data
    
    # Combine all examples to form the 'Examples' variable
    examples = ""
    for i, example in enumerate(temp_metric_data['examples'], 1):
        examples += f"Example {i}:\n"
        examples += f"Input: {example.get('input', '')}\n\n"
        examples += f"Response: {example.get('response', '')}\n\n"
        if "reference" in input_variables:
            examples += f"Reference: {example.get('reference', '')}\n\n"
        if "context" in input_variables:
            examples += f"Context: {example.get('context', '')}\n\n"
        examples += f"Score: {example.get('score', '')}\n\n"
        examples += f"Critique: {example.get('critique', '')}\n\n"

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
                    st.session_state.editing_metric = metric_name
                else:
                    st.error("Failed to generate prompt.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Edit and Save section for newly generated prompt
    if 'temp_prompt' in st.session_state and st.session_state.editing_metric == metric_name:
        st.subheader("Edit the Generated Prompt")
        edited_prompt = st.text_area("Edit Prompt", value=st.session_state.temp_prompt, height=300)
        temp_metric_data['prompt'] = edited_prompt
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Deploy Metric"):
                if not metric_name or not is_valid_variable_name(metric_name):
                    st.error("Please enter a valid metric name. It should start with a letter or underscore and contain only letters, numbers, or underscores.")
                elif metric_name.lower() in [m.lower() for m in reserved_metrics]:
                    st.error("Please choose a metric name that is not one of the Atla base metrics.")
                elif not criteria or not scoring_rubric:
                    st.error("Please fill in all required fields.")
                else:
                    st.session_state.custom_metrics[metric_name] = {
                        "criteria": criteria,
                        "scoring_rubric": scoring_rubric,
                        "input_variables": input_variables,
                        "prompt": edited_prompt,
                        "examples": temp_metric_data['examples']
                    }                            
                    st.success(f"Metric '{metric_name}' deployed successfully!")
                    clear_temp_state
                    st.experimental_rerun()
        with col2:
            if st.button("Clear"):
                del st.session_state.temp_prompt
                clear_temp_state
                st.experimental_rerun()
        with col3:
            if st.button("Regenerate Prompt"):
                try:
                    with st.spinner("Regenerating prompt..."):
                        generated_prompt = generate_prompt(input_variables, criteria, scoring_rubric, examples)
                    if generated_prompt:
                        temp_metric_data['prompt'] = generated_prompt
                        st.session_state.temp_prompt = generated_prompt
                        st.success("Prompt regenerated successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to regenerate prompt.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")