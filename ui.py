import streamlit as st
import requests

# FastAPI base URL
base_url = "http://localhost:8000"

st.title("RAG APP: Retrieval-Augmented Generation with Groq LLM")

# Initialize session states for each step and conversation history
if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "collection_name" not in st.session_state:
    st.session_state.collection_name = ""

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "n_results" not in st.session_state:
    st.session_state.n_results = 2  # Default value

# Move Step 1 (Initialization) to the sidebar if successful
with st.sidebar:
    st.header("Steps")
    if st.session_state.llm_initialized:
        st.markdown("**Step 1: LLM Initialized**")
        st.text_input("Groq API Key", value=st.session_state.api_key, disabled=True)
        st.text_input("Model Name", value=st.session_state.model_name, disabled=True)
        
    if st.session_state.file_uploaded:
        st.markdown("**Step 2: File Uploaded**")
        st.text("Collection Name: " + st.session_state.collection_name)

# Step 1: Initialize Groq API (Appears on main screen initially, moves to sidebar after success)
if not st.session_state.llm_initialized:
    st.header("Step 1: Initialize Groq API")
    with st.form("init_llm"):
        model_name = st.text_input("Model Name", value="llama3-8b-8192")
        api_key = st.text_input("Groq API Key", value="sk-xxxx", type="password")
        initialize = st.form_submit_button("Initialize")

    if initialize:
        init_response = requests.get(f"{base_url}/init_llm", params={"model_name": model_name, "api_key": api_key})
        if init_response.status_code == 200:
            st.session_state.llm_initialized = True
            st.session_state.model_name = model_name
            st.session_state.api_key = api_key
            st.success(f"LLM Initialized: {model_name}")
            st.experimental_rerun()
        else:
            st.error(f"Error: {init_response.json()['message']}")



# Step 2: Upload file (only after initialization, on the main screen)
elif st.session_state.llm_initialized and not st.session_state.file_uploaded:
    st.header("Step 2: Upload PDF/HTML File")
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Choose a PDF or HTML file", type=["pdf", "html"])
        collection_name = st.text_input("Collection Name", value="test_collection")
        upload = st.form_submit_button("Upload")

    if upload and uploaded_file is not None:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        data = {"collection_name": collection_name}
        upload_response = requests.post(f"{base_url}/upload", files=files, data=data)

        if upload_response.status_code == 200:
            st.session_state.file_uploaded = True
            st.session_state.collection_name = collection_name
            # st.success(f"File '{uploaded_file.name}' uploaded successfully")
            st.write(upload_response.json())
            # st.experimental_rerun()
        else:
            st.error(f"Error: {upload_response.json()['message']}")

# Step 3: Chat with LLM (only after file is uploaded)
elif st.session_state.file_uploaded:
    st.header("Chat with Groq LLM")

    # Show the conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input area for the user query and parameters
    if not st.session_state.collection_name:
        st.session_state.collection_name = st.text_input("Enter collection name")
    
    # Only show the number of results input if it hasn't been set
    if not st.session_state.n_results:
        st.session_state.n_results = st.number_input("Number of results", min_value=1, max_value=10, value=2)


    query = st.chat_input("Enter your query")
    # Handle the query submission and interaction with the LLM
    if query:
        # Store the user's query
        st.session_state.conversation_history.append({"role": "user", "content": query})

        # Display the user's message in the chat
        with st.chat_message("user"):
            st.markdown(query)

        # Prepare the query parameters
        query_params = {
            "query": query,
            "n_results": st.session_state.n_results,
            "collection_name": st.session_state.collection_name
        }

        # Send the request to the LLM
        query_response = requests.get(f"{base_url}/query", params=query_params)

        if query_response.status_code == 200:
            response_data = query_response.json()
            llm_response = response_data['llm_output']

            # Store the assistant's response in session state
            st.session_state.conversation_history.append({"role": "assistant", "content": llm_response['text']})

            # Display the assistant's response in the chat
            with st.chat_message("assistant"):
                st.markdown(llm_response['text'])
        else:
            st.error(f"Error: {query_response.json()['message']}")
