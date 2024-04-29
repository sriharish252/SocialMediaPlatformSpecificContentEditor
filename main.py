import streamlit as st
from socialMedia_crewAI import get_modified_content

st.title("Social Media Platform Specific Content Generator")

# Get content for user
content = st.text_area("Content: ")

if content and len(content)>0:
    # If content is populated, get modified content from LLMs
    response = get_modified_content(content)

    # Display the response
    for key, value in response.items():
        st.header(key)
        st.write(value)

