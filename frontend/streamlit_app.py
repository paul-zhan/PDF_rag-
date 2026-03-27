import streamlit as st
import requests

st.title("PDF RAG Bot")
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query:
        try:
            response = requests.post(
                "http://backend:8000/chatbot_answer",
                json={"query": query},
                timeout=60
            )

            if response.status_code == 200:
                answer = response.json().get("answer", "No answer found.")
                st.write(f"Answer: {answer}")
            else:
                st.write(f"Backend error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            st.write(f"Request failed: {e}")
    else:
        st.write("Please enter a query.")