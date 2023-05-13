import streamlit as st
from streamlit_chat import message
import time
from app import llm_chain

st.set_page_config(page_title="NDIS Chatbot", 
                   page_icon=":pixel-art-neutral:",
                   layout="centered", 
                   initial_sidebar_state="auto")
col1, col2 = st.columns([1,1])

col1.markdown(" # NDIS Chatbot ")
col1.markdown(" ##### Ask me anything about NDIS ")
col2.markdown(" Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat ")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

placeholder = st.empty() 
def get_text():
    input_text = st.text_input("You: ", value="", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = llm_chain(user_input)
    with st.spinner('Wait for it...'):
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["answer"])
        time.sleep(2)


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
