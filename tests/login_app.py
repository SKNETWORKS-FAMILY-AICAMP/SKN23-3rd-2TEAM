import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Login", layout="centered")
st.title("Login")

if "api_session" not in st.session_state:
    st.session_state.api_session = requests.Session()

api = st.session_state.api_session
st.write(api.cookies.get_dict())


def fetch_me() -> dict | None:
    try:
        response = api.get(f"{API_URL}/auth/me", timeout=10)
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None

    if response.status_code == 200:
        return response.json()
    return None


me = fetch_me()

if not me:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        try:
            response = api.post(
                f"{API_URL}/auth/login",
                data={"username": username, "password": password},
                timeout=10,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")
        else:
            if response.status_code == 200:
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Login failed: check username/password")

    st.info("Use sidebar page '1 Signup' to create an account.")
    st.page_link("pages/1_signup.py", label="Go to Signup")
else:
    st.write("Current user:", me["username"])
    if st.button("Logout"):
        try:
            api.post(f"{API_URL}/auth/logout", timeout=10, allow_redirects=False)
        except requests.RequestException as exc:
            st.error(f"Logout request failed: {exc}")
        st.session_state.api_session = requests.Session()
        st.rerun()
