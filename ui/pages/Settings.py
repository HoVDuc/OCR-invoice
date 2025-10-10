import streamlit as st 

class Settings:
    @staticmethod
    def render():
        st.title("⚙️ Settings")
        st.write("Adjust your application settings here.")
        # Example settings options
        theme = st.selectbox("Select Theme", ["Light", "Dark"])
        notifications = st.checkbox("Enable Notifications", value=True)
        st.write(f"Theme selected: {theme}")
        st.write(f"Notifications enabled: {notifications}")
        # You can add more settings options here

if __name__ == "__main__":
    st.write("# Settings Page")
    Settings.render()