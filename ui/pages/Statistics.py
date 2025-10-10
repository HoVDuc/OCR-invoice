import streamlit as st

class Statistics:
    @staticmethod
    def render():
        st.title("📊 Statistics")
        st.write("Here are some statistics about your invoices.")
        # Example statistics
        st.metric(label="Total Invoices", value="150")
        st.metric(label="Processed Invoices", value="120")
        st.metric(label="Pending Invoices", value="30")
        # You can add more detailed statistics and visualizations here

if __name__ == "__main__":
    st.write("# Invoice Statistics Page")
    Statistics.render()