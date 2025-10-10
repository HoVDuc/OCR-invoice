import streamlit as st

class History:
    @staticmethod
    def render():
        st.title("📜 History")
        st.write("Here is the history of your uploaded invoices.")
        # Example history data
        history_data = [
            {"filename": "invoice1.pdf", "date": "2023-10-01", "status": "Processed"},
            {"filename": "invoice2.jpg", "date": "2023-10-05", "status": "Pending"},
            {"filename": "invoice3.png", "date": "2023-10-10", "status": "Processed"},
        ]
        
        for record in history_data:
            st.write(f"**{record['filename']}** - {record['date']} - Status: {record['status']}")
        # You can add more detailed history and functionalities here

if __name__ == "__main__":
    st.write("# Invoice History Page")
    History.render()