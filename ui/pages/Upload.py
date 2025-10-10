import requests
import streamlit as st
import pandas as pd
from loguru import logger

class Upload:

    def extract_image(self, uploaded_file, model_name, prompt, check_return_raw=False, check_save_result=True):
        url = "http://localhost:8000/extract" 
        files = {
            "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
        }

        data = {
            "model_name": model_name,
            "prompt": prompt,
            "return_raw": str(check_return_raw).lower(),
            "save_result": str(check_save_result).lower()   
        }
        response = requests.post(url, files=files, data=data)
        return response

    
    @staticmethod
    def render():
        uploader = Upload()
        st.write("Please upload your invoice file.")

        response = None
        with st.form("upload_form"):
            model_name = st.selectbox("Select Model", ["models/gemini-flash-lite-latest",
                                                       "models/gemini-2.5-flash-lite", 
                                                       "models/gemini-2.5-pro", 
                                                       "models/gemini-flash-latest",
                                                       "models/gemini-flash-lite-latest",
                                                       "models/gemini-2.5-flash",
                                                       "models/gemini-2.5-flash-lite"])
            prompt = st.text_area("Prompt", value="Hãy trích xuất dữ liệu hóa đơn thành JSON")
            uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])
            return_raw = st.checkbox("Return Raw", value=False)
            save_result = st.checkbox("Save Result", value=True)
            submit_button = st.form_submit_button("Upload and Extract")
            if uploaded_file and submit_button:
                st.success("File uploaded successfully!")
                response = uploader.extract_image(uploaded_file, model_name, prompt, return_raw, save_result)
        if response:
            data = response.json()
            if response.status_code != 200 or "error" in data:
                st.error(f"Error: {data.get('error', 'Unknown error occurred')}")
                logger.error(f"Error: {data.get('error', 'Unknown error occurred')}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file, caption="Uploaded Invoice")
                with col2:
                    st.title(data.get("message", "Extraction Successful"))
                    data = data.get("data", {})
                    st.divider()
                    st.write(f"🛒 Hóa đơn từ: {data['SELLER']}")
                    st.write(f"🕒 Thời gian: {data['TIMESTAMP']}")
                    # Write address if exists
                    if data.get("ADDRESS"):
                        st.write(f"🏠 Địa chỉ: {data['ADDRESS']}")

                    # Construct product table
                    st.divider()
                    product_table = [
                        [p["PRODUCT"], p["NUM"], f"{p['VALUE']:,} ₫"]
                        for p in data["PRODUCTS"]   
                    ]
                    df = pd.DataFrame(product_table, columns=["Sản phẩm", "Số lượng", "Giá"])
                    st.table(df)
                    st.divider()
                    # Total cost
                    st.write(f"💰 TỔNG CỘNG: {data['TOTAL_COST']:,} ₫")
                    st.divider()
                    with st.expander("View Extracted Data Json format", expanded=False):
                        st.write(data)

                logger.info(f"Response from server: {response.text}")

if __name__ == "__main__":
    st.write("# Invoice Extraction App")
    Upload.render()
