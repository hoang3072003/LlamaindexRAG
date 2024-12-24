from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import streamlit as st
import time
# Function to generate image using Selenium
def generate_image(prompt):
    """ When calling the function, pass the prompt and wait 40 seconds for the image to be generated. """
    # Đường dẫn tới ChromeDriver (sửa lại nếu cần)
    driver_path = r"C:\Users\Legion\Downloads\Win_x64_1361142_chromedriver_win32\chromedriver_win32\chromedriver.exe"
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)

    try:
        # Mở trang web
        url = "https://deepai.org/machine-learning-model/text2img"
        driver.get(url)

        # Đợi trang tải và nhập prompt
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea.model-input-text-input.dynamic-border"))
        )
        input_box = driver.find_element(By.CSS_SELECTOR, "textarea.model-input-text-input.dynamic-border")

        # Xóa dữ liệu cũ (nếu có) và nhập dữ liệu mới
        input_box.clear()
        input_box.send_keys(prompt)  # Nhập prompt từ người dùng

        # Nhấn nút "Generate"
        generate_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Generate')]"))
        )
        generate_button.click()
        time.sleep(5)
        # Kiểm tra trạng thái và đợi ảnh được tạo
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div#place_holder_picture_model img"))
        )
        
        # Lấy link ảnh
        generated_image = driver.find_element(By.CSS_SELECTOR, "div#place_holder_picture_model img")
        image_src = generated_image.get_attribute("src")
        return image_src
    finally:
        # Đóng trình duyệt
        driver.quit()
