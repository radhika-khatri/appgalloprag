from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()  # Or Firefox, Edge, etc.
driver.get(LOGIN_URL)  # Replace with your actual login page URL

wait = WebDriverWait(driver, 20)  # Increased wait time for reliability

# Fill in credentials (update selectors as needed)
username_field = wait.until(EC.presence_of_element_located((By.NAME, 'username')))
username_field.send_keys('YOUR_USERNAME')  # Replace with your username

password_field = wait.until(EC.presence_of_element_located((By.NAME, 'password')))
password_field.send_keys('YOUR_PASSWORD')  # Replace with your password

# Click the login button (update selector as needed)
login_button = wait.until(EC.element_to_be_clickable((By.ID, 'login_btn')))
login_button.click()

# Debugging: Save page source and print title after login attempt
with open("debug_after_login.html", "w", encoding="utf-8") as f:
    f.write(driver.page_source)
print("Current page title after login:", driver.title)

# Wait for a unique dashboard element by its ID (replace 'preloader-active' with actual dashboard element ID)
wait.until(EC.presence_of_element_located((By.ID, 'preloader-active')))

print("Login successful. Dashboard loaded.")

# Manually navigate to the target URL after login
driver.get(TARGET_URL)  # Replace with the page you want to scrape

# Optional: Wait for a unique element on the target page to ensure it has loaded
wait.until(EC.presence_of_element_located((By.ID, 'setup-your-account')))

# Scrape the HTML
html = driver.page_source

# Save HTML
with open("desired_page2.html", "w", encoding="utf-8") as f:
    f.write(html)

driver.quit()
