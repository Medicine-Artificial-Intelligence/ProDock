import time
import pandas as pd
import gc
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import argparse


def start_driver(headless: bool = False):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://dtp.cancer.gov/RequestCompounds/forms/order.xhtml")
    return driver


def check_nci_availability(input_csv: str, output_csv: str, headless: bool = False):
    df = pd.read_csv(input_csv)
    if "NSC" not in df.columns:
        raise ValueError("Input CSV must contain a column named 'NSC'.")

    df["NCI_availability"] = 0  # default to 0 (unavailable)
    total_rows = len(df)

    # Ensure fresh file on new run
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"[i] Old file {output_csv} removed for a fresh run.")

    driver = start_driver(headless)
    wait = WebDriverWait(driver, 5)

    for idx, row in df.iterrows():
        nsc_value = str(row["NSC"])

        # Restart browser every 100 compounds
        if idx > 0 and idx % 100 == 0:
            print("[i] Restarting browser to reduce memory bloat...")
            driver.quit()
            gc.collect()
            driver = start_driver(headless)
            wait = WebDriverWait(driver, 5)

        # Reload page every 20 compounds
        elif idx > 0 and idx % 20 == 0:
            print("[i] Reloading page to refresh DOM...")
            driver.get("https://dtp.cancer.gov/RequestCompounds/forms/order.xhtml")
            time.sleep(0.1)

        try:
            print(f"[i] Processing NSC={nsc_value} ({idx + 1}/{total_rows})")

            # Step 1: Click "Vialed"
            vialed_button = wait.until(EC.element_to_be_clickable((By.ID, "orderForm:j_idt9")))
            vialed_button.click()
            time.sleep(0.1)

            # Step 2: Input NSC
            nsc_box = wait.until(EC.presence_of_element_located((By.ID, "orderForm:nsc")))
            nsc_box.clear()
            nsc_box.send_keys(nsc_value)

            # Step 3: Input amount
            amt_box = driver.find_element(By.ID, "orderForm:amt")
            amt_box.clear()
            amt_box.send_keys("1")

            # Step 4: Click "Add"
            add_button = driver.find_element(By.ID, "orderForm:j_idt28")
            add_button.click()
            time.sleep(0.1)

            # Step 5: Check for availability
            if "is not available" in driver.page_source:
                df.at[idx, "NCI_availability"] = 0
                print(f"[x] NSC={nsc_value} is not available.")
            else:
                df.at[idx, "NCI_availability"] = 1
                print(f"[✓] NSC={nsc_value} is available.")

        except Exception as e:
            print(f"[!] Error processing NSC={nsc_value}: {e}")
            df.at[idx, "NCI_availability"] = 0
        time.sleep(0.1)
        # Step 6: Reset form via button
        try:
            print("[i] Clicking Reset Form")
            reset_button = wait.until(EC.element_to_be_clickable((By.ID, "orderForm:j_idt17")))
            driver.execute_script("arguments[0].click();", reset_button)
            time.sleep(0.1)
        except Exception as e:
            print(f"[!] Could not reset form after NSC={nsc_value}: {e}")

        # Export every 100 entries (appending after first write)
        try:
            mode = 'a' if os.path.exists(output_csv) else 'w'
            header = not os.path.exists(output_csv)
            df.iloc[[idx]].to_csv(output_csv, mode=mode, header=header, index=False)
            if idx % 100 == 0 or idx == total_rows - 1:
                print(f"[i] Saved up to index {idx} into {output_csv}")
        except Exception as e:
            print(f"[!] Failed to save entry {idx}: {e}")

    driver.quit()
    print(f"[✓] Done. Final output saved to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Check NCI compound availability using Selenium.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV with 'NSC' column.")
    parser.add_argument("--output_csv", required=True, help="Path to save output CSV with 'NCI_availability' column.")
    parser.add_argument("--headless", default=False, action="store_true", help="Run browser in headless mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_nci_availability(args.input_csv, args.output_csv, headless=args.headless)