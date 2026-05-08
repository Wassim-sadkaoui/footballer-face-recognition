from ddgs import DDGS
import requests
import os

players = []
for player in players:
    folder = f"dataset/{player}"
    os.makedirs(folder, exist_ok=True)

    with DDGS() as ddgs:
        results = ddgs.images(f"{player} face", max_results=20)

        for i, r in enumerate(results):
            try:
                img_url = r["image"]
                img_data = requests.get(img_url, timeout=10).content

                with open(f"{folder}/{i}.jpg", "wb") as f:
                    f.write(img_data)

            except:
                pass

print("Dataset ready")