import requests
from PIL import Image
from io import BytesIO

url = "https://avatars.abstractapi.com/v1/"
api_key = "38a6608f645a41ebb9f5749f3e37a88a"
name = "Claire Florentz"

params = {
    "api_key": api_key,
    "name": name
}

response = requests.get(url, params=params)

if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    image.show()  # Display the image using the default image viewer
else:
    print("Error retrieving avatar image:", response.status_code)
