import google.generativeai as genai
genai.configure(api_key="AIzaSyA4qLjDDBz29iTlyFAzGWIHBl3oMZ48D5E")
for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
