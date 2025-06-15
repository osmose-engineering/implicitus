# ai_adapter/main.py
from ai_adapter.adapter import generate_model

def main():
    prompt = "create a unit sphere"
    model = generate_model(prompt)
    print(model)

if __name__ == "__main__":
    main()