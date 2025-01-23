import yaml
from fastapi import FastAPI

app = FastAPI()


@app.post("/")
def main(input: str):
    model = lambda x: x[::-1]
    return model(input)

    if __name__ == "__main__":
        app()
