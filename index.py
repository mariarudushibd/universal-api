from fastapi import FastAPI

app = FastAPI(
    title="Universal API",
    description="A universal API built with FastAPI on Vercel",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"message": "Welcome to Universal API", "Python": "on Vercel"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/hello")
def hello(name: str = "World"):
    return {"message": f"Hello, {name}!"}
