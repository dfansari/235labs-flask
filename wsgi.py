"""Application entry point."""
from labsAppFramework import init_app

app = init_app()

if __name__ == "__main__":
    app.run(host="localhost")
