from main import app
from uvicorn import Config, Server

class WSGIAdapter:
    def __init__(self, asgi_app):
        self.app = asgi_app

    def __call__(self, environ, start_response):
        server = Server(Config(self.app, loop="none", http="none"))
        server.run()

wsgi_app = WSGIAdapter(app)

