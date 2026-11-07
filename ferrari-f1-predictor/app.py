"""
WSGI entrypoint for Vercel deployment and local execution.
Exposes the Flask 'app' instance from api.app.
"""
from api.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
