from run_app import app, init_pipeline, processing_loop
import threading

# Initialize pipeline and start background processing thread
init_pipeline()
thread = threading.Thread(target=processing_loop, daemon=True)
thread.start()

# WSGI application for servers like gunicorn
application = app
