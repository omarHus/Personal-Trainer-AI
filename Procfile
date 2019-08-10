web: gunicorn app:app
worker: celery worker -A tasks.app -l INFO