web: gunicorn app:app
worker: celery worker -A app:celery -l INFO