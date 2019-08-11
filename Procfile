web: gunicorn app:app
worker: celery worker -A app:celery --concurrency=3 -l INFO