web: gunicorn app:app
worker: celery worker -A app:celery --concurrency=5 -l INFO