web: gunicorn app:app
worker: celery worker -A app:celery --concurrency=2 -l INFO