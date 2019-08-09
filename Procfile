web: gunicorn app:app --preload
worker: celery worker -A tasks.app -l INFO