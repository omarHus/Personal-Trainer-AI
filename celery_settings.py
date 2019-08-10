import os

CELERY_TASK_SERIALIZER = 'json'
BROKER_URL             = os.environ.get('REDISTOGO_URL','redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT  = ['json']
CELERY_RESULT_BACKEND  = os.environ.get('REDISTOGO_URL','redis://localhost:6379/0')
BROKER_POOL_LIMIT      = 0