import os

CELERY_TASK_SERIALIZER       = 'json'
BROKER_URL                   = os.environ.get('REDISTOGO_URL','redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT        = ['json']
CELERY_RESULT_BACKEND        = os.environ.get('REDISTOGO_URL','redis://localhost:6379/0')
BROKER_POOL_LIMIT            = None
CELERY_REDIS_MAX_CONNECTIONS = 20
CELERYD_WORKER_LOST_WAIT     = 20
CELERYD_MAX_TASKS_PER_CHILD  = 6