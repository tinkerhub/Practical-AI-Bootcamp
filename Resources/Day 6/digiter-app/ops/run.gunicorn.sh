#! /bin/bash

deploy_production() {
    gunicorn --bind 0.0.0.0:8000 --workers 10 --log-level debug --reload digiter.web:app
}

deploy_development() {
    gunicorn --bind 0.0.0.0:8000 --workers 2 --log-level debug --reload digiter.web:app
}

if [ $ENVIRONMENT == "production" ]; then
    deploy_production
else
    deploy_development
fi
