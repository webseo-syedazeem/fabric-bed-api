#!/usr/bin/env bash
gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:${PORT:-8000} app:app
