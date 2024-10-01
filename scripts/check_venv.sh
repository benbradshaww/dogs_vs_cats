#!/bin/bash

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not active"
    exit 1
else
    echo "Virtual environment active: $VIRTUAL_ENV"
fi
