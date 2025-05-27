#!/bin/bash

printf "Running Tests\n\n"

./build/bin/app "$1"

printf '\n\n'
echo "Testing complete"
