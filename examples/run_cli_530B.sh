#!/bin/bash
echo "Loading model and starting server.  May take several minutes"
./run_api_server_530B.sh
STATUS = 1
while [ $STATUS -eq 1]
do
	sleep 20
	curl -s -m 20 'http://localhost:5000/generate' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"sentences":["Test2"], "max_len":30}' | head -n 1 | grep "HTTP/1.[01] [23].." > /dev/null
	STATUS = $?
done
python tools/run_cli.py 'http://localhost:5000/generate' 
