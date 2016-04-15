while true; do sleep 1s; cat /proc/meminfo | grep MemFree >> RAMusage.txt; sleep 1m; done
