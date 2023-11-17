#!/bin/sh

sudo docker rm -f $(sudo docker ps -a -q)