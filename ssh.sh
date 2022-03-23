#!/bin/sh

host=""
user=""

sshpass -f pass_file ssh -o StrictHostKeyChecking=no $user@$host


