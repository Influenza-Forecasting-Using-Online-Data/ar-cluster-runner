#!/bin/sh

host=""
user=""

remote_path=/disks/sdb/$user/ar-cluster-runner/output_autoarima.txt

sshpass -f pass_file scp $user@$host:$remote_path $PWD
