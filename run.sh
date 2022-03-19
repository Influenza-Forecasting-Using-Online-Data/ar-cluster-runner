#!/bin/sh

remote_path=/disks/sdb/$user/ar-cluster-runner/ar_trainer_cluster_out

usage () {
  echo "Usage: run <trainer> | <autoarima>"
  echo ""
  echo "trainer		run ar_cluster_trainer.py detached"
  echo "autoarima	run ar_autoarima.py detached"
}

if [ -z $1 ]; then
   # No argument
   usage
elif [$1 == 'trainer']; then
   nohup python ar_trainer_cluster.py > trainer.out 2>&1 &
elif [$1 == 'autoarima']
   nohup python ar_autoarima.py > autoarima.out 2>&1 &
else
   usage
fi


