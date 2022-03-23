#!/bin/sh

host=""
user=""

remote_path=/disks/sdb/$user/ar-cluster-runner/ar_trainer_cluster_out

if [ -z $1 ]; then
   # No argument, exclude *_train_result_obj and *_test_result_obj files due to large size
   sshpass -f pass_file rsync --progress -e ssh -r --exclude='*_obj' $user@$host:$remote_path $PWD
elif [$1 == '-p']; then
   sshpass -f pass_file rsync --progress --compress -r -e ssh $user@$host:$remote_path $PWD
else
   echo "Usage:ar_trainer_copy [options]"
   echo ""
   echo "-p	include pickles such as SARIMAXResults object saved to files"
fi

