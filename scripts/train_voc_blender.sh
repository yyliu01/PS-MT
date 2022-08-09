#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -l labelled_num -b backbone in use -g gpus"
   echo -e "\t-l labelled examples in experiments."
   echo -e "\t-g gpus that in use. "
   echo -e "\t-b backbone in use based on res 50 or 101"
   exit 1
}

while getopts "l:b:g:" opt; do
  case "$opt" in
    l ) labelled="$OPTARG"
          if ((labelled != 662 && labelled != 1323 && labelled != 2646 && labelled != 5291)); then
                 echo "we support the experimental setup for voc12 aug as follows:"
                 echo "
    +-------------+------------+------------+------------+------------+
    | hyper-param | 1/16 (6620) | 1/8 (13230) | 1/4 (26460) | 1/2 (52910) |
    +-------------+------------+------------+------------+------------+
    |    epoch    |     80     |     80     |     200    |     300    |
    +-------------+------------+------------+------------+------------+
    |    weight   |     1.5    |     1.5    |     1.5    |     1.5    |
    +-------------+------------+------------+------------+------------+"
              exit 1
        fi
        ;;
    b ) backbone="$OPTARG"
      if ((backbone != 50 && backbone != 101)); then
        echo "backbone should be 50 or 101"
        exit 1
      fi
      ;;
    g ) gpus="$OPTARG" ;;
    ? ) helpFunction ;;
  esac
done

if [ "${labelled}" == 2646 ]; then
  max_epochs=200
elif [ "${labelled}" == 5291 ]; then
  max_epochs=300
else
  max_epochs=80
fi
unsup_weight=1.5

nohup python3 VocCode/main.py --labeled_examples="${labelled}0" --gpus=${gpus} --backbone=${backbone} --warm_up=5 --batch_size=8 --semi_p_th=.6 --semi_n_th=.0 --learning-rate=2.5e-3 \
--epochs=${max_epochs} --unsup_weight=${unsup_weight} > voc_aug_"${labelled}"_"${backbone}".out &


