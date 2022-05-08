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
          if ((labelled != 92 && labelled != 183 && labelled != 366 && labelled != 732)); then
                 echo "we support the experimental setup for voc12 aug as follows:"
                 echo "
                  +-------------+-----------+-----------+-----------+-----------+
                  | hyper-param | 1/16 (92) | 1/8 (183) | 1/4 (366) | 1/2 (732) |
                  +-------------+-----------+-----------+-----------+-----------+
                  |    epoch    |     80    |     80    |    80     |     80    |
                  +-------------+-----------+-----------+-----------+-----------+
                  |    weight   |    0.06   |    0.6    |    0.6    |    0.6    |
                  +-------------+-----------+-----------+-----------+-----------+"
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

if [ "${labelled}" == 92 ]; then
  unsup_weight=0.06
else
  unsup_weight=0.6
fi

max_epochs=80

nohup python3 VocCode/main.py --labeled_examples="${labelled}" --gpus=${gpus} --backbone=${backbone} --warm_up=5 --batch_size=8 --semi_p_th=.6 --semi_n_th=.0 --learning-rate=2.5e-3 \
--epochs=${max_epochs} --unsup_weight=${unsup_weight} > voc_hq_"${labelled}"_"${backbone}".out &
