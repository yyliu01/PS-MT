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
          if ((labelled != 372 && labelled != 744 && labelled != 1488 )); then
                 echo "we support the experimental setup for cityscapes as follows:"
                 echo "
    +-------------+------------+------------+------------+
    | hyper-param | 1/8 (372)  | 1/4 (744)  | 1/2 (1488) |
    +-------------+------------+------------+------------+
    |    epoch    |     320    |     450    |     550    |
    +-------------+------------+------------+------------+
    |    weight   |     3.0    |     3.0    |     3.0    |
    +-------------+------------+------------+------------+"
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

if [ "${labelled}" == 744 ]; then
  max_epochs=450
elif [ "${labelled}" == 1488 ]; then
  max_epochs=550
else
  max_epochs=320
fi


nohup python3 CityCode/main.py --labeled_examples="${labelled}" --gpus=${gpus} --backbone=${backbone} --warm_up=5 --batch_size=4 --semi_p_th=.6 --semi_n_th=.0 \
--epochs=${max_epochs} > city_"${labelled}"_"${backbone}".out &


