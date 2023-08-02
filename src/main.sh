#!/bin/bash

echo "$0 is not updated, we are sorry"
exit 0

print_usage(){
    echo "usate: $0 [options]"
    echo "Options:"
    echo "  -h, --help          View help message."
    echo "  -t, --train         train PPO."
    echo "  -i, --inference     inference PPO."
    echo "  -c, --config <path> Specify custom configuration file path."
    echo "  --log-path <path>   Specify custom log path (for training only)."
    echo "  --ckpt-path <path>  Specify custom checkpoint path (for training only)."
    echo "  --render                Enable rendering (for training and testing)."
    echo "  --process-image         Enable resnet18 processing (for training and testing)."
}

if [ $# -eq 0 ]; then
    echo "Error: No parameters == do nothing ..."
    print_usage
    exit 1
fi

script="train.py"
config_file="/home/user/src/configs/CartPole-v1.json"
log_path="/home/user/PPO_logs"
ckpt_path="/home/user/PPO_preTrained" 
training_selected=false
render=false
process_image=false

while getopts ":htic:" opt; do
    case $opt in
        h)
            print_usage
            exit 0
            ;;
        t)
            script="train.py"
            ;;
        i)
            script="test.py"
            ;;
        c)
            config_file="$OPTARG"
            ;;
        -)
            # Handle long options
            case $OPTARG in
                help)
                    print_usage
                    exit 0
                    ;;
                log-path)
                    log_path="$2"
                    shift
                    ;;
                ckpt-path)
                    ckpt_path="$2"
                    shift
                    ;;
                render)
                    render=true
                    ;;
                process-image)
                    process_image=true
                    ;;
                *)
                    echo "Error: Invalid option --$OPTARG"
                    print_usage
                    exit 1
                    ;;
            esac
            ;;
        \?)
            echo "Error: Invalid option -$OPTARG"
            print_usage
            exit 1
            ;;
        :)
            echo "Error: Option -$OPTARG requires an argument."
            print_usage
            exit 1
            ;;
    esac
done

# Shift the parsed options, so that $1 points to the first non-option argument
shift $((OPTIND - 1))

# Check if a script was selected
if [ -z "$script" ]; then
    echo "Error: No script option selected (-t/--train or -i/--inference)."
    print_usage
    exit 1
fi

# Construct the command based on whether training is selected or not
command="$script $config_file "
command+="--log-path "$log_path" " 
command+="--ckpt-path "$ckpt_path" "

if [ $render ]; then
    command+="--render "
fi

if [ $process_image ]; then
    command+="--process-image "
fi

echo "Running: $command"
python $command