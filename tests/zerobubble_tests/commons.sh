function setup() {
  pushd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
  mkdir -p test_logs
  set -e
  # Model setup
  export LAYERS=14
  export HIDDEN_SIZE=4096
  export ATTENTION_HEADS=32
  export MICRO_BATCH_SIZE=1
  
  export EXIT_INTERVAL=300

  export NCCL_DEBUG=INFO
}

function log() {
  cat "test_logs/$AIP_RUN_NAME"
}
function launch() {
  if [ -z "$NO_RUN" ]; then
    bash examples/pretrain_llama_7b.sh $@ 2>&1 | tee test_logs/$AIP_RUN_NAME
  fi
  # echo ''
}

function check_eq() {
  # echo "Checking $1 == $2"
  if [ "$1" != "$2" ]; then
    echo "Error: $1 != $2"
    exit 1
  fi
}

function check_near() {
  # echo "Checking $1 == $2"
  diff=$(bc -l <<< "scale=10; if ($1 >= $2) ($1-$2)/$2 else ($2-$1)/$1")
  if (( $(echo "$diff >= 0.001" | bc -l) )); then
    echo "Error: $1 not near $2"
    exit 1
  fi

}

function loss_of() {
  cat test_logs/$1 | grep ' 300/' | awk '{print $27}'
}
function check_loss() {
  # fbw="$(cat $logsecondlast| grep ILP | tail -n 1 | cut -d ' ' -f 6,7,8 --output-delimiter=',')"
  # o="$(cat $logsecondlast | grep 'optimizer time' | cut -d ' ' -f 12 | bash median.sh)"
  # t="$(cat $loglast  | grep 'elapsed time per itera' | tail -n 10 | awk '{print $14}' | bash median.sh)"
  # l="$(cat $loglast | grep ' 100/' | awk '{print $27}')"
  check_eq "$(loss_of $AIP_RUN_NAME)" "$1"
  # l100="$(cat $loglast | grep  ' [1-9]0/ .*lm loss' | awk '{print $27}' | md5sum | cut -d ' ' -f 1)"
  # mmin="$(cat $logall | grep 'max allocated' | awk '{print $14}' | sort -n | tail -n 1)"
  # mmax="$(cat $logall | grep 'max allocated' | awk '{print $14}' | sort -n | head -n 1)"
}

function check_loss_near() {
  # fbw="$(cat $logsecondlast| grep ILP | tail -n 1 | cut -d ' ' -f 6,7,8 --output-delimiter=',')"
  # o="$(cat $logsecondlast | grep 'optimizer time' | cut -d ' ' -f 12 | bash median.sh)"
  # t="$(cat $loglast  | grep 'elapsed time per itera' | tail -n 10 | awk '{print $14}' | bash median.sh)"
  # l="$(cat $loglast | grep ' 100/' | awk '{print $27}')"
  check_near "$(loss_of $AIP_RUN_NAME)" "$1"
  # l100="$(cat $loglast | grep  ' [1-9]0/ .*lm loss' | awk '{print $27}' | md5sum | cut -d ' ' -f 1)"
  # mmin="$(cat $logall | grep 'max allocated' | awk '{print $14}' | sort -n | tail -n 1)"
  # mmax="$(cat $logall | grep 'max allocated' | awk '{print $14}' | sort -n | head -n 1)"
}