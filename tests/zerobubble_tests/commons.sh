function setup() {
  cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
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
    bash examples/pretrain_zero_bubble.sh $@ 2>&1 | tee test_logs/$AIP_RUN_NAME
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
  check_loss_exists
  if (( $(echo "$diff >= 0.0015" | bc -l) )); then
    echo "Error: $1 not near $2"
    exit 1
  fi

}

function loss_of() {
  cat test_logs/$1 | grep ' 300/' | cut -d '/' -f 2 | awk '{print $25}'
}
function check_loss() {
  check_eq "$(loss_of $AIP_RUN_NAME)" "$1"
}

function check_validation_same() {
  a=$(cat test_logs/$1 | grep 'validation loss' | md5sum)
  b=$(cat test_logs/$AIP_RUN_NAME | grep 'validation loss' | md5sum)
  check_eq "$a" "$b"
}
function check_loss_near() {
  check_near "$(loss_of $AIP_RUN_NAME)" "$1"
}

function check_loss_exists() {
  if [ -z "$(loss_of $AIP_RUN_NAME)" ]; then
    echo "Error: loss not found"
    exit 1
  fi
}