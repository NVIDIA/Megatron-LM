pushd $(dirname "${BASH_SOURCE[0]}")

set -e
for i in $(ls 0_test_*.sh); do
    echo "Running $i"
    bash $i
done
for i in $(ls 1_test_*.sh); do
    echo "Running $i"
    bash $i
done
