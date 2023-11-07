pushd $(dirname "${BASH_SOURCE[0]}")

set -e
bash test_pp_1f1b_exact.sh
bash test_pp_zb2p_exact.sh
bash test_pp_interleaved_1f1b_exact.sh
bash test_pp_zbv_exact.sh
bash test_pp_1f1b.sh
bash test_pp_zb2p.sh
bash test_pp_zbv.sh
bash test_local_ddp_pp_1f1b.sh
bash test_local_ddp_pp_zb2p.sh
bash test_local_ddp_pp_zbv.sh