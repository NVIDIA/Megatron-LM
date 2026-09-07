#!/usr/bin/env bash
set -euo pipefail

readonly SEARCH_ROOTS=(docker docs examples)
readonly ESCAPED_CHECKSUM='\${YQ_SHA256}'
readonly CHECKSUM_COMMAND='echo "${YQ_SHA256}  '

mapfile -t checksum_files < <(/usr/bin/grep -R -l -F 'YQ_SHA256=' "${SEARCH_ROOTS[@]}")
if [[ ${#checksum_files[@]} -eq 0 ]]; then
  echo "No dynamic yq checksum definitions found" >&2
  exit 1
fi

for file in "${checksum_files[@]}"; do
  if /usr/bin/grep -q -F "$ESCAPED_CHECKSUM" "$file"; then
    echo "Escaped yq checksum variable in $file" >&2
    exit 1
  fi
  if ! /usr/bin/grep -q -F "$CHECKSUM_COMMAND" "$file"; then
    echo "Missing dynamic yq checksum verification in $file" >&2
    exit 1
  fi
done

temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT
printf payload >"$temporary_dir/yq"
checksum=$(sha256sum "$temporary_dir/yq" | awk '{print $1}')

assert_checksum_command_expands() {
  local file=$1
  local installed_path=$2
  local command

  command=$(/usr/bin/grep -F "$CHECKSUM_COMMAND" "$file")
  command="${command#"${command%%[![:space:]]*}"}"
  command=${command/"$installed_path"/"$temporary_dir/yq"}
  YQ_SHA256="$checksum" bash -eu -o pipefail -c "$command"
}

assert_checksum_command_expands \
  docker/Dockerfile.ci.dev \
  /usr/local/bin/yq
assert_checksum_command_expands \
  examples/moe_recipes/deepseek_v3/b200/mxfp8_256GPU_TP1PP8EP32.yaml \
  /usr/bin/yq
