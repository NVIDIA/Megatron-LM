#!/bin/bash
mkdir -p ~/.ngc
mkdir -p ~/.cache
mkdir -p ~/.ssh
[ ! -f ~/.netrc ] && touch ~/.netrc
exit 0