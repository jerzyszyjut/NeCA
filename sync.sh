#!/bin/bash
source .cred

NAME=neca

echo "$LOGIN@$IP"

tempfile=$(mktemp)
git ls-files --cached --others --exclude-standard > "$tempfile"
rsync -a --update --progress --files-from="$tempfile" ./ "$LOGIN@$IP:$NAME/"
