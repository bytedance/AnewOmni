# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

QDB="$1"
TDB="$2"
RES="$3"
# create link to data file which contains a list of all targets that should be aligned
ln -s "${TDB}.index" "${RES}"
# create new index repeatedly pointing to same entry
INDEX_SIZE="$(echo $(wc -c < "${TDB}.index"))"
awk -v size=$INDEX_SIZE '{ print $1"\t0\t"size; }' "${QDB}.index" > "${RES}.index"
# create dbtype (7)
awk 'BEGIN { printf("%c%c%c%c",7,0,0,0); exit; }' > "${RES}.dbtype"
