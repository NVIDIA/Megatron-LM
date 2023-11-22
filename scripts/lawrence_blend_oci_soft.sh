#!/bin/bash

set -u

if [ "$#" = 0 ]; then
    ENG_DATA_HOME="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/data/843m/english"
elif [ "$#" = 1 ]; then
    ENG_DATA_HOME=$1
else
    echo "specialize for $# args."
    exitt 1
fi


#english datasets
# ENG_DATA_HOME="/lustre/fsw/adlr/adlr-nlp/mpatwary/data/multilingual/multi-1.1t-gtc/english"
# ENG_DATA_HOME="/lustre/fsw/adlr/adlr-nlp/lmcafee/retro/data"
# ENG_DATA_HOME="/lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english"
B3="${ENG_DATA_HOME}/MTNLG/Books3_shuf_text_document"
OWT2="${ENG_DATA_HOME}/MTNLG/OpenWebText2_shuf_text_document"
SE="${ENG_DATA_HOME}/MTNLG/StackExchange_shuf_text_document"
PM="${ENG_DATA_HOME}/MTNLG/PubMedAbs_shuf_text_document"
WIK="${ENG_DATA_HOME}/MTNLG/Wikipedia_shuf_text_document"
GUT="${ENG_DATA_HOME}/MTNLG/Gutenberg_shuf_text_document"
BC2="${ENG_DATA_HOME}/MTNLG/BookCorpus2_shuf_text_document"
NIH="${ENG_DATA_HOME}/MTNLG/NIHExporter_shuf_text_document"
ARX="${ENG_DATA_HOME}/MTNLG/ArXiv_shuf_text_document"
ST="${ENG_DATA_HOME}/MTNLG/Stories_shuf_text_document"
BIGSC="${ENG_DATA_HOME}/BigScience/BigScience_shuf_text_document"
REDDIT="${ENG_DATA_HOME}/Reddit-Plus/Reddit_all_dialogue_shuf_text_document"
# RN="${ENG_DATA_HOME}/MTNLG/RealNews_shuf_text_document"
CCNEWS="${ENG_DATA_HOME}/CC-NEWS/CC-NEWS_shuf_text_document"
PCC="${ENG_DATA_HOME}/MTNLG/Pile-CC_shuf_text_document"
CC202050="${ENG_DATA_HOME}/CC-MAIN-2020-50/CC-MAIN-2020-50_shuf_text_document"
CC202240_0="${ENG_DATA_HOME}/CC-MAIN-2022-40/CC-MAIN-2022-40_00_shuf_text_document"
CC202240_1="${ENG_DATA_HOME}/CC-MAIN-2022-40/CC-MAIN-2022-40_01_shuf_text_document"
CC201935="${ENG_DATA_HOME}/CC-MAIN-2019-35/CC-MAIN-2019-35_shuf_text_document"
CC202104="${ENG_DATA_HOME}/MTNLG/CC-2021-04_shuf_text_document"
MC4="${ENG_DATA_HOME}/mc4-en_1T-url/mc4-en_shuf_text_document"

DATA_BLEND=" \
0.01920	${B3} \
0.01602	${OWT2} \
0.00751	${SE} \
0.00324	${PM} \
0.00653	${WIK} \
0.00193	${GUT} \
0.00117	${BC2} \
0.00023	${NIH} \
0.01143	${ARX} \
0.00366	${ST} \
0.03992	${BIGSC} \
0.04768	${REDDIT} \
0.07199	${CCNEWS} \
0.02180	${PCC} \
0.07633	${CC202050} \
0.07644	${CC202240_0} \
0.07644	${CC202240_1} \
0.09414	${CC201935} \
0.03890	${CC202104} \
0.08544	${MC4} \
"

# eof
