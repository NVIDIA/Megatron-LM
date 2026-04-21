TEXT_HOME=/lustre/fsw/portfolios/llmservice/users/ravirajj/datasets/llama3_1/fake_acad/data/tokens-shuffle/text
CODE_HOME=/lustre/fsw/portfolios/llmservice/users/ravirajj/datasets/llama3_1/fake_acad/data/tokens-shuffle/code

FACAD="${TEXT_HOME}/fake_acad_docs_text_document"
BEIGEK="${TEXT_HOME}/beige-kingfisher_text_document"

C_0_CODE="${CODE_HOME}/C_shuf_0_content_document"
C_1_CODE="${CODE_HOME}/C_shuf_1_content_document"
C_2_CODE="${CODE_HOME}/C_shuf_2_content_document"
C_3_CODE="${CODE_HOME}/C_shuf_3_content_document"
C_4_CODE="${CODE_HOME}/C_shuf_4_content_document"
C_5_CODE="${CODE_HOME}/C_shuf_5_content_document"
C_6_CODE="${CODE_HOME}/C_shuf_6_content_document"
C_7_CODE="${CODE_HOME}/C_shuf_7_content_document"
C_8_CODE="${CODE_HOME}/C_shuf_8_content_document"
C_9_CODE="${CODE_HOME}/C_shuf_9_content_document"

CPP_0_CODE="${CODE_HOME}/C++_shuf_0_content_document"
CPP_1_CODE="${CODE_HOME}/C++_shuf_1_content_document"
CPP_2_CODE="${CODE_HOME}/C++_shuf_2_content_document"
CPP_3_CODE="${CODE_HOME}/C++_shuf_3_content_document"
CPP_4_CODE="${CODE_HOME}/C++_shuf_4_content_document"
CPP_5_CODE="${CODE_HOME}/C++_shuf_5_content_document"
CPP_6_CODE="${CODE_HOME}/C++_shuf_6_content_document"
CPP_7_CODE="${CODE_HOME}/C++_shuf_7_content_document"
CPP_8_CODE="${CODE_HOME}/C++_shuf_8_content_document"
CPP_9_CODE="${CODE_HOME}/C++_shuf_9_content_document"

JAVA_0_CODE="${CODE_HOME}/Java_shuf_0_content_document"
JAVA_1_CODE="${CODE_HOME}/Java_shuf_1_content_document"
JAVA_2_CODE="${CODE_HOME}/Java_shuf_2_content_document"
JAVA_3_CODE="${CODE_HOME}/Java_shuf_3_content_document"
JAVA_4_CODE="${CODE_HOME}/Java_shuf_4_content_document"
JAVA_5_CODE="${CODE_HOME}/Java_shuf_5_content_document"
JAVA_6_CODE="${CODE_HOME}/Java_shuf_6_content_document"
JAVA_7_CODE="${CODE_HOME}/Java_shuf_7_content_document"
JAVA_8_CODE="${CODE_HOME}/Java_shuf_8_content_document"
JAVA_9_CODE="${CODE_HOME}/Java_shuf_9_content_document"

JS_0_CODE="${CODE_HOME}/JavaScript_shuf_0_content_document"
JS_1_CODE="${CODE_HOME}/JavaScript_shuf_1_content_document"
JS_2_CODE="${CODE_HOME}/JavaScript_shuf_2_content_document"
JS_3_CODE="${CODE_HOME}/JavaScript_shuf_3_content_document"
JS_4_CODE="${CODE_HOME}/JavaScript_shuf_4_content_document"
JS_5_CODE="${CODE_HOME}/JavaScript_shuf_5_content_document"
JS_6_CODE="${CODE_HOME}/JavaScript_shuf_6_content_document"
JS_7_CODE="${CODE_HOME}/JavaScript_shuf_7_content_document"
JS_8_CODE="${CODE_HOME}/JavaScript_shuf_8_content_document"
JS_9_CODE="${CODE_HOME}/JavaScript_shuf_9_content_document"

PYTHON_0_CODE="${CODE_HOME}/Python_shuf_0_content_document"
PYTHON_1_CODE="${CODE_HOME}/Python_shuf_1_content_document"
PYTHON_2_CODE="${CODE_HOME}/Python_shuf_2_content_document"
PYTHON_3_CODE="${CODE_HOME}/Python_shuf_3_content_document"
PYTHON_4_CODE="${CODE_HOME}/Python_shuf_4_content_document"
PYTHON_5_CODE="${CODE_HOME}/Python_shuf_5_content_document"
PYTHON_6_CODE="${CODE_HOME}/Python_shuf_6_content_document"
PYTHON_7_CODE="${CODE_HOME}/Python_shuf_7_content_document"
PYTHON_8_CODE="${CODE_HOME}/Python_shuf_8_content_document"
PYTHON_9_CODE="${CODE_HOME}/Python_shuf_9_content_document"

# 0.033 ${C_CODE} \
# 0.033 ${CPP_CODE} \
# 0.034 ${JAVA_CODE} \
# 0.034 ${JS_CODE} \
# 0.083 ${PYTHON_CODE} \

DATA_BLEND="0.740 ${FACAD} \
0.043 ${BEIGEK} \
0.0033 ${C_0_CODE} \
0.0033 ${C_1_CODE} \
0.0033 ${C_2_CODE} \
0.0033 ${C_3_CODE} \
0.0033 ${C_4_CODE} \
0.0033 ${C_5_CODE} \
0.0033 ${C_6_CODE} \
0.0033 ${C_7_CODE} \
0.0033 ${C_8_CODE} \
0.0033 ${C_9_CODE} \
0.0033 ${CPP_0_CODE} \
0.0033 ${CPP_1_CODE} \
0.0033 ${CPP_2_CODE} \
0.0033 ${CPP_3_CODE} \
0.0033 ${CPP_4_CODE} \
0.0033 ${CPP_5_CODE} \
0.0033 ${CPP_6_CODE} \
0.0033 ${CPP_7_CODE} \
0.0033 ${CPP_8_CODE} \
0.0033 ${CPP_9_CODE} \
0.0034 ${JAVA_0_CODE} \
0.0034 ${JAVA_1_CODE} \
0.0034 ${JAVA_2_CODE} \
0.0034 ${JAVA_3_CODE} \
0.0034 ${JAVA_4_CODE} \
0.0034 ${JAVA_5_CODE} \
0.0034 ${JAVA_6_CODE} \
0.0034 ${JAVA_7_CODE} \
0.0034 ${JAVA_8_CODE} \
0.0034 ${JAVA_9_CODE} \
0.0034 ${JS_0_CODE} \
0.0034 ${JS_1_CODE} \
0.0034 ${JS_2_CODE} \
0.0034 ${JS_3_CODE} \
0.0034 ${JS_4_CODE} \
0.0034 ${JS_5_CODE} \
0.0034 ${JS_6_CODE} \
0.0034 ${JS_7_CODE} \
0.0034 ${JS_8_CODE} \
0.0034 ${JS_9_CODE} \
0.0083 ${PYTHON_0_CODE} \
0.0083 ${PYTHON_1_CODE} \
0.0083 ${PYTHON_2_CODE} \
0.0083 ${PYTHON_3_CODE} \
0.0083 ${PYTHON_4_CODE} \
0.0083 ${PYTHON_5_CODE} \
0.0083 ${PYTHON_6_CODE} \
0.0083 ${PYTHON_7_CODE} \
0.0083 ${PYTHON_8_CODE} \
0.0083 ${PYTHON_9_CODE}"