# Dockerfile for quant_cy_npu on 910B environment
FROM swr.cn-north-4.myhuaweicloud.com/ascendhub/ascend-pytorch:23.0.RC1-ubuntu20.04-py3.8

# Set environment variables
ENV ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
ENV PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH

# Install additional dependencies
RUN pip3 install --upgrade pip && \
    pip3 install numpy pybind11

# Set working directory
WORKDIR /workspace/quant_cy_npu

# Copy source code
COPY . .

# Make build script executable
RUN chmod +x build.sh test_quant_cy_npu.py

# Build the extension
RUN ./build.sh

# Test the installation
RUN python3 test_quant_cy_npu.py

# Default command
CMD ["/bin/bash"]
