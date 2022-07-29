python pytorch_model.py
###
# git clone --recursive --branch v2020.1 https://github.com/inducer/pycuda.git
# cd pycuda
# python configure.py --cuda-root=/usr/local/cuda-10.2
# pip install -e .
###

/usr/src/tensorrt/bin/trtexec --explicitBatch \
                                --onnx=resnet50.onnx \
                                --saveEngine=resnet50.trt \
                                --minShapes=input:1x3x224x224 \
                                --optShapes=input:32x3x224x224 \
                                --maxShapes=input:32x3x224x224 \
                                --verbose \
                                --fp16

python inference.py