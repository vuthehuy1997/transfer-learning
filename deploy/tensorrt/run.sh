python pytorch_model.py

/usr/src/tensorrt/bin/trtexec --explicitBatch \
                                --onnx=resnet50.onnx \
                                --saveEngine=resnet50.trt \
                                --minShapes=input:1x3x224x224 \
                                --optShapes=input:32x3x224x224 \
                                --maxShapes=input:32x3x224x224 \
                                --verbose \
                                --fp16

python inference.py