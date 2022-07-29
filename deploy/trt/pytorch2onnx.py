import argparse
import os
import numpy as np

import cv2
import onnx
import onnxruntime
import torch
from model.network.cnn import CNNModel



def main(weight):
    #Load pytorch model ------------------------------------------------------------------------------------------------
    ckpt = torch.load(weight)
    config = ckpt['config']
    config['device'] = 'cuda:0'

    # Network
    net = CNNModel(
        fe_name=config['model']['cnn']['module'], version=config['model']['cnn']['version'],
        feature_extract=config['model']['cnn']['feature_extract'], pretrained=config['model']['cnn']['pretrained'],
        number_class=config['data']['num_classes'], drop_p=config['regularization']['dropout']).to(config['device'])

    net.load_state_dict(ckpt['model'])
    print('Finished loading model!')
    # print(net)
    
    net = net.to(config['device'])
    net.eval()

    # convert to ONNX --------------------------------------------------------------------------------------------------
    ONNX_FILE_PATH = os.path.splitext(weight)[0] + '.onnx'
    img = torch.rand(6, 3, config['model']['dataset']['max_height'], config['model']['dataset']['max_width']).to(config['device'])
    output = net(img)
    
    torch.onnx.export(net, 
        img, 
        ONNX_FILE_PATH, 
        export_params=True, 
        opset_version=12, 
        do_constant_folding=True,  
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 
                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)

    print("Model was successfully converted to ONNX format.")
    print("It was saved to", ONNX_FILE_PATH)

    session = onnxruntime.InferenceSession(ONNX_FILE_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input = {session.get_inputs()[0].name: np.array(img.cpu())}
    print('output pytorch: ', output)
    print('output pytorch: ', output.cpu().detach().numpy().shape)
    output = session.run(None, input)
    print('output onnxruntime: ', output)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt')
    args = parser.parse_args()
    main(args.ckpt)