from parse_model_outputs import Parser
import cv2
import numpy as np
import onnxruntime
import pickle
import os
import time




SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
def slice_outputs(model_outputs: np.ndarray) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    if SEND_RAW_PRED:
      parsed_model_outputs['raw_pred'] = model_outputs.copy()
    return parsed_model_outputs


# 读取两张彩色图片
image1 = cv2.imread("1.png")
image2 = cv2.imread("2.png")

# 调整图片大小为 (256, 512)
image1_resized = cv2.resize(image1, (512, 256))
image2_resized = cv2.resize(image2, (512, 256))

# 转换为YUV格式
yuv_image1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2YUV)
yuv_image2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2YUV)

# 提取Y、U、V通道
y1, u1, v1 = cv2.split(yuv_image1)
y2, u2, v2 = cv2.split(yuv_image2)

# 将U、V通道的大小调整为 (128, 256)，与Y通道一致
u1_resized = cv2.resize(u1, (256, 128))
v1_resized = cv2.resize(v1, (256, 128))
u2_resized = cv2.resize(u2, (256, 128))
v2_resized = cv2.resize(v2, (256, 128))

# 合并6个通道
yuv_tensor = np.zeros((1, 12, 128, 256), dtype=np.float16)

# Full-res Y channel
yuv_tensor[0, 0, :, :] = y1[::2, ::2]  # Channel 0
yuv_tensor[0, 1, :, :] = y1[::2, 1::2]  # Channel 1
yuv_tensor[0, 2, :, :] = y1[1::2, ::2]  # Channel 2
yuv_tensor[0, 3, :, :] = y1[1::2, 1::2]  # Channel 3

# Half-res U and V channels
yuv_tensor[0, 4, :, :] = u1_resized  # Channel 4
yuv_tensor[0, 5, :, :] = v1_resized  # Channel 5

# Full-res Y channel from second image
yuv_tensor[0, 6, :, :] = y2[::2, ::2]  # Channel 6
yuv_tensor[0, 7, :, :] = y2[::2, 1::2]  # Channel 7
yuv_tensor[0, 8, :, :] = y2[1::2, ::2]  # Channel 8
yuv_tensor[0, 9, :, :] = y2[1::2, 1::2]  # Channel 9

# Half-res U and V channels from second image
yuv_tensor[0, 10, :, :] = u2_resized  # Channel 10
yuv_tensor[0, 11, :, :] = v2_resized  # Channel 11

# 转换为float16类型
yuv_tensor = yuv_tensor.astype(np.float16)

print('image的type:',yuv_tensor.shape)  # 输出形状信息

desire= np.zeros((1, 100, 8),dtype=np.float16)
print('image的type:',yuv_tensor.shape)
traffic_convention= np.array([[1, 0]],dtype=np.float16)
print('traffic_convention的type:',traffic_convention.shape)

lateral_control_params=np.array([[7.6,0.3]],dtype=np.float16)

print('lateral_control_params的type:',lateral_control_params.shape)
prev_desired_curv=np.zeros((1, 100, 1),dtype=np.float16)

nav_features= np.zeros((1, 256),dtype=np.float16)
nav_instructions=np.zeros((1, 150),dtype=np.float16)
features_buffer=np.zeros((1, 99, 512),dtype=np.float16)

ort_session = onnxruntime.InferenceSession("supercombo.onnx",providers=[ 'CUDAExecutionProvider'])
for i in range(0,50):
    time_start=time.time()

    ort_inputs = {'input_imgs': yuv_tensor,'big_input_imgs': yuv_tensor,'desire':desire,
                'traffic_convention':traffic_convention,'lateral_control_params':lateral_control_params,
                'prev_desired_curv':prev_desired_curv,'nav_features':nav_features,
                'nav_instructions':nav_instructions,'features_buffer':features_buffer}
  
    ort_outs = ort_session.run(None, ort_inputs)
    time_end=time.time()
    time_c=time_end-time_start
    print("第",i,"次运行",time_c)

ort_outs=np.array(ort_outs).reshape(6504,)
METADATA_PATH="supercombo_metadata.pkl"
with open(METADATA_PATH, 'rb') as f:
    model_metadata = pickle.load(f)

output_slices = model_metadata['output_slices']
net_output_size = model_metadata['output_shapes']['outputs'][1]
output = np.zeros(net_output_size, dtype=np.float32)

parser = Parser()
outputs = parser.parse_outputs(slice_outputs(ort_outs))