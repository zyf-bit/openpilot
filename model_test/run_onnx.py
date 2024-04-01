import cv2
import numpy as np
import onnxruntime
import pickle
import os
import sys
sys.path.append("F://")
print(sys.path)
from openpilot.selfdrive.modeld.parse_model_outputs import Parser


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

# 构建输入的字典和计算输出结果
ort_inputs = {'input_imgs': yuv_tensor,'big_input_imgs': yuv_tensor,'desire':desire,
              'traffic_convention':traffic_convention,'lateral_control_params':lateral_control_params,
              'prev_desired_curv':prev_desired_curv,'nav_features':nav_features,
              'nav_instructions':nav_instructions,'features_buffer':features_buffer}
# ort_inputs = {ort_session.get_inputs()[0].name: yuv_tensor,ort_session.get_inputs()[1].name: yuv_tensor,
#               ort_session.get_inputs()[2].name: desire,ort_session.get_inputs()[3].name: traffic_convention,
#               ort_session.get_inputs()[4].name: nav_features,ort_session.get_inputs()[5].name:features_buffer,
#               }
ort_outs = ort_session.run(None, ort_inputs)
ort_outs=np.array(ort_outs).reshape(6504,)
print('ort_outs的shape:',np.array(ort_outs).shape)
#np.savetxt('test.txt',np.array(ort_outs).reshape(-1, np.array(ort_outs).shape[-1]),fmt='%f',delimiter=',')
METADATA_PATH="supercombo_metadata.pkl"
with open(METADATA_PATH, 'rb') as f:
      model_metadata = pickle.load(f)

output_slices = model_metadata['output_slices']
net_output_size = model_metadata['output_shapes']['outputs'][1]
output = np.zeros(net_output_size, dtype=np.float32)
print("从源代码拿来的output的shape:",output.shape)
parser = Parser()
outputs = parser.parse_outputs(slice_outputs(ort_outs))
#print(outputs)






#模型输出结果
print("模型输出结果:")
#plan输出
print("规划轨迹:")
print("plan\n", outputs['plan'])
print("plan的形状:", outputs['plan'].shape)
print("plan_weights:\n", outputs['plan_weights'])
print("plan_weights的形状:", outputs['plan_weights'].shape)
print("plan_hypotheses:\n", outputs['plan_hypotheses'])
print("plan_hypotheses的形状:", outputs['plan_hypotheses'].shape)
print("plan_stds:\n", outputs['plan_stds'])
print("plan_stds的形状:", outputs['plan_stds'].shape)
print("plan_stds_hypotheses:\n", outputs['plan_stds_hypotheses'])
print("plan_stds_hypotheses的形状:", outputs['plan_stds_hypotheses'].shape)
print("横向期望曲率:", outputs['desired_curvature'])
#lane_lines输出
print("车道线:")
print("lane_lines:\n", outputs['lane_lines'])
print("lane_lines的形状:", outputs['lane_lines'].shape)
print("lane_lines_stds:\n", outputs['lane_lines_stds'])
print("lane_lines_stds的形状:", outputs['lane_lines_stds'].shape)
print("lane_lines_prob:\n", outputs['lane_lines_prob'])
print("lane_lines_prob的形状:", outputs['lane_lines_prob'].shape)
#road_edges输出
print("道路边缘线:")
print("road_edges:\n", outputs['road_edges'])
print("road_edges的形状:", outputs['road_edges'].shape)
print("road_edges_stds:\n", outputs['road_edges_stds'])
print("road_edges_stds的形状:", outputs['road_edges_stds'].shape)
#lead输出
print("前车信息:")
print("lead:\n", outputs['lead'])
print("lead的形状:", outputs['lead'].shape)
print("lead_hypotheses:\n", outputs['lead_hypotheses'])
print("lead_hypotheses的形状:", outputs['lead_hypotheses'].shape)
print("lead_stds:\n", outputs['lead_stds'])
print("lead_stds的形状:", outputs['lead_stds'].shape)
print("lead_stds_hypotheses:\n", outputs['lead_stds_hypotheses'])
print("lead_stds_hypotheses的形状:", outputs['lead_stds_hypotheses'].shape)
print("lead_prob:\n", outputs['lead_prob'])
print("lead_prob的形状:", outputs['lead_prob'].shape)
print("lead_weights:\n", outputs['lead_weights'])
print("lead_weights的形状:", outputs['lead_weights'].shape)
print("desire_state:", outputs['desire_state'])
print("desire_state的形状:", outputs['desire_state'].shape)
print("desire_pred:",  outputs['desire_pred'])
print("desire_pred的形状:", outputs['desire_pred'].shape)
#meta输出
print("meta数据:")
print("meta:",outputs['meta'])
print("meta的形状:", outputs['meta'].shape)
#pose输出
print("自车姿态信息:")
print("pose:", outputs['pose'])
print("pose的形状:", outputs['pose'].shape)
print("pose_stds:", outputs['pose_stds'])
print("pose_stds的形状:", outputs['pose_stds'].shape)
print("sim_pose:", outputs['sim_pose'])
print("sim_pose的形状:", outputs['sim_pose'].shape)
print("sim_pose_stds:", outputs['sim_pose_stds'])
print("sim_pose_stds的形状:", outputs['sim_pose_stds'].shape)
#其他输出
print("其他输出:")
print("road_transform:", outputs['road_transform'])
print("road_transform的形状:", outputs['road_transform'].shape)
print("road_transform_stds:", outputs['road_transform_stds'])
print("road_transform_stds的形状:", outputs['road_transform_stds'].shape)
print("desired_curvature:", outputs['desired_curvature'])
print("desired_curvature的形状:", outputs['desired_curvature'].shape)
print("desired_curvature_stds:", outputs['desired_curvature_stds'])
print("desired_curvature_stds的形状:", outputs['desired_curvature_stds'].shape)
print("hidden_state:", outputs['hidden_state'])
print("hidden_state的形状:", outputs['hidden_state'].shape)
print("wide_from_device_euler:", outputs['wide_from_device_euler'])
print("wide_from_device_euler的形状:", outputs['wide_from_device_euler'].shape)
print("wide_from_device_euler_stds:", outputs['wide_from_device_euler_stds'])
print("wide_from_device_euler_stds的形状:", outputs['wide_from_device_euler_stds'].shape)
#print("置信度:", modelV2.confidence)
#打印outputs的数据结构
print("outputs的类型:", type(outputs))
if isinstance(outputs, dict):
    for key, value in outputs.items():
        print("键:", key)
        print("值类型:", type(value))
if isinstance(value, np.ndarray):
    print("值形状:", value.shape)
print()
print("!!!!!!!!!!!!!!!!!!!!!!!")