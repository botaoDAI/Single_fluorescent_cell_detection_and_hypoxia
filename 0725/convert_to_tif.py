import os
import bioformats
import javabridge
import tifffile
import numpy as np

# 输入和输出路径
input_dir = os.path.join(os.path.dirname(__file__), '_temp', '_ois3D9B_', 'stack1')
output_dir = os.path.join(os.path.dirname(__file__), 'tif_output')
os.makedirs(output_dir, exist_ok=True)

# 自动查找.ets或.vsi文件
input_file = None
for fname in os.listdir(input_dir):
    if fname.lower().endswith('.ets') or fname.lower().endswith('.vsi'):
        input_file = os.path.join(input_dir, fname)
        break
if input_file is None:
    raise FileNotFoundError('未找到.ets或.vsi文件')

output_tif = os.path.join(output_dir, 'stack1.tif')

# 启动Java虚拟机
javabridge.start_vm(class_path=bioformats.JARS)

try:
    # 读取元数据
    metadata = bioformats.get_omexml_metadata(input_file)
    # 读取图像数据
    with bioformats.ImageReader(input_file) as reader:
        rdr = reader.rdr
        num_c = rdr.getSizeC()
        size_y = rdr.getSizeY()
        size_x = rdr.getSizeX()
        print(f"C轴帧数: {num_c}, Y={size_y}, X={size_x}")

        # 只保存C轴的第三帧（索引2）
        img = reader.read(t=0, z=0, c=2, rescale=False)
        print(f"C轴第3帧 均值:{img.mean()} min:{img.min()} max:{img.max()}")

        metadata_ascii = metadata.encode('ascii', 'xmlcharrefreplace').decode('ascii')
        tifffile.imwrite(output_tif, img, description=metadata_ascii, photometric='minisblack')
        print(f"保存成功: {output_tif}")
finally:
    javabridge.kill_vm() 