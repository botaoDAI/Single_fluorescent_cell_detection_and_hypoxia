import os
import numpy as np
import bioformats
import javabridge
import tifffile

# 十六进制编号列表，从9A到B5
hex_ids = [format(x, 'X') for x in range(int('EA8', 16), int('EB3', 16)+1)]
folder_names = [f'_ois3{hid}_' for hid in hex_ids]

# 输入和输出路径
base_dir = os.path.join(os.path.dirname(__file__), '_temp')
output_tif = os.path.join(os.path.dirname(__file__), 'all_stacks.tif')

# 启动Java虚拟机
javabridge.start_vm(class_path=bioformats.JARS)

images = []
metadata = None

try:
    for folder in folder_names:
        stack1_dir = os.path.join(base_dir, folder, 'stack1')
        # 自动查找.ets或.vsi文件
        input_file = None
        for fname in os.listdir(stack1_dir):
            if fname.lower().endswith('.ets') or fname.lower().endswith('.vsi'):
                input_file = os.path.join(stack1_dir, fname)
                break
        if input_file is None:
            raise FileNotFoundError(f'{stack1_dir} 未找到.ets或.vsi文件')
        with bioformats.ImageReader(input_file) as reader:
            rdr = reader.rdr
            size_y = rdr.getSizeY()
            size_x = rdr.getSizeX()
            img = reader.read(t=0, z=0, c=2, rescale=False)
            images.append(img)
            print(f"读取 {folder} 第三帧 shape={img.shape} 均值={img.mean()}")
            if metadata is None:
                metadata = bioformats.get_omexml_metadata(input_file)
    images = np.stack(images, axis=0)  # shape: (12, Y, X)
    metadata_ascii = metadata.encode('ascii', 'xmlcharrefreplace').decode('ascii')
    tifffile.imwrite(output_tif, images, description=metadata_ascii, photometric='minisblack')
    print(f"保存成功: {output_tif}")
finally:
    javabridge.kill_vm() 