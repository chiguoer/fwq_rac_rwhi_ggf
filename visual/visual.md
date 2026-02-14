PYTHONPATH=. python tools/visual.py

使用标签生成目标框可视化指南（NuScenes + RaCFormer）
一、环境与文件说明
二、NuScenes 图像的组织与选择逻辑
三、使用标签在图像上绘制 GT 目标框
四、如何运行可视化脚本 & 指定自己想要的图片
1. 运行方式（在项目根目录）
2. 只画某一台相机的单张图片
使用标签生成目标框可视化指南（NuScenes + RaCFormer）
本指南说明如何：

基于 NuScenes 数据集标签（GT）生成 2D 图像上的 3D 目标框可视化；
在 NuScenes 中选择自己想要的图像（sample / camera）；
使用本仓库中的 ​visual/visual.py​ 脚本完成上述工作。
本文默认你的 NuScenes 数据放在 ​data/nuscenes/​，和 config 里保持一致。

一、环境与文件说明
代码根目录（示意）：
​​configs/​：模型配置，如 ​racformer_r50_nuimg_704x256_f8.py​
​​loaders/​：数据集定义，如 ​nuscenes_dataset.py​
​​visual/​：
​​visual.py​：可视化脚本（已包含基于标签的 GT 框绘制函数）
​​visual_eval_and_gt.md​：评估与可视化的详细说明
NuScenes 数据路径（假设）：
​​data/nuscenes/​
​​samples/​：关键帧图像和点云
​​sweeps/​：非关键帧
​​maps/​
​​v1.0-trainval/​：标注 JSON（​sample.json​、​sample_data.json​、​sample_annotation.json​ 等）
依赖要求：

已安装并能导入：
​​nuscenes-devkit​（​from nuscenes.nuscenes import NuScenes​）
​​mmcv​, ​matplotlib​, ​pyquaternion​, ​Pillow​ 等
二、NuScenes 图像的组织与选择逻辑
NuScenes 中与“图像”相关的核心概念有三个：

sample：一个时间点的“样本”，拥有多个传感器数据（6 个相机 + LIDAR_TOP + RADAR_*）。
sample_data：某个具体传感器在该样本下的一条数据（例如一张图像或一帧点云）。
token：这两类记录的唯一 ID 字符串。
典型选择步骤如下：

初始化 NuScenes API：

from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval',
                dataroot='data/nuscenes',
                verbose=True)
选择一个 sample（时间点）：

# 方式 A：按索引取
sample = nusc.sample[0]
sample_token = sample['token']

# 方式 B：如果你已有感兴趣的 sample_token
# sample_token = '...'
# sample = nusc.get('sample', sample_token)
选择具体相机通道（图像）：

NuScenes 规定了 6 个相机通道名：

​​CAM_FRONT​
​​CAM_FRONT_LEFT​
​​CAM_FRONT_RIGHT​
​​CAM_BACK​
​​CAM_BACK_LEFT​
​​CAM_BACK_RIGHT​
通过 sample 的 ​data​ 字段从 sample -> sample_data：

cam = 'CAM_FRONT'
sd_token = sample['data'][cam]           # sample_data 的 token
sd_record = nusc.get('sample_data', sd_token)

# 图像文件相对路径
rel_path = sd_record['filename']         # 如 'samples/CAM_FRONT/xxx.jpg'

# 也可以直接得到绝对路径
img_path = nusc.get_sample_data_path(sd_token)
同时获取该视角下的 GT 标注框：

推荐使用官方 helper：

from nuscenes.utils.geometry_utils import BoxVisibility

data_path, boxes_gt, camera_intrinsic = nusc.get_sample_data(
    sd_token,
    box_vis_level=BoxVisibility.ANY
)
​​data_path​：图像路径（与 ​img_path​ 相同）。
​​boxes_gt​：该相机视角下所有可见的 GT 3D 框（以相机坐标系表示的 ​Box​ 对象）。
​​camera_intrinsic​：3×3 内参，用于投影 3D box 到 2D 像素平面。
三、使用标签在图像上绘制 GT 目标框
​​visual/visual.py​ 中已经提供了一个仅依赖标签的可视化函数 ​render_sample_gt_only​，核心思路：

通过 ​sample_token​ 取出该样本的 6 个相机视角。
对每个相机，调用 ​nusc.get_sample_data()​ 获取：
图像路径
GT 3D boxes（​boxes_gt​）
相机内参
用 ​Box.render(...)​ 将 3D 框投在图像上，使用 matplotlib 显示或保存。
示例（简化版逻辑，已在 ​visual.py​ 中实现）：

from nuscenes.utils.geometry_utils import BoxVisibility
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def render_sample_gt_only(sample_token: str,
                          box_vis_level: BoxVisibility = BoxVisibility.ANY,
                          out_path: str = None,
                          verbose: bool = True):
    sample = nusc.get('sample', sample_token)
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]

    _, axes = plt.subplots(2, 3, figsize=(24, 12))
    for i, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]
        data_path, boxes_gt, camera_intrinsic = nusc.get_sample_data(
            sample_data_token, box_vis_level=box_vis_level)

        img = Image.open(data_path)
        row, col = divmod(i, 3)
        ax = axes[row, col]
        ax.imshow(img)

        # 绘制 GT 框
        for box in boxes_gt:
            c = np.array(get_color(box.name)) / 255.0  # 颜色函数见 visual.py
            box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

        ax.set_xlim(0, img.size[0])
        ax.set_ylim(img.size[1], 0)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title(cam)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()
    plt.close()
实际工程中，​render_sample_gt_only​ 已经写在 ​visual/visual.py​，你可以直接导入或在 ​__main__​ 中调用。

四、如何运行可视化脚本 & 指定自己想要的图片
1. 运行方式（在项目根目录）
cd d:\chiguoer\master\derma\rac_rwhi_ggf\rac-rwhi-ggf-2_1\fwq_rac_rwhi_ggf-ggf_add

mkdir visual_outputs_gt  # 若还没创建

python visual/visual.py
可以在 ​visual.py​ 的 ​if __name__ == '__main__':​ 里写你想要的逻辑。例如：

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='./data/nuscenes',
                    verbose=True)

    # 方式 A：随便取第 0 个 sample
    token = nusc.sample[0]['token']

    # 方式 B：如果你已有一个感兴趣的 sample_token
    # token = 'a4f1c0e5f2a24e3e9b689ed0c1c0b8c3'

    out_file = './visual_outputs_gt/' + token
    render_sample_gt_only(token, out_path=out_file)
执行后会在 ​visual_outputs_gt/​ 下生成一张包含 6 个相机视角的 GT 可视化图。

2. 只画某一台相机的单张图片
如果你只对某个相机（比如 ​CAM_FRONT​）感兴趣，可以参考下面的辅助函数（自己加到 ​visual.py​ 中）：

def render_single_camera_gt(sample_token: str,
                            camera: str = 'CAM_FRONT',
                            box_vis_level: BoxVisibility = BoxVisibility.ANY,
                            out_path: str = None,
                            verbose: bool = True):
    sample = nusc.get('sample', sample_token)
    sd_token = sample['data'][camera]
    data_path, boxes_gt, camera_intrinsic = nusc.get_sample_data(
        sd_token, box_vis_level=box_vis_level)

    img = Image.open(data_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(img)

    for box in boxes_gt:
        c = np.array(get_color(box.name)) / 255.0
        box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

    ax.set_xlim(0, img.size[0])
    ax.set_ylim(img.size[1], 0)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_title(camera)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()
    plt.close()
在 ​__main__​ 中调用示例：

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='./data/nuscenes',
                    verbose=True)

    token = nusc.sample[0]['token']
    render_single_camera_gt(
        token,
        camera='CAM_FRONT',
        out_path='./visual_outputs_gt/{}_CAM_FRONT.png'.format(token)
    )
这样就可以非常精确地：

按 ​sample_token​ 选择你想看的时间点；
按相机名（​CAM_FRONT​ 等）选你想要的那一张图；
基于 数据集标签（GT） 画出对应的 3D 目标框在图像上的投影。

3. 通过图片文件名（编号）反查 sample 并可视化
在实际使用中，我们经常是“先有一张图片的文件名（编号）”，例如：

n008-2018-05-21-11-06-59-0400_CAM_FRONT__1526915283912465.jpg

希望做两件事：

1）知道它在 nusc.sample 里是第几个 sample；
2）直接基于这张图片对应的 sample，画出 6 个相机视角的 GT 可视化图。

为此，在 tools/visual.py 中实现了两个辅助函数：find_sample_index_by_image_name 和 render_by_image_name。

3.1 find_sample_index_by_image_name 的功能与用法
函数定义（位于 tools/visual.py）：

def find_sample_index_by_image_name(img_name: str):
    """
    根据图片文件名（不含路径）查找：
      - 它对应的 sample 在 nusc.sample 中是第几个（index）
      - 对应的 sample_token
      - 对应的 sample_data 记录（camera）

    Args:
        img_name: 比如 'n008-2018-05-21-11-06-59-0400_CAM_FRONT__1526915283912465.jpg'

    Returns:
        (sample_index, sample_token, sample_data_record)
        如果找不到，返回 (None, None, None)
    """
    target_sd = None
    for sd in nusc.sample_data:
        # NuScenes 中 filename 通常类似 'samples/CAM_FRONT/xxx.jpg'
        if img_name in sd['filename'] and sd['sensor_modality'] == 'camera':
            target_sd = sd
            break

    if target_sd is None:
        return None, None, None

    sample_token = target_sd['sample_token']

    # 找到该 sample_token 在 nusc.sample 中的下标
    sample_index = None
    for i, s in enumerate(nusc.sample):
        if s['token'] == sample_token:
            sample_index = i
            break

    return sample_index, sample_token, target_sd

使用示例：

img_name = 'n008-2018-05-21-11-06-59-0400_CAM_FRONT__1526915283912465.jpg'
idx, token, sd = find_sample_index_by_image_name(img_name)
print('这是第几个 sample:', idx)
print('对应的 sample_token:', token)
print('对应的相机通道:', sd['channel'])

3.2 render_by_image_name 的功能与用法
在知道图片文件名之后，如果你只想“一步到位”地完成可视化，可以使用 render_by_image_name：

def render_by_image_name(img_name: str,
                         out_dir: str = './visual_outputs_gt/',
                         box_vis_level: BoxVisibility = BoxVisibility.ANY,
                         verbose: bool = True):
    """
    通过图片文件名，自动找到对应的 sample，并调用 render_sample_gt_only 进行可视化。

    Args:
        img_name: 比如 'n008-2018-05-21-11-06-59-0400_CAM_FRONT__1526915283912465.jpg'
        out_dir: 输出图片的目录前缀
    """
    sample_index, sample_token, sd = find_sample_index_by_image_name(img_name)

    if sample_token is None:
        print(f'未在 nusc.sample_data 中找到图片: {img_name}')
        return

    print(f'图片 {img_name} 对应的 sample_index = {sample_index}, sample_token = {sample_token}')
    out_path = out_dir + sample_token
    render_sample_gt_only(sample_token,
                          box_vis_level=box_vis_level,
                          out_path=out_path,
                          verbose=verbose)

最典型的用途就是：你只知道一张 NuScenes 图片的“编号”（文件名），通过 render_by_image_name 即可：

1）自动找到它属于哪个 sample（时间点）；
2）用 render_sample_gt_only 画出该 sample 的 6 个相机视角的 GT 框图。

3.3 使用 find_sample_index_by_image_name 和 render_by_image_name 的完整示例
你可以在 tools/visual.py 的末尾将 __main__ 部分改成如下形式，基于图片文件名驱动可视化：

if __name__ == '__main__':
    # 1. 初始化 NuScenes
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='./data/nuscenes',
                    verbose=True)

    # 2. 指定你感兴趣的图片文件名（不带路径）
    img_name = 'n008-2018-05-21-11-06-59-0400_CAM_FRONT__1526915283912465.jpg'

    # 3. 使用 find_sample_index_by_image_name 查询它是第几个 sample
    idx, token, sd = find_sample_index_by_image_name(img_name)
    if token is None:
        print('未找到图片:', img_name)
    else:
        print('图片 {} 对应的 sample_index = {}, sample_token = {}'.format(img_name, idx, token))
        print('该图片的相机通道为:', sd['channel'])

        # 4. 使用 render_by_image_name 直接生成 6 个相机视角的 GT 可视化图
        #    输出会保存在 ./visual_outputs_gt/<sample_token>.png（由 render_sample_gt_only 决定具体命名）
        render_by_image_name(img_name,
                             out_dir='./visual_outputs_gt/',
                             box_vis_level=BoxVisibility.ANY,
                             verbose=True)

通过这两个函数，你就可以：

从一张 NuScenes 图片的文件名（编号）快速反查到它是第几个 sample；
直接通过文件名触发完整的 GT 可视化，无需自己手动查 token 或 sample 索引。

3.4 只对这一张图片进行可视化（不画同一个 token 的 6 张）
上面的 render_by_image_name 会针对该图片所属的 sample_token，画出 6 个相机视角的 GT 图。
如果你只想“这一张图片”的可视化（不管其它相机），可以使用 tools/visual.py 中的：

def render_single_image_by_name(img_name: str,
                                box_vis_level: BoxVisibility = BoxVisibility.ANY,
                                out_path: str = None,
                                verbose: bool = True):
    """
    只根据“图片文件名”对这一张图片进行 GT 可视化，
    不再画同一个 sample_token 下的其它 5 个相机视角。

    Args:
        img_name: 图片文件名（不含路径），例如
                  'n008-2018-05-21-11-06-59-0400_CAM_FRONT__1526915283912465.jpg'
        box_vis_level: BoxVisibility，可控制可见性要求，默认 ANY。
        out_path: 若不为 None，则保存到该路径；否则只进行 plt.show()。
    """
    sample_index, sample_token, sd = find_sample_index_by_image_name(img_name)
    if sd is None:
        print(f'未在 nusc.sample_data 中找到图片: {img_name}')
        return

    # 这里直接用该 sample_data 的 token，获取图像与该视角下的 GT 框
    sd_token = sd['token']
    data_path, boxes_gt, camera_intrinsic = nusc.get_sample_data(
        sd_token, box_vis_level=box_vis_level)

    img = Image.open(data_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(img)

    for box in boxes_gt:
        c = np.array(get_color(box.name)) / 255.0
        box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

    ax.set_xlim(0, img.size[0])
    ax.set_ylim(img.size[1], 0)
    ax.axis('off')
    ax.set_aspect('equal')
    # 标题直接用该图片对应的相机通道（如 CAM_FRONT）
    ax.set_title(sd['channel'])

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()
    plt.close()

在 __main__ 中调用示例（只生成这一张图片的可视化）：

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='./data/nuscenes',
                    verbose=True)

    img_name = 'n008-2018-05-21-11-06-59-0400_CAM_FRONT__1526915283912465.jpg'

    # 只画这一张图片
    render_single_image_by_name(
        img_name,
        box_vis_level=BoxVisibility.ANY,
        out_path='./visual_outputs_gt/{}_single.png'.format(img_name)
    )

这样，你可以：

只生成这张图片的 GT 可视化（单图版）；
或者用 render_by_image_name 生成该 sample 的 6 张相机视角可视化（多图版），根据需要选择使用。

PYTHONPATH=. python tools/visual.py
