# scrfd_crop_face
模块代码位于： service/scrfd_crop_face
测试单元位于： test/test_scrfd_crop_face.py
使用函数： get_max_frames

输入参数说明：接口包含以下参数
| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| face_path | 视频路径 | 无 |
| frame_info_json_path | 视频最长有效片段信息json文件 | 无 |

输出结果说明:
    json文件 {"112":[Xmin, Ymin, Xmax, Ymax],"112_kkps":[5个关键点坐标],...}

算法性能:
耗时比 1:0.5左右
| 视频分辨率 | 显卡 | 显存 | 视频时长 | 人脸检测耗时|
| --- | --- | --- | --- | --- |
| 1080P | 4090 | 600M | 20s | 10s |
