# 单个目标检测结果（像素坐标）
int32 class_id        # 类别 ID（与模型训练类别对应）
string class_name     # 类别名（装甲板数字）
float32 confidence    # 置信度 0~1
float32 x             # 检测框左上角 x（原图像素）
float32 y             # 检测框左上角 y（原图像素）
float32 width         # 检测框宽度（像素）
float32 height        # 检测框高度（像素）
