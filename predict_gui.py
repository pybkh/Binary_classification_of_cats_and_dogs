import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("400x200")

        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        self.model.to(self.device)
        try:
            self.model.load_state_dict(torch.load('resnet18_cat_dog.pth', map_location=self.device))
            self.model.eval()
        except FileNotFoundError:
            messagebox.showerror("错误", "找不到 mnist_model.pth，请先运行训练脚本！")

        self.btn_select = tk.Button(root, text="选择图片", command=self.select_and_predict, width=20, height=2)
        self.btn_select.pack(pady=10)

        self.label_result = tk.Label(root, text="预测结果: ---", font=("Arial", 16, "bold"))
        self.label_result.pack(pady=20)

    def select_and_predict(self):
        # 调起文件选择器
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        
        if not file_path:
            return

        # 2. 预处理并预测
        try:
            img = Image.open(file_path).convert('RGB')
            
            transform = transforms.Compose([  
                transforms.Resize((224, 224)),     # ResNet 输入 224x224    transforms.ToTensor(),  
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],  
                                    [0.229, 0.224, 0.225])   # ResNet 官方均值方差  
            ])  
            
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                # 注意：你的模型最后是 Sigmoid，输出是一个概率值 (0~1)
                # 而不是多分类的 softmax
                prob = output.item() 
                predicted = "狗" if prob > 0.5 else "猫"
                confidence = prob if prob > 0.5 else 1 - prob

            # 3. 更新界面结果
            res_text = f"结果: {predicted}"
            self.label_result.config(text=res_text)
            
        except Exception as e:
            messagebox.showerror("预测失败", f"处理图片时出错: {e}")

# 启动程序
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()