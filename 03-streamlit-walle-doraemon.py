import streamlit as st
from fastai.vision.all import *
import pathlib
import traceback

@st.cache_resource
def load_model():
    """加载并缓存模型"""
    try:
        model_path = pathlib.Path(__file__).parent / "doraemon_walle_model.pkl"
        learner = load_learner(model_path)
        st.success("模型加载成功！")
        return learner
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.error(traceback.format_exc())
        return None

# 主应用
st.title("图像分类应用")
st.write("上传一张图片，应用将预测对应的标签。")

model = load_model()

if model is None:
    st.error("模型加载失败，请检查模型文件是否存在。")
    st.stop()

uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("正在处理图片...")
    
    # 方法1：尝试直接创建 PILImage
    try:
        st.write("尝试方法1：直接从上传文件创建 PILImage")
        image = PILImage.create(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
        
        # 重置文件指针并尝试预测
        uploaded_file.seek(0)
        image = PILImage.create(uploaded_file)
        
        st.write("开始预测...")
        pred, pred_idx, probs = model.predict(image)
        st.success(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")
        
    except Exception as e1:
        st.warning(f"方法1失败: {e1}")
        
        # 方法2：通过PIL先处理
        try:
            st.write("尝试方法2：使用PIL先处理图片")
            from PIL import Image
            uploaded_file.seek(0)
            pil_img = Image.open(uploaded_file)
            
            # 确保是RGB模式
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            st.image(pil_img, caption="上传的图片", use_container_width=True)
            
            # 创建 FastAI 图像
            image = PILImage.create(pil_img)
            st.write("开始预测...")
            pred, pred_idx, probs = model.predict(image)
            st.success(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")
            
        except Exception as e2:
            st.error(f"方法2也失败: {e2}")
            
            # 方法3：保存到临时文件再加载
            try:
                st.write("尝试方法3：临时文件方法")
                import tempfile
                import os
                
                uploaded_file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                image = PILImage.create(tmp_path)
                st.image(image, caption="上传的图片", use_container_width=True)
                
                st.write("开始预测...")
                pred, pred_idx, probs = model.predict(image)
                st.success(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")
                
                # 清理临时文件
                os.unlink(tmp_path)
                
            except Exception as e3:
                st.error(f"所有方法都失败了:")
                st.error(f"方法1错误: {e1}")
                st.error(f"方法2错误: {e2}")
                st.error(f"方法3错误: {e3}")
                st.error("详细错误信息:")
                st.error(traceback.format_exc())
                
                st.write("**可能的解决方案:**")
                st.write("1. 尝试上传不同格式的图片")
                st.write("2. 确保图片文件没有损坏")
                st.write("3. 模型可能需要使用相同版本的 fastai 重新训练") 