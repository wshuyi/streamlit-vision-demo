import streamlit as st
from fastai.vision.all import *
import pathlib
from PIL import Image

@st.cache_resource
def load_model():
    """加载并缓存模型"""
    try:
        model_path = pathlib.Path(__file__).parent / "doraemon_walle_model.pkl"
        return load_learner(model_path)
    except Exception as e:
        st.error(f"模型加载失败: {e}")
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
    try:
        # 使用PIL打开图像，然后转换为PILImage
        pil_image = Image.open(uploaded_file)
        image = PILImage.create(pil_image)
        st.image(pil_image, caption="上传的图片", use_container_width=True)
        
        pred, pred_idx, probs = model.predict(image)
        st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")
        
    except Exception as e:
        st.error(f"图片处理失败: {e}")
        # 尝试备用方法
        try:
            st.write("尝试备用处理方法...")
            # 直接使用字节流创建PILImage
            uploaded_file.seek(0)  # 重置文件指针
            image = PILImage.create(uploaded_file)
            pred, pred_idx, probs = model.predict(image)
            st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")
        except Exception as e2:
            st.error(f"备用方法也失败了: {e2}")
            st.write("请尝试上传不同格式的图片，或联系开发者。") 