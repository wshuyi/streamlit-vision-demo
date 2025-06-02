import streamlit as st
from fastai.vision.all import *
import pathlib

@st.cache_resource
def load_model():
    """加载并缓存模型"""
    model_path = pathlib.Path(__file__).parent / "doraemon_walle_model.pkl"
    # Force loading onto CPU
    return load_learner(model_path, cpu=True)

# 主应用
st.title("图像分类应用")
st.write("上传一张图片，应用将预测对应的标签。")

model = load_model()

if model is None:
    st.error("模型未能成功加载，请检查日志。")
    st.stop()

uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = PILImage.create(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
        
        # Ensure prediction also respects CPU context if model is on CPU
        # Fastai's predict should handle this if learner is on CPU
        pred, pred_idx, probs = model.predict(image)
        st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")
    except Exception as e:
        st.error(f"图像处理或预测时发生错误: {e}")
        st.error("详细错误信息:")
        # Import traceback for detailed error logging
        import traceback
        st.code(traceback.format_exc())
        st.markdown("**提示**: 如果问题持续，请尝试重新导出模型 (`learn.export()`) 或仅加载模型权重。") 