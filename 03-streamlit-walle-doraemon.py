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
        st.image(image, caption="上传的图片 (方法1)", use_container_width=True)
        
        # 重置文件指针并尝试预测
        uploaded_file.seek(0) # Crucial: reset pointer before re-reading
        image_for_predict = PILImage.create(uploaded_file) # Re-create for prediction
        
        st.write("开始预测 (方法1)...")
        pred, pred_idx, probs = model.predict(image_for_predict)
        st.success(f"预测结果 (方法1): {pred}; 概率: {probs[pred_idx]:.04f}")
        
    except Exception as e1:
        st.warning(f"方法1失败: {e1}")
        st.info(f"方法1错误详情: {traceback.format_exc()}")
        
        # 方法2：通过PIL先处理
        try:
            st.write("尝试方法2：使用PIL先处理图片")
            from PIL import Image # Local import to ensure it's available
            uploaded_file.seek(0) # Reset pointer
            pil_img = Image.open(uploaded_file)
            
            # 确保是RGB模式
            if pil_img.mode != 'RGB':
                st.write(f"图片模式为 {pil_img.mode}, 转换为 RGB...")
                pil_img = pil_img.convert('RGB')
            else:
                st.write("图片已为 RGB 模式")
            
            st.image(pil_img, caption="上传的图片 (方法2)", use_container_width=True)
            
            # 创建 FastAI 图像
            image_for_predict = PILImage.create(pil_img)
            st.write("开始预测 (方法2)...")
            pred, pred_idx, probs = model.predict(image_for_predict)
            st.success(f"预测结果 (方法2): {pred}; 概率: {probs[pred_idx]:.04f}")
            
        except Exception as e2:
            st.error(f"方法2也失败: {e2}")
            st.info(f"方法2错误详情: {traceback.format_exc()}")
            
            # 方法3：保存到临时文件再加载
            try:
                st.write("尝试方法3：临时文件方法")
                import tempfile
                import os
                
                uploaded_file.seek(0) # Reset pointer
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                st.write(f"临时文件已创建: {tmp_path}")
                image_for_predict = PILImage.create(tmp_path)
                st.image(image_for_predict, caption="上传的图片 (方法3)", use_container_width=True)
                
                st.write("开始预测 (方法3)...")
                pred, pred_idx, probs = model.predict(image_for_predict)
                st.success(f"预测结果 (方法3): {pred}; 概率: {probs[pred_idx]:.04f}")
                
                # 清理临时文件
                os.unlink(tmp_path)
                st.write(f"临时文件已删除: {tmp_path}")
                
            except Exception as e3:
                st.error(f"所有方法都失败了.")
                st.subheader("错误总结:")
                st.write(f"**方法1错误**: {e1}")
                st.write(f"**方法2错误**: {e2}")
                st.write(f"**方法3错误**: {e3}")
                st.subheader("详细错误信息 (最后一次尝试 - 方法3):")
                st.code(traceback.format_exc())
                
                st.subheader("可能的解决方案:")
                st.markdown("""
                1. **尝试上传不同格式或来源的图片。**
                2. **确保图片文件没有损坏。**
                3. **模型兼容性问题：** 模型 (`.pkl` 文件) 可能是在与当前 Streamlit Cloud 环境 (fastai/torch/python版本) 不完全兼容的环境中训练和导出的。如果是这样，最佳方案通常是：
                    * 在与 Streamlit Cloud 环境尽可能一致的环境中重新训练模型，或至少重新导出模型 (`learn.export()`)。
                    * 或者，只保存模型权重 (`learn.save('model_weights')`) 并在 Streamlit 应用中重新构建 Learner 结构后加载权重 (`learn.load('model_weights')`)。
                """) 