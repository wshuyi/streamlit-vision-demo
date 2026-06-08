import streamlit as st
import sys

# Python 版本检查
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

import importlib
import pkgutil

# --- 兼容性修复：plum-dispatch 2.x 把子模块重命名（plum.function -> plum._function 等）---
# 本模型 .pkl 是用旧版 plum-dispatch（1.x）序列化的，反序列化时会尝试 import 旧的
# 模块名（plum.function / plum.resolver / plum.signature ...）。新版里这些模块都已加下划线前缀，
# 于是 torch.load 抛出 ModuleNotFoundError；而 fastai 的 load_learner 把这个 ImportError
# 静默吞掉，最终报出令人困惑的 “UnboundLocalError: res”。
# 这里在加载前把所有 plum._X 反向别名为 plum.X，让旧 pickle 能找到类。
import plum
for _m in pkgutil.iter_modules(plum.__path__):
    if _m.name.startswith("_"):
        sys.modules.setdefault("plum." + _m.name[1:], importlib.import_module("plum." + _m.name))

from fastai.vision.all import *
import pathlib

@st.cache_resource
def load_model():
    """加载并缓存模型"""
    model_path = pathlib.Path(__file__).parent / "doraemon_walle_model.pkl"
    return load_learner(model_path)

# 主应用
st.title("图像分类应用")
st.write("上传一张图片，应用将预测对应的标签。")

model = load_model()

uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption="上传的图片", use_container_width=True)
    
    pred, pred_idx, probs = model.predict(image)
    st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}") 