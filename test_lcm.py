# test_lcm.py
# 作用：在本地 D 盘加载 DreamShaper-7 + LCM-LoRA，生成几张图并打印耗时/显存峰值
# 运行前：确保已激活你的 conda 环境，并且模型已下载到 D:\xzr_deeplearning\models\...

import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler


# ====== 你本地的模型路径（按你前面下载的目录）======
MODEL_DIR = r"D:\xzr_deeplearning\models\dreamshaper-7"
LCM_LORA_DIR = r"D:\xzr_deeplearning\models\lcm-lora-sdv1-5"

OUT_DIR = r"D:\xzr_deeplearning\projects\outputs_lcm"
PROMPT = "a photo of an astronaut riding a horse on mars, cinematic lighting, ultra detailed"
SEED = 0


def load_pipeline(model_dir: str):
    """
    优先按 fp16 variant 加载；如果模型仓库不含 variant 文件，则回退到普通加载。
    """
    kwargs = dict(torch_dtype=torch.float16)
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_dir, variant="fp16", **kwargs)
    except Exception:
        pipe = AutoPipelineForText2Image.from_pretrained(model_dir, **kwargs)
    return pipe


@torch.inference_mode()
def generate_one(pipe, prompt: str, steps: int, guidance: float, seed: int, out_path: Path):
    # 固定随机种子
    g = torch.Generator(device="cuda").manual_seed(seed)

    # 计时与显存峰值
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    img = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
    ).images[0]

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    img.save(out_path)
    print(f"Saved: {out_path.name} | steps={steps} guidance={guidance} | time={dt:.3f}s | peak={peak_mb:.0f}MB")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请确认你安装的是 CUDA 版 PyTorch，并且 nvidia-smi 正常。")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 可选：允许 TF32（对速度有帮助，不影响脚本正确性）
    torch.backends.cuda.matmul.allow_tf32 = True

    # 1) 加载 base 文生图 pipeline
    pipe = load_pipeline(MODEL_DIR).to("cuda")

    # 2) 为 8GB 显存做一些保守设置
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    # 3) 替换为 LCM 调度器 + 加载 LCM-LoRA
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM_LORA_DIR)
    # fuse 可减少推理开销（如果你后续还要卸载/切换 LoRA，可以注释掉这行）
    if hasattr(pipe, "fuse_lora"):
        pipe.fuse_lora()

    # 4) 预热一次（避免首次编译/加载影响计时）
    _ = pipe(prompt="warmup", num_inference_steps=2, guidance_scale=0.0).images[0]
    torch.cuda.synchronize()

    # 5) 正式生成：LCM-LoRA 常用 2~8 步；guidance 建议 0 或 1~2
    settings = [
        (4, 0.0),
        (8, 1.5),
    ]

    for steps, guidance in settings:
        out = Path(OUT_DIR) / f"lcm_steps{steps}_gs{guidance}_seed{SEED}.png"
        generate_one(pipe, PROMPT, steps, guidance, SEED, out)

    print("Done.")


if __name__ == "__main__":
    main()
