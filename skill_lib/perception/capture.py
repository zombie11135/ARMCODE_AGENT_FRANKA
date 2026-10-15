# skill_lib/perception/capture.py

from typing import Dict, Any

from core.skill_base import BaseSkill, SkillResult


def _remove_raw_arrays(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    capture service 返回里可能包含 numpy array。
    这些原始数组不适合直接交给 Agent / LLM / JSON 记录。
    perception skill 对外主要返回图片路径和相机参数。
    """
    clean = dict(result)
    clean.pop("color", None)
    clean.pop("depth", None)
    return clean


class CaptureGlobalImageSkill(BaseSkill):
    name = "capture_global_image"
    description = "从全局 RealSense 相机获取当前 RGB 图像，用于 VDM、场景描述和目标识别。"

    input_schema = {}

    output_schema = {
        "camera_name": "global",
        "color_path": "全局相机 RGB 图像路径",
        "color_intrinsics": "全局相机彩色图内参",
        "timestamp": "拍摄时间戳",
    }

    def run(self, **kwargs) -> SkillResult:
        if self.context.camera is None:
            return SkillResult(
                success=False,
                error="camera service is not initialized",
            )

        try:
            result = self.context.camera.capture_global()
            data = _remove_raw_arrays(result)

            color_path = data.get("color_path")
            if color_path:
                self.context.emit_image(
                    image_path=color_path,
                    caption="全局相机当前 RGB 图像",
                )

            return SkillResult(
                success=True,
                data=data,
                message="全局相机拍照成功",
            )

        except Exception as e:
            error = f"全局相机拍照失败: {e}"
            self.context.emit_error(error)
            return SkillResult(
                success=False,
                error=error,
            )


class CaptureWristRGBDSkill(BaseSkill):
    name = "capture_wrist_rgbd"
    description = "从腕部 RealSense 相机获取当前 RGB-D 图像，用于近距离 OCR、分割、深度定位和抓取确认。"

    input_schema = {}

    output_schema = {
        "camera_name": "wrist",
        "color_path": "腕部相机 RGB 图像路径",
        "depth_path": "腕部相机深度图路径",
        "depth_vis_path": "腕部相机深度可视化图路径",
        "color_intrinsics": "彩色图内参",
        "depth_intrinsics": "深度图内参",
        "depth_scale": "深度尺度",
        "timestamp": "拍摄时间戳",
    }

    def run(self, **kwargs) -> SkillResult:
        if self.context.camera is None:
            return SkillResult(
                success=False,
                error="camera service is not initialized",
            )

        try:
            result = self.context.camera.capture_wrist()
            data = _remove_raw_arrays(result)

            color_path = data.get("color_path")
            depth_vis_path = data.get("depth_vis_path")

            if color_path:
                self.context.emit_image(
                    image_path=color_path,
                    caption="腕部相机当前 RGB 图像",
                )

            if depth_vis_path:
                self.context.emit_image(
                    image_path=depth_vis_path,
                    caption="腕部相机当前深度可视化图",
                )

            return SkillResult(
                success=True,
                data=data,
                message="腕部相机 RGB-D 拍照成功",
            )

        except Exception as e:
            error = f"腕部相机拍照失败: {e}"
            self.context.emit_error(error)
            return SkillResult(
                success=False,
                error=error,
            )