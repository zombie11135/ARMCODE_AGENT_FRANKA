# skill_lib/perception/describe_scene.py

from core.skill_base import BaseSkill, SkillResult


DEFAULT_MULTI_VIEW_SCENE_PROMPT = """
你是一个机器人视觉描述模块 VDM。现在会给你两张图像：

第一张图像来自全局相机：
- 用于观察整个桌面场景；
- 可以判断桌面上有哪些物体、物体之间的大致位置关系、托盘或容器的位置；
- 但是全局相机没有手眼标定，所以不能输出机械臂坐标或精确三维位置。

第二张图像来自腕部相机：
- 用于观察机械臂末端附近的局部区域；
- 可以辅助判断夹爪附近的目标、药盒文字、局部遮挡、抓取前状态；
- 如果腕部图像看不到完整场景，请不要过度推断。

请综合两张图像，输出适合机器人 Agent 后续规划的场景描述。
两张图可能观察的是同一场景，请不要简单相加物体数量。
请尝试合并两个视角中可能对应同一物体的观察结果。
请重点描述：
1. 桌面上有哪些可操作物体，例如药盒、水果、小方块、托盘、盒子、杯子等；
2. 哪些物体可能是抓取目标；
3. 哪些区域或容器可能是放置目标；
4. 全局图和腕部图分别提供了什么信息；
5. 如果图中有药盒或文字，可以说明可能存在文字，但不要编造无法确认的具体文字；
6. 不要输出机械臂精确坐标；
7. 如果信息不确定，请明确说“不确定”。
8. 聚焦于桌面，忽略背景放置的杂物，比如人等，请聚焦于桌面

请用中文输出，格式如下：

【全局相机观察】
...

【腕部相机观察】
...

【综合场景理解】
...

【可操作物体】
1. ...
2. ...
"""


GLOBAL_ONLY_SCENE_PROMPT = """
你是一个机器人视觉描述模块 VDM。现在只给你一张全局相机图像。
请描述桌面上的可操作物体、可能的药盒、托盘/盘子/容器及其颜色和大致相对位置。
不要输出机械臂精确坐标；不确定的信息请明确说“不确定”。
请用中文输出。
"""


WRIST_ONLY_SCENE_PROMPT = """
你是一个机器人视觉描述模块 VDM。现在只给你一张腕部相机图像。
请描述腕部视角中可见的药盒、文字线索、托盘/盘子/容器、遮挡情况和局部相对位置。
如果看不到完整场景，请不要过度推断；不确定的信息请明确说“不确定”。
请用中文输出。
"""


class VDMSkill(BaseSkill):
    name = "describe_scene"
    description = "使用全局相机和腕部相机 RGB 图像生成 VDM 场景描述。"

    input_schema = {
        "global_image_path": "可选。全局相机 RGB 图像路径。",
        "wrist_image_path": "可选。腕部相机 RGB 图像路径。",
        "use_wrist": "是否同时使用腕部相机图像，默认 True。",
        "prompt": "可选。自定义 VDM prompt。",
        "max_tokens": "可选。最大输出 token 数。",
    }

    output_schema = {
        "global_image_path": "用于 VDM 的全局相机图像路径",
        "wrist_image_path": "用于 VDM 的腕部相机图像路径",
        "scene_text": "Qwen-VL 生成的综合场景描述",
    }

    def run(
        self,
        global_image_path: str = None,
        wrist_image_path: str = None,
        use_wrist: bool = True,
        prompt: str = None,
        max_tokens: int = 1024,
        **kwargs,
    ) -> SkillResult:
        try:
            vlm = self.context.require("vlm")
            fallback_vlm = getattr(self.context, "fallback_vlm", None)
            camera = self.context.require("camera")

            # 1. 如果没有传入全局图，则拍一张
            if global_image_path is None:
                global_capture = camera.capture_global()
                global_image_path = global_capture["color_path"]

                self.context.emit_image(
                    image_path=global_image_path,
                    caption="VDM 使用的全局相机 RGB 图像",
                )

            # 2. 如果需要腕部图，但没有传入，则拍一张腕部 RGB-D
            if use_wrist and wrist_image_path is None:
                wrist_capture = camera.capture_wrist()
                wrist_image_path = wrist_capture["color_path"]

                self.context.emit_image(
                    image_path=wrist_image_path,
                    caption="VDM 使用的腕部相机 RGB 图像",
                )

            prompt = prompt or DEFAULT_MULTI_VIEW_SCENE_PROMPT

            image_paths = [global_image_path]

            if use_wrist and wrist_image_path is not None:
                image_paths.append(wrist_image_path)

            self.context.emit_text(
                f"正在调用 VDM 生成多视角场景描述，输入图像数量：{len(image_paths)}"
            )

            scene_text, provider = self._describe_images_with_fallback(
                primary_vlm=vlm,
                fallback_vlm=fallback_vlm,
                image_paths=image_paths,
                prompt=prompt,
                max_tokens=max_tokens,
            )

            self.context.emit_text(f"多视角 VDM 场景描述生成完成，provider={provider}。")
            self.context.emit_text(scene_text)

            return SkillResult(
                success=True,
                data={
                    "global_image_path": global_image_path,
                    "wrist_image_path": wrist_image_path,
                    "use_wrist": use_wrist,
                    "scene_text": scene_text,
                    "provider": provider,
                },
                message="多视角 VDM 场景描述生成成功",
            )

        except Exception as e:
            error = f"多视角 VDM 场景描述失败: {e}"

            try:
                self.context.emit_error(error)
            except Exception:
                pass

            return SkillResult(
                success=False,
                data={},
                error=error,
            )

    def _describe_images_with_fallback(
        self,
        primary_vlm,
        fallback_vlm,
        image_paths,
        prompt: str,
        max_tokens: int,
    ):
        try:
            scene_text = primary_vlm.describe_images(
                image_paths=image_paths,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            return scene_text, "primary_vlm"
        except Exception as primary_error:
            if self._is_single_image_limit_error(primary_error) and len(image_paths) > 1:
                try:
                    self.context.emit_warning(
                        "VDM 服务一次只支持 1 张图，自动改为单图分步描述后合并。"
                    )
                except Exception:
                    pass
                scene_text = self._describe_multiview_as_single_images(
                    vlm=primary_vlm,
                    image_paths=image_paths,
                    max_tokens=max_tokens,
                )
                return scene_text, "primary_vlm_single_image_fallback"

            if fallback_vlm is None:
                raise

            try:
                self.context.emit_warning(
                    f"主 VDM 调用失败，切换到 API fallback: {primary_error}"
                )
            except Exception:
                pass

            try:
                scene_text = fallback_vlm.describe_images(
                    image_paths=image_paths,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            except Exception as fallback_error:
                if self._is_single_image_limit_error(fallback_error) and len(image_paths) > 1:
                    try:
                        self.context.emit_warning(
                            "Fallback VDM 服务一次只支持 1 张图，自动改为单图分步描述后合并。"
                        )
                    except Exception:
                        pass
                    scene_text = self._describe_multiview_as_single_images(
                        vlm=fallback_vlm,
                        image_paths=image_paths,
                        max_tokens=max_tokens,
                    )
                    return scene_text, "fallback_vlm_single_image_fallback"
                raise
            return scene_text, "fallback_vlm"

    def _describe_multiview_as_single_images(
        self,
        vlm,
        image_paths,
        max_tokens: int,
    ) -> str:
        global_text = vlm.describe_images(
            image_paths=[image_paths[0]],
            prompt=GLOBAL_ONLY_SCENE_PROMPT,
            max_tokens=max(256, max_tokens // 2),
        )
        wrist_text = ""
        if len(image_paths) > 1:
            wrist_text = vlm.describe_images(
                image_paths=[image_paths[1]],
                prompt=WRIST_ONLY_SCENE_PROMPT,
                max_tokens=max(256, max_tokens // 2),
            )

        return (
            "【全局相机观察】\n"
            f"{global_text}\n\n"
            "【腕部相机观察】\n"
            f"{wrist_text or '未使用腕部相机图像。'}\n\n"
            "【综合场景理解】\n"
            "以上内容由单图 VDM 分步生成后合并；如果两个视角存在冲突，后续规划应优先保持保守。"
        )

    @staticmethod
    def _is_single_image_limit_error(error: Exception) -> bool:
        text = str(error).lower()
        return (
            "at most 1 image" in text
            or "最多" in text and "1" in text and "图" in text
            or "only support" in text and "1 image" in text
        )
