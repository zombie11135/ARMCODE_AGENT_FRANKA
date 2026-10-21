"""Compose services for the simple agent and execution runtime.

This is the shared assembly point for backend selection. Heavyweight imports
remain inside factories; constructing a context may load models, while camera
start/stop and robot execution remain the caller's responsibility.
"""
from core.context import RuntimeContext
from runtime.camera import TaskCachedCamera


def build_planner_context(config):
    """Build reference/name and language-model services; the caller attaches VDM capture."""
    from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient
    from services.object_reference_service import build_reference_service
    context = RuntimeContext(object_references=build_reference_service(config))
    context.target_name_bindings = dict(config.target_name_bindings)
    if config.use_llm_planner and config.planner_llm_base_url:
        context.vlm = OpenAIChatGPTVLMClient(
            base_url=config.planner_llm_base_url,
            api_key=config.planner_llm_api_key,
            model_name=config.planner_llm_model,
            timeout=config.vlm_timeout,
        )
    if config.fallback_vlm_base_url:
        context.fallback_vlm = OpenAIChatGPTVLMClient(
            base_url=config.fallback_vlm_base_url,
            api_key=config.fallback_vlm_api_key,
            model_name=config.fallback_vlm_model,
            timeout=config.vlm_timeout,
        )
    return context


def build_camera(config):
    """Return an unstarted camera service and the parameters used to cache it."""
    if getattr(config, "hardware_config_path", None):
        from hardware.runtime import camera_service
        return camera_service(config.hardware_config_path, config.capture_dir)
    from services.realsense_dual_camera_service import CameraConfig, RealSenseDualCameraService
    from services.runtime_service_cache import get_or_create_service

    params = {
        "global_serial": config.global_serial,
        "wrist_serial": config.wrist_serial,
        "capture_dir": config.capture_dir,
        "show_window": True,
        "global_color_width": 1280,
        "global_color_height": 720,
        "wrist_color_width": 1280,
        "wrist_color_height": 720,
        "wrist_depth_width": 1280,
        "wrist_depth_height": 720,
    }
    camera = get_or_create_service(
        "dual_realsense_camera",
        params,
        lambda: RealSenseDualCameraService(
            global_config=CameraConfig(
                name="global",
                serial=config.global_serial,
                color_width=1280,
                color_height=720,
                color_fps=30,
                enable_depth=False,
            ),
            wrist_config=CameraConfig(
                name="wrist",
                serial=config.wrist_serial,
                color_width=1280,
                color_height=720,
                color_fps=30,
                enable_depth=True,
                depth_width=1280,
                depth_height=720,
                depth_fps=30,
            ),
            save_dir=config.capture_dir,
            show_window=True,
        ),
    )
    return camera, params


def build_execution_context(config, *, camera, event_callback=None):
    """Return a skill context and its underlying camera, without starting capture.

    The caller owns the camera lifecycle. Model initialization and service-cache
    behavior match the original agent/runtime assembly.
    """
    cached_camera = TaskCachedCamera(camera)

    from services.grasp_service import build_grasp_service, build_segmenter, detector_model_path
    from services.grounding_dino_detector_service import GroundingDINODetectorService
    from services.recognition_runtime import build_lazy_ocr
    from services.object_reference_service import build_reference_service
    from services.runtime_service_cache import get_or_create_service

    detector_params = {
        "model_name_or_path": detector_model_path(config),
        "device": config.detector_device,
        "threshold": 0.22,
        "text_threshold": 0.18,
        "nms_iou_threshold": 0.50,
        "local_files_only": True,
        "use_fp16": False,
    }
    detector = get_or_create_service(
        "grounding_dino",
        detector_params,
        lambda: GroundingDINODetectorService(**detector_params),
    )
    ocr_params = {"lang": "ch", "device": "cpu", "use_textline_orientation": True}
    ocr = build_lazy_ocr(ocr_params)
    anygrasp = build_grasp_service(config)
    workers = getattr(camera, 'perception_workers', 1)
    if type(workers) is not int:
        workers = 1
    return (
        RuntimeContext(
            camera=cached_camera,
            detector=detector,
            ocr=ocr,
            object_references=build_reference_service(config),
            anygrasp=anygrasp,
            segmenter=build_segmenter(config),
            event_callback=event_callback,
            perception_workers=workers,
        ),
        camera,
    )
