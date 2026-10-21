"""Select a working EGL device without modifying host graphics drivers."""

from __future__ import annotations

import os
import sys


def configure_renderer():
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ["MUJOCO_GL"] != "egl":
        return
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") is not None or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return  # Respect an explicitly selected device.
    from mujoco.egl import egl_ext as egl
    from OpenGL import error

    # robosuite picks EGL device 0 without trying the rest. A host may expose
    # an unusable device 0 and a working Mesa device 1. Probe as native MuJoCo does.
    for index, device in enumerate(egl.eglQueryDevicesEXT()):
        display = egl.eglGetPlatformDisplayEXT(egl.EGL_PLATFORM_DEVICE_EXT, device, None)
        if display == egl.EGL_NO_DISPLAY:
            continue
        try:
            initialized = egl.eglInitialize(display, None, None)
        except error.GLError:
            continue
        if initialized:
            vendor = egl.eglQueryString(display, egl.EGL_VENDOR).decode()
            egl.eglTerminate(display)
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(index)
            print(f"Selected EGL device {index}: {vendor}", file=sys.stderr)
            return
    raise RuntimeError("No working EGL device; try MUJOCO_GL=osmesa or a working OpenGL desktop.")


def renderer_info():
    from OpenGL.GL import glGetString, glGetError, GL_NO_ERROR, GL_RENDERER, GL_VENDOR, GL_VERSION
    # The robosuite render pass can leave a pending GL error unrelated to these queries.
    while glGetError() != GL_NO_ERROR:
        pass
    return {"backend": os.environ.get("MUJOCO_GL"),
            "egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            **{name: (glGetString(key) or b"unknown").decode()
               for name, key in [("vendor", GL_VENDOR), ("renderer", GL_RENDERER), ("version", GL_VERSION)]}}
